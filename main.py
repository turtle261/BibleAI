import json
import faiss
import numpy as np
import asyncio
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import uvicorn
import logging
import os
import re
import requests
import threading
import io
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import bz2
import lzma
import zlib
import numpy as np
import torch
from enum import Enum

from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from config import settings

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget, QComboBox, 
    QCheckBox, QMessageBox, QProgressBar, QSplitter, QFrame, QStyleFactory,
    QScrollArea, QStatusBar, QDialog, QDialogButtonBox, QTextBrowser, QMenuBar, QAction, QMenu, QSpinBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, QRunnable, QThreadPool, pyqtSignal, QObject, pyqtSlot, QSize, QUrl
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QTextCursor, QDesktopServices, QPixmap

# Ensure data directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ===== Logging Setup =====
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"bible_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('bmh')  
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelcolor': '#2b5b84',
    'axes.titlecolor': '#2b5b84',
    'xtick.color': '#2b5b84',
    'ytick.color': '#2b5b84'
})

class Query(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    include_greek: bool = False
    analyze_complexity: bool = False

class VerseResult(BaseModel):
    book: str
    chapter: int
    verse: int
    text: str
    greek_text: Optional[str] = None
    complexity_score: Optional[float] = None
    similar_by_complexity: Optional[List[Dict[str, Any]]] = None
    complexities_by_language: Optional[Dict[str, float]] = None

class ResponseResult(BaseModel):
    answer: str
    verses: List[VerseResult]
    conversation_id: str
    thoughts: List[str]

class ToolType(Enum):
    SEMANTIC_SEARCH = "semantic_search"
    VERSE_LOOKUP = "verse_lookup"
    GREEK_ANALYSIS = "greek_analysis"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    RESPONSE_GENERATION = "response_generation"
    REASONING = "reasoning"
    CONTINUE_REASONING = "continue_reasoning"

class OllamaAPI:
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL):
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.embeddings_url = f"{base_url}/api/embeddings"
        self.session = requests.Session()
        self.session.timeout = settings.OLLAMA_TIMEOUT
        
    async def generate(self, model: str, prompt: str, system: Optional[str] = None) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system
            
        def sync_post():
            return self.session.post(self.generate_url, json=payload)
        
        try:
            response = await asyncio.to_thread(sync_post)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

    async def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings_batch = []
            for text in batch:
                payload = {
                    "model": settings.EMBEDDING_MODEL,
                    "prompt": text
                }
                def sync_post():
                    return self.session.post(self.embeddings_url, json=payload)
                try:
                    response = await asyncio.to_thread(sync_post)
                    response.raise_for_status()
                    embedding = response.json().get("embedding")
                    if not embedding:
                        raise ValueError("No embedding returned")
                    if len(embedding) != settings.EMBEDDING_DIMENSION:
                        raise ValueError("Embedding dimension mismatch")
                    embeddings_batch.append(embedding)
                except Exception as e:
                    logger.error(f"Error getting embeddings: {str(e)}")
                    embeddings_batch.append([0.0] * settings.EMBEDDING_DIMENSION)
            all_embeddings.extend(embeddings_batch)
        return all_embeddings

class ExecutionPlan:
    def __init__(self, steps: List[Dict[str, Any]], total_weight: int = 100):
        self.steps = steps
        self.current_step = 0
        self.total_weight = total_weight
        self.step_weights = self._calculate_step_weights()
        self.progress = 0.0
        self.thoughts = []
        self.completed_steps = set()
        self.lock = threading.Lock()

    def _calculate_step_weights(self) -> List[int]:
        total_declared_weight = sum(s.get('weight', 1) for s in self.steps)
        weights = []
        remaining_weight = self.total_weight
        for step in self.steps:
            step_weight = round((step.get('weight', 1)/total_declared_weight)*self.total_weight)
            weights.append(step_weight)
            remaining_weight -= step_weight
        if remaining_weight > 0:
            weights[-1] += remaining_weight
        return weights

    def add_thought(self, thought: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_thought = f"[{timestamp}] {thought}"
        self.thoughts.append(formatted_thought)
        logger.info(f"Thought: {formatted_thought}")

    def update_progress(self, step_progress: float):
        with self.lock:
            if self.current_step <= len(self.steps):
                completed_weight = sum(self.step_weights[:self.current_step])
                if self.current_step > 0:
                    current_step_contribution = self.step_weights[self.current_step - 1]*step_progress
                else:
                    current_step_contribution = 0
                self.progress = (completed_weight+current_step_contribution)/self.total_weight*100

    def next_step(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            if self.current_step < len(self.steps):
                step = self.steps[self.current_step]
                self.current_step += 1
                return step
            return None

    def mark_step_complete(self, step_type: str):
        self.completed_steps.add(step_type)
        self.add_thought(f"Completed step: {step_type}")

class KolmogorovAnalyzer:
    def __init__(self):
        self.complexity_cache = {}

    def calculate_complexity_for_language(self, text: str, lang: str) -> float:
        if not text:
            return 0.0
        cache_key = (lang, text)
        if cache_key in self.complexity_cache:
            return self.complexity_cache[cache_key]
        c = self._calculate_complexity(text)
        self.complexity_cache[cache_key] = c
        return c
    
    def _calculate_complexity(self, text: str) -> float:
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return 0.0
        original_len = len(normalized_text.encode('utf-8'))
        if original_len == 0:
            return 0.0

        zlib_ratio = len(zlib.compress(normalized_text.encode('utf-8')))/original_len
        bz2_ratio = len(bz2.compress(normalized_text.encode('utf-8')))/original_len
        lzma_ratio = len(lzma.compress(normalized_text.encode('utf-8')))/original_len

        char_entropy = self._calculate_entropy(normalized_text)
        normalized_entropy = char_entropy/8.0

        word_complexity = self._calculate_word_complexity(normalized_text)

        complexities = [zlib_ratio, bz2_ratio, lzma_ratio, normalized_entropy, word_complexity]
        weights = [0.166,0.166,0.166,0.3,0.2]
        combined = sum(c*w for c,w in zip(complexities, weights))
        combined = max(0.0, min(1.0, combined))
        return combined

    def _normalize_text(self, text: str) -> str:
        text = ' '.join(text.lower().split())
        text = re.sub(r'[^\w\s.]','',text)
        return text

    def _calculate_entropy(self, text: str) -> float:
        freq = {}
        for c in text:
            freq[c]=freq.get(c,0)+1
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count/length
            entropy -= p*np.log2(p)
        return entropy

    def _calculate_word_complexity(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        avg_word_length = sum(len(w) for w in words)/len(words)
        unique_ratio = len(set(words))/len(words)
        return (avg_word_length/15.0 + unique_ratio)/2.0

    def find_most_similar_verse(self, verse: dict, all_verses: List[dict], greek_data: dict) -> Tuple[List[dict], List[dict]]:
        target_eng = self.calculate_complexity_for_language(verse['text'], 'en')
        target_greek = None
        gr_text = greek_data.get(verse['book'], {}).get(str(verse['chapter']),{}).get(str(verse['verse']))
        if gr_text:
            target_greek = self.calculate_complexity_for_language(gr_text, 'gr')
        candidates=[]
        for v in all_verses:
            if v['book']==verse['book'] and v['chapter']==verse['chapter'] and v['verse']==verse['verse']:
                continue
            eng_complexity = self.calculate_complexity_for_language(v['text'], 'en')
            eng_diff = abs(eng_complexity-target_eng)
            if target_greek is not None:
                v_gr_text = greek_data.get(v['book'], {}).get(str(v['chapter']),{}).get(str(v['verse']))
                if v_gr_text:
                    gr_complexity = self.calculate_complexity_for_language(v_gr_text, 'gr')
                    gr_diff = abs(gr_complexity - target_greek)
                    combined_diff = (eng_diff+gr_diff)/2.0
                    candidates.append((combined_diff,eng_diff,gr_diff,v))
                else:
                    candidates.append((eng_diff,eng_diff,None,v))
            else:
                candidates.append((eng_diff,eng_diff,None,v))
        candidates.sort(key=lambda x:x[0])
        top_matches = candidates[:2]
        eng_similar = [m[3] for m in top_matches]
        greek_candidates = [m for m in candidates if m[2] is not None]
        if greek_candidates:
            greek_candidates.sort(key=lambda x:x[0])
            greek_similar = [m[3] for m in greek_candidates[:2]]
        else:
            greek_similar=[]
        return eng_similar, greek_similar

class VerseScoringSystem:
    def calculate_verse_score(self, verse: dict, semantic_score: float) -> float:
        verse_position_weight = 1.0/(1.0+0.01*verse['verse'])
        return semantic_score*verse_position_weight
        
    def get_context_verses(self, verse: dict, all_verses: List[dict], window=2) -> List[dict]:
        context_verses=[]
        for v in all_verses:
            if v['book']==verse['book'] and v['chapter']==verse['chapter'] and abs(v['verse']-verse['verse'])<=window and v['verse']!=verse['verse']:
                context_verses.append(v)
        return context_verses

class BibleAgent:
    def __init__(self):
        self.model_name = settings.MODEL_NAME
        self.ollama = OllamaAPI()
        self.conversations = {}
        self.complexity_analyzer = KolmogorovAnalyzer()
        self.verse_scorer = VerseScoringSystem()
        self.progress_lock = threading.Lock()

        self.book_names={
            'gn':'Genesis','ex':'Exodus','lv':'Leviticus','nm':'Numbers',
            'dt':'Deuteronomy','js':'Joshua','jud':'Judges','rt':'Ruth',
            '1sm':'1 Samuel','2sm':'2 Samuel','1kgs':'1 Kings','2kgs':'2 Kings',
            '1ch':'1 Chronicles','2ch':'2 Chronicles','ezr':'Ezra','ne':'Nehemiah',
            'et':'Esther','job':'Job','ps':'Psalms','prv':'Proverbs',
            'ec':'Ecclesiastes','so':'Song of Solomon','is':'Isaiah',
            'jr':'Jeremiah','lm':'Lamentations','ez':'Ezekiel','dn':'Daniel',
            'ho':'Hosea','jl':'Joel','am':'Amos','ob':'Obadiah','jn':'Jonah',
            'mi':'Micah','na':'Nahum','hk':'Habakkuk','zp':'Zephaniah',
            'hg':'Haggai','zc':'Zechariah','ml':'Malachi','mt':'Matthew',
            'mk':'Mark','lk':'Luke','jo':'John','act':'Acts','rm':'Romans',
            '1co':'1 Corinthians','2co':'2 Corinthians','gl':'Galatians',
            'eph':'Ephesians','ph':'Philippians','cl':'Colossians',
            '1ts':'1 Thessalonians','2ts':'2 Thessalonians','1tm':'1 Timothy',
            '2tm':'2 Timothy','tt':'Titus','phm':'Philemon','hb':'Hebrews',
            'jm':'James','1pe':'1 Peter','2pe':'2 Peter','1jo':'1 John',
            '2jo':'2 John','3jo':'3 John','jd':'Jude','re':'Revelation'
        }
        self.index=None
        self.bible_data=[]
        self.greek_data={}
        self.translations_data={}
        self._initialized=False
        self._initializing=False

    async def _extract_themes(self, text: str) -> str:
        system_prompt = (
            "You are a Bible study assistant. Extract the main biblical themes and concepts from the given user query.\n"
            "Return only the key themes, separated by commas."
        )
        try:
            themes = await self.ollama.generate(
                model=self.model_name,
                prompt=f"Extract key biblical themes from: {text}",
                system=system_prompt
            )
            themes_str = themes.strip()
            if themes_str:
                return themes_str
            else:
                return self._basic_theme_extraction(text)
        except:
            return self._basic_theme_extraction(text)

    def _basic_theme_extraction(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        themes = [w for w in words if w not in stop_words and len(w) > 2]
        return ', '.join(set(themes))

    async def _create_dynamic_plan(self, question: str, include_greek: bool, analyze_complexity: bool) -> List[dict]:
        system_prompt = (
            "You are a tool selection and planning module for a Bible study assistant. "
            "Tools: REASONING, VERSE_LOOKUP, SEMANTIC_SEARCH, GREEK_ANALYSIS, COMPLEXITY_ANALYSIS, RESPONSE_GENERATION, enhance_verses. "
            "Decide steps to answer. Always start with REASONING. If verse reference is mentioned, include VERSE_LOOKUP. "
            "Use SEMANTIC_SEARCH to find verses. If greek/complexity requested, use enhance_verses. "
            "End with RESPONSE_GENERATION. Return only JSON array of steps."
        )

        prompt = f"Create a plan for this question: {question}\ninclude_greek={include_greek}, analyze_complexity={analyze_complexity}\nReturn only a JSON array."

        try:
            response = await self.ollama.generate(model=self.model_name, prompt=prompt, system=system_prompt)
            plan_str = response.strip()
            plan_str = re.sub(r'^```json\s*|\s*```$', '', plan_str, flags=re.MULTILINE)
            plan_str = re.sub(r'^```\s*|\s*```$', '', plan_str, flags=re.MULTILINE)
            json_match = re.search(r'\[.*\]', plan_str, re.DOTALL)
            if json_match:
                plan_str = json_match.group(0)
            else:
                logger.error("No valid JSON array found in response.")
                raise ValueError("No valid JSON array found in response.")
            try:
                steps = json.loads(plan_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error: {e}")
                raise ValueError("Failed to decode JSON from response.")
            if not isinstance(steps, list):
                logger.error("Plan is not a list.")
                raise ValueError("Plan is not a list.")
            valid_steps = []
            for s in steps:
                if 'type' not in s:
                    logger.warning("Missing 'type' in step, skipping step.")
                    continue
                if 'weight' not in s:
                    s['weight'] = 1
                valid_steps.append(s)
            if not valid_steps:
                logger.error("No valid steps found in plan.")
                raise ValueError("No valid steps found in plan.")
            return valid_steps
        except Exception as e:
            logger.error(f"Dynamic plan creation failed: {e}. Falling back to default steps.")
            return await self._create_default_steps(question, include_greek, analyze_complexity)

    async def _create_default_steps(self, question: str, include_greek: bool, analyze_complexity: bool) -> List[dict]:
        steps = [{'type': ToolType.REASONING.value, 'weight': 1}]
        
        if self._extract_verse_reference(question):
            steps.append({'type': ToolType.VERSE_LOOKUP.value, 'weight': 2})
        
        themes = await self._extract_themes(question)
        
        steps.append({
            'type': ToolType.SEMANTIC_SEARCH.value,
            'weight': 2,
            'query': question
        })
        
        if themes:
            steps.append({
                'type': ToolType.SEMANTIC_SEARCH.value,
                'weight': 2,
                'query': themes
            })
        
        if include_greek or analyze_complexity:
            steps.append({'type': 'enhance_verses', 'weight': 3})
        
        steps.append({'type': ToolType.RESPONSE_GENERATION.value, 'weight': 2})
        
        return steps

    async def _create_execution_plan(self, question: str, include_greek: bool, analyze_complexity: bool) -> ExecutionPlan:
        steps = await self._create_default_steps(question, include_greek, analyze_complexity)
        return ExecutionPlan(steps)

    async def initialize(self):
        if self._initialized or self._initializing:
            return
        self._initializing=True
        try:
            await self._initialize_data()
            self._initialized=True
        finally:
            self._initializing=False

    async def ensure_initialized(self):
        if not self._initialized:
            await self.initialize()

    async def _initialize_data(self):
        await asyncio.gather(
            self._load_bible_data(),
            self._load_greek_data(),
            self._load_additional_translations(),
            self._setup_semantic_search()
        )

    async def _load_bible_data(self):
        with open(settings.BIBLE_DATA_PATH,'r',encoding='utf-8-sig') as f:
            raw_data = json.load(f)
        for book in raw_data:
            abbrev = book.get('abbrev','unknown')
            book_name=self.book_names.get(abbrev.lower(),abbrev)
            chapters=book.get('chapters',[])
            for chapter_idx,chapter in enumerate(chapters,1):
                for verse_idx,verse_text in enumerate(chapter,1):
                    clean_text=re.sub(r'\{.*?\}','',verse_text)
                    clean_text=re.sub(r'\s+',' ',clean_text).strip()
                    verse_entry={
                        'book':book_name,
                        'chapter':chapter_idx,
                        'verse':verse_idx,
                        'text':clean_text
                    }
                    self.bible_data.append(verse_entry)

    async def _load_greek_data(self):
        try:
            with open(settings.GREEK_DATA_PATH,'r',encoding='utf-8-sig') as f:
                raw_data=json.load(f)
            for book in raw_data:
                abbrev=book.get('abbrev','unknown')
                book_name=self.book_names.get(abbrev.lower(),abbrev)
                if book_name not in self.greek_data:
                    self.greek_data[book_name]={}
                chapters=book.get('chapters',[])
                for chapter_idx,chapter in enumerate(chapters,1):
                    chapter_str=str(chapter_idx)
                    if chapter_str not in self.greek_data[book_name]:
                        self.greek_data[book_name][chapter_str]={}
                    for verse_idx,verse_text in enumerate(chapter,1):
                        clean_text=re.sub(r'\s+',' ',verse_text).strip()
                        self.greek_data[book_name][chapter_str][str(verse_idx)]=clean_text
        except:
            self.greek_data={}

    async def _load_additional_translations(self):
        self.translations_data['en_bbe']=await self._load_translation_data(settings.EN_BBE_DATA_PATH)
        self.translations_data['fr_apee']=await self._load_translation_data(settings.FR_APEE_DATA_PATH)

    async def _load_translation_data(self,path:str)->Dict[str,Dict[str,Dict[str,str]]]:
        data_dict={}
        if not os.path.exists(path):
            return data_dict
        try:
            with open(path,'r',encoding='utf-8-sig') as f:
                raw_data=json.load(f)
            for book in raw_data:
                abbrev=book.get('abbrev','unknown')
                book_name=self.book_names.get(abbrev.lower(),abbrev)
                if book_name not in data_dict:
                    data_dict[book_name]={}
                chapters=book.get('chapters',[])
                for chapter_idx,chapter in enumerate(chapters,1):
                    chapter_str=str(chapter_idx)
                    if chapter_str not in data_dict[book_name]:
                        data_dict[book_name][chapter_str]={}
                    for verse_idx,verse_text in enumerate(chapter,1):
                        clean_text=re.sub(r'\s+',' ',verse_text).strip()
                        data_dict[book_name][chapter_str][str(verse_idx)]=clean_text
            return data_dict
        except:
            return data_dict

    async def _setup_semantic_search(self):
        os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH),exist_ok=True)
        if os.path.exists(settings.FAISS_INDEX_PATH):
            self.index=faiss.read_index(settings.FAISS_INDEX_PATH)
        else:
            await self._build_faiss_index()

    async def _build_faiss_index(self):
        verses = [v['text'] for v in self.bible_data]
        embeddings_list=[]
        batch_size=settings.EMBEDDING_BATCH_SIZE
        for i in range(0,len(verses),batch_size):
            batch=verses[i:i+batch_size]
            emb=await self.ollama.get_embeddings(batch,batch_size=batch_size)
            embeddings_list.extend(emb)
        embeddings=np.array(embeddings_list,dtype=np.float32)
        if embeddings.shape[1]!=settings.EMBEDDING_DIMENSION:
            raise ValueError("Embedding dimension mismatch")
        self.index=faiss.IndexFlatL2(settings.EMBEDDING_DIMENSION)
        self.index.add(embeddings)
        faiss.write_index(self.index, settings.FAISS_INDEX_PATH)

    async def process_query(self, question:str, conversation_id=None, include_greek=False, analyze_complexity=False):
        if not conversation_id:
            conversation_id="conv_"+datetime.now().strftime("%Y%m%d_%H%M%S")

        plan=await self._create_execution_plan(question,include_greek,analyze_complexity)
        verses=[]
        enhanced_verses=[]
        answer="No answer generated."

        while (step:=plan.next_step()) is not None:
            step_type=step['type']
            if step_type==ToolType.REASONING.value:
                plan.add_thought("Reasoning about the query and deciding on approach...")
                plan.update_progress(1.0)
                plan.mark_step_complete(step_type)
            elif step_type==ToolType.VERSE_LOOKUP.value:
                plan.add_thought("Looking up a specific verse reference...")
                verse_ref=self._extract_verse_reference(question)
                if verse_ref:
                    book,chapter,verse_num=verse_ref
                    direct_verse=self._lookup_verse(book,chapter,verse_num)
                    if direct_verse:
                        verses.append(direct_verse)
                plan.update_progress(1.0)
                plan.mark_step_complete(step_type)
            elif step_type==ToolType.SEMANTIC_SEARCH.value:
                plan.add_thought("Performing semantic search...")
                query_str = step.get('query', question)
                semantic_verses=await self._semantic_search(query_str)
                verses.extend(semantic_verses)
                plan.update_progress(1.0)
                plan.mark_step_complete(step_type)
            elif step_type=='enhance_verses':
                plan.add_thought("Enhancing verses...")
                if not verses:
                    plan.update_progress(1.0)
                    plan.mark_step_complete(step_type)
                    continue
                for i,v in enumerate(verses):
                    ev=v.copy()
                    if include_greek:
                        gtxt=self._get_greek_text(v)
                        if gtxt:
                            ev['greek_text']=gtxt
                    if analyze_complexity:
                        c=self.complexity_analyzer.calculate_complexity_for_language(v['text'],'en')
                        ev['complexity_score']=c
                        lang_complexities={'en':c}
                        gr_txt=self._get_greek_text(v)
                        if gr_txt:
                            lang_complexities['gr']=self.complexity_analyzer.calculate_complexity_for_language(gr_txt,'gr')
                        bbe_txt=self._get_translation_text(v,'en_bbe')
                        if bbe_txt:
                            lang_complexities['en_bbe']=self.complexity_analyzer.calculate_complexity_for_language(bbe_txt,'en_bbe')
                        fr_txt=self._get_translation_text(v,'fr_apee')
                        if fr_txt:
                            lang_complexities['fr_apee']=self.complexity_analyzer.calculate_complexity_for_language(fr_txt,'fr_apee')
                        ev['complexities_by_language']=lang_complexities
                        eng_sim,gr_sim=self.complexity_analyzer.find_most_similar_verse(v,self.bible_data,self.greek_data)
                        sim=[]
                        for engv in eng_sim:
                            sim.append({
                                'book':engv['book'],
                                'chapter':engv['chapter'],
                                'verse':engv['verse'],
                                'text':engv['text'],
                                'match_type':'english'
                            })
                        for grv in gr_sim:
                            if not any(s['book']==grv['book'] and s['chapter']==grv['chapter'] and s['verse']==grv['verse'] for s in sim):
                                sim.append({
                                    'book':grv['book'],
                                    'chapter':grv['chapter'],
                                    'verse':grv['verse'],
                                    'text':grv['text'],
                                    'match_type':'greek'
                                })
                        ev['similar_by_complexity']=sim
                    enhanced_verses.append(ev)
                    plan.update_progress((i+1)/len(verses))
                plan.mark_step_complete(step_type)
            elif step_type==ToolType.RESPONSE_GENERATION.value:
                plan.add_thought("Generating final answer...")
                final_verses=enhanced_verses if enhanced_verses else verses
                answer=await self._generate_response(question,final_verses,plan)
                plan.update_progress(1.0)
                plan.mark_step_complete(step_type)

        self._update_conversation(conversation_id,question,answer,enhanced_verses if enhanced_verses else verses,plan)
        return answer,(enhanced_verses if enhanced_verses else verses),conversation_id,plan.thoughts

    async def _generate_response(self, question: str, verses: List[dict], plan: ExecutionPlan) -> str:
        system_prompt = (
            "You are a knowledgeable Bible study assistant. Be accurate, contextual, and helpful. "
            "Consider all the information gathered. Explain how different pieces connect."
        )
        
        verses_by_score = sorted(
            verses,
            key=lambda v: self.verse_scorer.calculate_verse_score(v, v.get('score', 1.0)),
            reverse=True
        )
        
        verses_text = "\n".join([
            f"{v['book']} {v['chapter']}:{v['verse']} - {v['text']}"
            + (f"\nGreek: {v.get('greek_text', '')}" if v.get('greek_text') else "")
            + (f"\nComplexity: {v.get('complexity_score', '')}" if v.get('complexity_score') is not None else "")
            + (f"\nLangComplexities: {v.get('complexities_by_language', '')}" if 'complexities_by_language' in v else "")
            for v in verses_by_score
        ])
        
        reasoning_history = "\n".join(plan.thoughts)
        
        prompt = (
            f"Question: {question}\n\n"
            f"Reasoning Process:\n{reasoning_history}\n\n"
            f"Relevant Verses:\n{verses_text}\n\n"
            "Provide a comprehensive answer addressing the question."
        )
        
        plan.add_thought("Generating final response with comprehensive analysis...")
        return await self.ollama.generate(
            model=self.model_name,
            prompt=prompt,
            system=system_prompt
        )

    def _extract_verse_reference(self, question: str) -> Optional[Tuple[str,int,int]]:
        pattern=r'(?i)([1-3]?\s?[A-Za-z]+)\s+(\d+):(\d+)'
        match=re.search(pattern,question)
        if match:
            try:
                book=match.group(1).strip()
                chapter=int(match.group(2))
                verse=int(match.group(3))
                return (book,chapter,verse)
            except:
                return None
        return None

    def _lookup_verse(self, book: str, chapter:int, verse:int)->Optional[dict]:
        for v in self.bible_data:
            if v['book'].lower()==book.lower() and v['chapter']==chapter and v['verse']==verse:
                return v
        return None

    def _update_conversation(self, conv_id:str, question:str, answer:str, verses:List[dict], plan:ExecutionPlan):
        if conv_id not in self.conversations:
            self.conversations[conv_id]=[]
        entry={
            "timestamp":datetime.now().isoformat(),
            "question":question,
            "answer":answer,
            "verses":verses,
            "tools_used":list(plan.completed_steps),
            "thoughts":plan.thoughts
        }
        self.conversations[conv_id].append(entry)

    async def _semantic_search(self, query:str, top_k:int=None)->List[dict]:
        if top_k is None:
            top_k=settings.TOP_K_RESULTS
        query_embedding=await self.ollama.get_embeddings([query])
        if not query_embedding or len(query_embedding[0])!=settings.EMBEDDING_DIMENSION:
            raise ValueError("Invalid query embedding dimension")
        query_vector=np.array(query_embedding,dtype=np.float32)
        distances,indices=self.index.search(query_vector,top_k*2)
        scored_verses=[]
        for idx,distance in zip(indices[0],distances[0]):
            if idx<len(self.bible_data):
                verse=self.bible_data[idx].copy()
                semantic_score=1.0/(1.0+distance)
                verse['score']=semantic_score
                scored_verses.append(verse)
        scored_verses.sort(key=lambda x:x['score'],reverse=True)
        top_verses=scored_verses[:top_k]
        final_verses=[]
        seen=set()
        for verse in top_verses:
            key=(verse['book'],verse['chapter'],verse['verse'])
            if key not in seen:
                final_verses.append(verse)
                seen.add(key)
                context=self.verse_scorer.get_context_verses(verse,self.bible_data)
                for ctx in context:
                    ctx_key=(ctx['book'],ctx['chapter'],ctx['verse'])
                    if ctx_key not in seen:
                        final_verses.append(ctx)
                        seen.add(ctx_key)
        return final_verses

    def _get_greek_text(self, verse:dict)->Optional[str]:
        return self.greek_data.get(verse['book'],{}).get(str(verse['chapter']),{}).get(str(verse['verse']))

    def _get_translation_text(self, verse:dict, lang_key:str)->Optional[str]:
        data=self.translations_data.get(lang_key,{})
        return data.get(verse['book'],{}).get(str(verse['chapter']),{}).get(str(verse['verse']))

agent=BibleAgent()


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await agent.initialize()
    yield
    # Shutdown
    pass

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_server():
    try:
        base_port = settings.PORT
        max_retries = 5
        current_port = base_port
        
        while max_retries > 0:
            try:
                logger.info(f"Attempting to start server on port {current_port}...")
                config = uvicorn.Config(
                    app=app,
                    host=settings.HOST,
                    port=current_port,
                    log_level="info",
                    access_log=False
                )
                server = uvicorn.Server(config)
                server.run()
                break
            except OSError as e:
                if e.errno:
                    logger.warning(f"Port {current_port} is in use, trying next port...")
                    current_port += 1
                    max_retries -= 1
                else:
                    logger.error(f"Server error: {str(e)}")
                    raise
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


@app.post("/query",response_model=ResponseResult)
async def query_bible_ai(q: Query):
    try:
        await agent.ensure_initialized()
        answer,verses,conv_id,thoughts=await agent.process_query(
            q.question,
            q.conversation_id,
            q.include_greek,
            q.analyze_complexity
        )
        verse_results=[]
        for v in verses:
            vd={
                "book":v["book"],
                "chapter":v["chapter"],
                "verse":v["verse"],
                "text":v["text"]
            }
            if "greek_text" in v:
                vd["greek_text"]=v["greek_text"]
            if "complexity_score" in v:
                vd["complexity_score"]=v["complexity_score"]
                vd["similar_by_complexity"]=v.get("similar_by_complexity",[])
            if "complexities_by_language" in v:
                vd["complexities_by_language"]=v["complexities_by_language"]
            verse_results.append(VerseResult(**vd))
        return ResponseResult(answer=answer,verses=verse_results,conversation_id=conv_id,thoughts=thoughts)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500,detail=str(e))

@app.get("/complexity_graph")
async def complexity_graph(book: str, chapter: int, verse: int):
    await agent.ensure_initialized()
    try:
        v = agent._lookup_verse(book, chapter, verse)
        if not v:
            raise HTTPException(status_code=404, detail="Verse not found")
        
        complexities = {}
        complexities['en'] = agent.complexity_analyzer.calculate_complexity_for_language(v['text'], 'en')
        
        gr_txt = agent._get_greek_text(v)
        if gr_txt:
            complexities['gr'] = agent.complexity_analyzer.calculate_complexity_for_language(gr_txt, 'gr')
        
        bbe_txt = agent._get_translation_text(v, 'en_bbe')
        if bbe_txt:
            complexities['en_bbe'] = agent.complexity_analyzer.calculate_complexity_for_language(bbe_txt, 'en_bbe')
        
        fr_txt = agent._get_translation_text(v, 'fr_apee')
        if fr_txt:
            complexities['fr_apee'] = agent.complexity_analyzer.calculate_complexity_for_language(fr_txt, 'fr_apee')

        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 6))
        
        langs = list(complexities.keys())
        values = [complexities[l] for l in langs]
        
        colors = ['#2b5b84', '#4682b4', '#87ceeb', '#b0c4de']
        bars = ax.bar(langs, values, color=colors[:len(langs)])
        
        ax.set_title(f"Complexity Analysis for {book} {chapter}:{verse}", 
                    fontsize=14, pad=20, color='#2b5b84')
        ax.set_ylabel("Complexity Score (0-1)", color='#2b5b84')
        ax.set_ylim(0, 1.1)
        
        for i, v_val in enumerate(values):
            ax.text(i, v_val + 0.02, f'{v_val:.3f}', 
                   ha='center', va='bottom',
                   fontweight='bold', color='#2b5b84')
        
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        for spine in ax.spines.values():
            spine.set_color('#2b5b84')
            spine.set_linewidth(0.5)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close('all')
        
        return Response(content=buf.getvalue(), media_type="image/png")
    
    except Exception as e:
        logger.error(f"Error generating complexity graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating graph: {str(e)}")

@app.get("/progress")
async def get_progress():
    return {"progress":0.0}

# ===== GUI Integration =====
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import QThreadPool

class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn=fn
        self.args=args
        self.kwargs=kwargs
        self.signals=WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result=self.fn(*self.args,**self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))

class CustomTextEdit(QTextEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.setReadOnly(True)
        self.setAcceptRichText(True)
        self.setTextInteractionFlags(
            Qt.TextSelectableByMouse | 
            Qt.TextSelectableByKeyboard | 
            Qt.LinksAccessibleByMouse | 
            Qt.LinksAccessibleByKeyboard
        )
        self.setStyleSheet("""
            QTextEdit {
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
                background-color: palette(base);
                selection-background-color: #667;
            }
        """)
        font = self.font()
        font.setFamily("Arial")
        self.setFont(font)

    def _markdown_to_html(self, text: str) -> str:
        text = re.sub(r'^# (.*?)$', r'<h1 style="color: #2b5b84;">\1</h1>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', r'<h2 style="color: #2b5b84;">\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.*?)$', r'<h3 style="color: #2b5b84;">\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b style="color: #2b5b84;">\1</b>', text)
        text = re.sub(r'__(.*?)__', r'<b style="color: #2b5b84;">\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i style="color: #667;">\1</i>', text)
        text = re.sub(r'_(.*?)_', r'<i style="color: #666;">\1</i>', text)
        text = re.sub(r'`(.*?)`', r'<code style="background-color: rgba(43, 91, 132, 0.1); padding: 2px 4px;border-radius:3px;font-family:monospace;">\1</code>', text)
        text = re.sub(r'^\* (.*?)$', r'<ul><li>\1</li></ul>', text, flags=re.MULTILINE)
        text = re.sub(r'^\d\. (.*?)$', r'<ol><li>\1</li></ol>', text, flags=re.MULTILINE)
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" style="color: #2b5b84;text-decoration:none;border-bottom:1px solid #2b5b84;">\1</a>', text)
        text = re.sub(r'^> (.*?)$', r'<blockquote style="border-left:3px solid #2b5b84;padding:10px;background-color:rgba(43,91,132,0.05);">\1</blockquote>', text, flags=re.MULTILINE)
        return text

class CustomLineEdit(QLineEdit):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.setStyleSheet("""
            QLineEdit {
                border:1px solid #555;
                border-radius:5px;
                padding:8px;
                background-color:palette(base);
                selection-background-color:#666;
            }
            QLineEdit:focus {
                border:1px solid #888;
            }
        """)

class CustomButton(QPushButton):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.setStyleSheet("""
            QPushButton {
                background-color:#2b5b84;
                color:white;
                border:none;
                padding:8px 15px;
                border-radius:5px;
            }
            QPushButton:hover {
                background-color:#3d7ab8;
            }
            QPushButton:pressed {
                background-color:#1d4d74;
            }
            QPushButton:disabled {
                background-color:#555;color:#888;
            }
        """)

class CustomCheckBox(QCheckBox):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.setStyleSheet("""
            QCheckBox {spacing:8px;}
            QCheckBox::indicator {
                width:18px;height:18px;border-radius:3px;
            }
            QCheckBox::indicator:unchecked {
                border:2px solid #555;
                background-color:palette(base);
            }
            QCheckBox::indicator:checked {
                background-color:#2b5b84;
                border:2px solid #2b5b84;
            }
        """)

class ImageDialog(QDialog):
    def __init__(self, image_data: bytes, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        
        label = QLabel(self)
        pixmap = QPixmap()
        if pixmap.loadFromData(image_data):
            screen_size = QApplication.primaryScreen().size()
            max_width = int(screen_size.width() * 0.8)
            max_height = int(screen_size.height() * 0.8)
            
            if pixmap.width() > max_width or pixmap.height() > max_height:
                pixmap = pixmap.scaled(max_width, max_height, 
                                     Qt.KeepAspectRatio, 
                                     Qt.SmoothTransformation)
            
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
        else:
            label.setText("Failed to load image")
            label.setStyleSheet("color: red;")
        
        scroll.setWidget(label)
        layout.addWidget(scroll)
        
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save Image")
        save_button.clicked.connect(lambda: self.save_image(image_data))
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        self.adjustSize()
    
    def save_image(self, image_data: bytes):
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", 
                "PNG Files (*.png);;All Files (*)"
            )
            if file_name:
                if not file_name.lower().endswith('.png'):
                    file_name += '.png'
                with open(file_name, 'wb') as f:
                    f.write(image_data)
                QMessageBox.information(self, "Success", "Image saved successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save image: {str(e)}")

class CustomTextBrowser(QTextBrowser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.setOpenExternalLinks(True)
        self.setStyleSheet("""
            QTextBrowser {
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
                background-color: palette(base);
                selection-background-color: #666;
            }
        """)
        font = self.font()
        font.setFamily("Arial")
        self.setFont(font)

    def _markdown_to_html(self, text: str) -> str:
        text = re.sub(r'^# (.*?)$', r'<h1 style="color: #2b5b84;">\1</h1>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', r'<h2 style="color: #2b5b84;">\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.*?)$', r'<h3 style="color: #2b5b84;">\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b style="color: #2b5b84;">\1</b>', text)
        text = re.sub(r'__(.*?)__', r'<b style="color: #2b5b84;">\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i style="color: #666;">\1</i>', text)
        text = re.sub(r'_(.*?)_', r'<i style="color: #666;">\1</i>', text)
        text = re.sub(r'`(.*?)`', r'<code style="background-color: rgba(43, 91, 132, 0.1); padding: 2px 4px;border-radius:3px;font-family:monospace;">\1</code>', text)
        text = re.sub(r'^\* (.*?)$', r'<ul><li>\1</li></ul>', text, flags=re.MULTILINE)
        text = re.sub(r'^\d\. (.*?)$', r'<ol><li>\1</li></ol>', text, flags=re.MULTILINE)
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" style="color: #2b5b84;text-decoration:none;border-bottom:1px solid #2b5b84;">\1</a>', text)
        text = re.sub(r'^> (.*?)$', r'<blockquote style="border-left:3px solid #2b5b84;padding:10px;background-color:rgba(43,91,132,0.05);">\1</blockquote>', text, flags=re.MULTILINE)
        return text

class SettingsDialog(QDialog):
    def __init__(self, parent=None, api_url="http://localhost:8000", ollama_url=settings.OLLAMA_BASE_URL, 
                 top_k=settings.TOP_K_RESULTS, complexity_graph=False, model_name=settings.MODEL_NAME,
                 theme="System", font_size="Medium"):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.api_url = api_url
        self.ollama_url = ollama_url
        self.top_k = top_k
        self.complexity_graph = complexity_graph
        self.model_name = model_name
        self.theme = theme
        self.font_size = font_size
        layout = QVBoxLayout(self)

        h_theme = QHBoxLayout()
        h_theme.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["System", "Light", "Dark"])
        self.theme_combo.setCurrentText(self.theme)
        h_theme.addWidget(self.theme_combo)
        layout.addLayout(h_theme)

        h_font = QHBoxLayout()
        h_font.addWidget(QLabel("Font Size:"))
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Small", "Medium", "Large"])
        self.font_combo.setCurrentText(self.font_size)
        h_font.addWidget(self.font_combo)
        layout.addLayout(h_font)

        h_ollama = QHBoxLayout()
        h_ollama.addWidget(QLabel("Ollama URL:"))
        self.ollama_url_edit = QLineEdit(self.ollama_url)
        h_ollama.addWidget(self.ollama_url_edit)
        layout.addLayout(h_ollama)

        h_topk = QHBoxLayout()
        h_topk.addWidget(QLabel("Top K Results:"))
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 50)
        self.topk_spin.setValue(self.top_k)
        h_topk.addWidget(self.topk_spin)
        layout.addLayout(h_topk)

        self.comp_graph_check = QCheckBox("Enable Complexity Analysis/Graph")
        self.comp_graph_check.setChecked(self.complexity_graph)
        layout.addWidget(self.comp_graph_check)

        h_models = QHBoxLayout()
        h_models.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem(self.model_name)
        h_models.addWidget(self.model_combo)
        layout.addLayout(h_models)

        h_api = QHBoxLayout()
        h_api.addWidget(QLabel("API URL:"))
        self.api_url_edit = QLineEdit(self.api_url)
        h_api.addWidget(self.api_url_edit)
        layout.addLayout(h_api)

        self.test_button = QPushButton("Test Ollama Connection")
        self.test_button.clicked.connect(self.test_ollama_connection)
        layout.addWidget(self.test_button)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.save)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

        self.fetch_models()

    def fetch_models(self):
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                models = []
                for model in data.get("models", []):
                    name = model.get("name", "")
                    if name:
                        base_name = name.split(":")[0] if ":" in name else name
                        if base_name not in models:
                            models.append(base_name)
                
                if models:
                    self.model_combo.clear()
                    self.model_combo.addItems(sorted(models))
                    current_base = self.model_name.split(":")[0] if ":" in self.model_name else self.model_name
                    if current_base in models:
                        self.model_combo.setCurrentText(current_base)
                    self.status_label.setText("Models loaded successfully")
                    self.status_label.setStyleSheet("color: green;")
                else:
                    self.status_label.setText("No models found")
                    self.status_label.setStyleSheet("color: orange;")
            else:
                self.status_label.setText("Failed to fetch models")
                self.status_label.setStyleSheet("color: red;")
        except Exception as e:
            self.status_label.setText(f"Error fetching models: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def test_ollama_connection(self):
        try:
            r = requests.get(f"{self.ollama_url_edit.text().strip()}/api/tags", timeout=5)
            if r.status_code == 200:
                QMessageBox.information(self, "Success", "Successfully connected to Ollama server")
                self.ollama_url = self.ollama_url_edit.text().strip()
                self.fetch_models()
            else:
                QMessageBox.warning(self, "Error", f"Failed to connect to Ollama server: {r.status_code}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect to Ollama server: {str(e)}")

    def save(self):
        api_url = self.api_url_edit.text().strip()
        if not api_url.startswith(("http://", "https://")):
            QMessageBox.warning(self, "Invalid URL", "API URL must start with http:// or https://")
            return

        ollama_url = self.ollama_url_edit.text().strip()
        if not ollama_url.startswith(("http://", "https://")):
            QMessageBox.warning(self, "Invalid URL", "Ollama URL must start with http:// or https://")
            return

        self.api_url = api_url
        self.ollama_url = ollama_url
        self.top_k = self.topk_spin.value()
        self.complexity_graph = self.comp_graph_check.isChecked()
        self.model_name = self.model_combo.currentText()
        self.theme = self.theme_combo.currentText()
        self.font_size = self.font_combo.currentText()
        self.accept()

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About")
        layout = QVBoxLayout(self)
        about_label = QLabel("This is the Bible AI Study Assistant, leveraging advanced models.")
        about_label.setWordWrap(True)
        layout.addWidget(about_label)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.accept)
        layout.addWidget(buttonBox)

class BibleAIGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.conversation_id = None
        self.setWindowTitle("Bible AI - Intelligent Bible Study Assistant")
        self.setGeometry(100, 100, 1200, 800)
        
        self.api_url = f"http://{settings.HOST}:{settings.PORT}"
        self.ollama_url = settings.OLLAMA_BASE_URL
        self.top_k = settings.TOP_K_RESULTS
        self.complexity_graph = False
        self.model_name = settings.MODEL_NAME
        self.theme = "System"
        self.font_size = "Medium"

        self.create_menubar()
        self.setup_ui()
        self.threadpool = QThreadPool()
        self.load_settings()
        self.apply_theme(self.theme)
        self.change_font_size()
        self.test_ollama_connection()

    def test_ollama_connection(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code != 200:
                QMessageBox.warning(
                    self,
                    "Ollama Warning",
                    "Could not connect to Ollama server. Please ensure Ollama is running."
                )
        except:
            QMessageBox.warning(
                self,
                "Ollama Warning",
                "Could not connect to Ollama server. Please ensure Ollama is running."
            )

    def create_menubar(self):
        menubar = QMenuBar()
        self.setMenuBar(menubar)

        settings_menu = menubar.addMenu("Settings")
        configure_action = QAction("Configure...", self)
        configure_action.triggered.connect(self.open_settings_dialog)
        settings_menu.addAction(configure_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.open_about_dialog)
        help_menu.addAction(about_action)

    def open_settings_dialog(self):
        dlg = SettingsDialog(
            self, 
            api_url=self.api_url, 
            ollama_url=self.ollama_url, 
            top_k=self.top_k, 
            complexity_graph=self.complexity_graph, 
            model_name=self.model_name,
            theme=self.theme,
            font_size=self.font_size
        )
        if dlg.exec_() == QDialog.Accepted:
            self.api_url = dlg.api_url
            self.ollama_url = dlg.ollama_url
            self.top_k = dlg.top_k
            self.complexity_graph = dlg.complexity_graph
            self.model_name = dlg.model_name
            self.theme = dlg.theme
            self.font_size = dlg.font_size
            self.save_settings()
            self.apply_theme(self.theme)
            self.change_font_size()

    def open_about_dialog(self):
        dlg = AboutDialog(self)
        dlg.exec_()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_frame.setStyleSheet("""
            QFrame#inputFrame {
                background-color: palette(base);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        input_layout = QVBoxLayout(input_frame)

        self.input_field = CustomLineEdit()
        self.input_field.setPlaceholderText("Enter your Bible study question here...")
        self.input_field.returnPressed.connect(self.send_query)
        
        self.send_button = CustomButton("Send Query")
        self.send_button.clicked.connect(self.send_query)

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_field, 1)
        input_row.addWidget(self.send_button)
        input_layout.addLayout(input_row)

        content_splitter = QSplitter(Qt.Horizontal)
        
        qna_widget = QWidget()
        qna_layout = QVBoxLayout(qna_widget)
        qna_layout.setContentsMargins(0, 0, 0, 0)
        
        qna_label = QLabel("Questions & Answers")
        qna_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.qna_display = CustomTextBrowser()
        self.qna_display.setOpenExternalLinks(True)
        
        qna_layout.addWidget(qna_label)
        qna_layout.addWidget(self.qna_display)
        content_splitter.addWidget(qna_widget)

        quotes_widget = QWidget()
        quotes_layout = QVBoxLayout(quotes_widget)
        quotes_layout.setContentsMargins(0, 0, 0, 0)
        
        quotes_label = QLabel("Relevant Bible Quotes")
        quotes_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.quotes_display = CustomTextBrowser()
        self.quotes_display.setOpenExternalLinks(False)
        self.quotes_display.anchorClicked.connect(self.on_anchor_clicked)
        
        quotes_layout.addWidget(quotes_label)
        quotes_layout.addWidget(self.quotes_display)
        content_splitter.addWidget(quotes_widget)

        content_splitter.setSizes([600, 600])
        content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555;
                width: 2px;
            }
        """)

        main_layout.addWidget(input_frame)
        main_layout.addWidget(content_splitter, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                border-top: 1px solid #555;
                padding: 5px;
                background-color: palette(base);
                color: palette(text);
            }
        """)

    def apply_theme(self, theme_name):
        if theme_name == "Dark":
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
            self.setPalette(dark_palette)
        elif theme_name == "Light":
            self.setPalette(self.style().standardPalette())
        else:
            self.setPalette(QApplication.style().standardPalette())

    def load_settings(self):
        try:
            with open("data/settings.json", "r") as f:
                s_data = json.load(f)
                self.theme = s_data.get("theme", "System")
                self.font_size = s_data.get("font_size", "Medium")
                self.api_url = s_data.get("api_url", self.api_url)
                self.ollama_url = s_data.get("ollama_url", self.ollama_url)
                self.top_k = s_data.get("top_k", self.top_k)
                self.complexity_graph = s_data.get("complexity_graph", self.complexity_graph)
                self.model_name = s_data.get("model_name", self.model_name)
        except:
            pass

    def save_settings(self):
        s_data = {
            "theme": self.theme,
            "font_size": self.font_size,
            "api_url": self.api_url,
            "ollama_url": self.ollama_url,
            "top_k": self.top_k,
            "complexity_graph": self.complexity_graph,
            "model_name": self.model_name
        }
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/settings.json", "w") as f:
                json.dump(s_data, f, indent=4)
            self.status_bar.showMessage("Settings saved successfully", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", "Failed to save settings.")

    def change_font_size(self):
        size_map = {"Small": 10, "Medium": 12, "Large": 14}
        size = size_map.get(self.font_size, 12)
        font = QFont()
        font.setPointSize(size)
        QApplication.setFont(font)
        self.qna_display.setFont(font)
        self.quotes_display.setFont(font)

    def send_query(self):
        question = self.input_field.text().strip()
        if not question:
            QMessageBox.warning(self, "Input Error", "Please enter a question.")
            return
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.status_bar.showMessage("Processing query...")
        include_greek = False
        analyze_complexity = self.complexity_graph

        def query_api():
            try:
                payload = {
                    "question": question,
                    "conversation_id": self.conversation_id,
                    "include_greek": include_greek,
                    "analyze_complexity": analyze_complexity
                }
                r = requests.post(f"{self.api_url}/query", json=payload)
                if r.status_code == 200:
                    return r.json()
                else:
                    error_msg = r.text
                    try:
                        error_data = r.json()
                        if 'detail' in error_data:
                            error_msg = error_data['detail']
                    except:
                        pass
                    raise Exception(f"Server error: {error_msg}")
            except requests.exceptions.ConnectionError:
                raise Exception("Could not connect to server. Please check if the server is running.")
            except Exception as e:
                raise Exception(f"Error processing query: {str(e)}")

        worker = Worker(query_api)
        worker.signals.finished.connect(self.on_query_finished)
        worker.signals.error.connect(self.on_query_error)
        self.threadpool.start(worker)

    def on_query_finished(self, result):
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.status_bar.showMessage("Query completed", 3000)
        
        self.conversation_id = result.get("conversation_id")
        
        answer = result.get("answer", "No answer generated.")
        verses = result.get("verses", [])
        thoughts = result.get("thoughts", [])

        thought_text = '<div style="margin-bottom:15px;padding:10px;background-color:rgba(43,91,132,0.1);border-radius:5px;">'
        thought_text += '<p style="color:#2b5b84;"><b>Processing Steps:</b></p>'
        for t in thoughts:
            thought_text += f'<p>• {t}</p>'
        thought_text += '</div>'

        qa_text = f'{thought_text}<div style="margin-bottom:10px;"><p style="color:#2b5b84;"><b>Q: {self.input_field.text()}</b></p><p><b>A:</b></p>'
        qa_text += self.qna_display._markdown_to_html(answer)
        qa_text += '</div>'
        
        self.qna_display.append(qa_text)
        self.qna_display.moveCursor(QTextCursor.End)

        self.update_quotes_display(verses)
        self.input_field.clear()

    def update_quotes_display(self, verses):
        self.quotes_display.clear()
        html_chunks = []
        for v in verses:
            chunk = [f'<div style="margin-bottom:20px;padding:15px;border-radius:5px;background-color:rgba(43,91,132,0.02);">'
                      f'<p style="color:#2b5b84;font-size:16px;margin-bottom:10px;"><b>{v["book"]} {v["chapter"]}:{v["verse"]}</b></p>'
                      f'<p style="margin-bottom:10px;">{self.quotes_display._markdown_to_html(v["text"])}</p>']
            if v.get("greek_text"):
                chunk.append(
                    f'<div style="margin:10px 0;padding:10px;background-color:rgba(43,91,132,0.05);'
                    f'border-left:3px solid #2b5b84;border-radius:0 5px 5px 0;">'
                    f'<p style="color:#2b5b84;"><b>Greek Text:</b></p>'
                    f'<p style="font-style:italic;">{self.quotes_display._markdown_to_html(v["greek_text"])}</p>'
                    f'</div>'
                )
            if v.get("complexity_score") is not None:
                cscore = v["complexity_score"]
                chunk.append(
                    f'<div style="margin:15px 0;padding:10px;background-color:rgba(43,91,132,0.05);border-radius:5px;">'
                    f'<p style="color:#2b5b84;margin-bottom:10px;"><b>Complexity Analysis</b></p>'
                    f'<p>Complexity Score: <b>{cscore:.3f}</b></p>'
                )
                book = v["book"]
                chapter = v["chapter"]
                verse = v["verse"]
                chunk.append(
                    f'<p><a href="graph://{book}:{chapter}:{verse}" style="color:#2b5b84;text-decoration:none;border-bottom:1px solid #2b5b84;">View Complexity Graph</a></p>'
                )
                if v.get("similar_by_complexity"):
                    sim = v["similar_by_complexity"]
                    chunk.append('<div style="margin-top:10px;"><p style="color:#2b5b84;"><b>Similar Verses by Complexity:</b></p>')
                    eng_verses = [sv for sv in sim if sv.get("match_type") == "english"]
                    greek_verses = [sv for sv in sim if sv.get("match_type") == "greek"]
                    if eng_verses:
                        chunk.append('<div style="margin:10px 0;"><p style="color:#2b5b84;"><i>Similar in English Structure:</i></p>')
                        for s in eng_verses:
                            chunk.append(
                                f'<div style="margin:5px 0 10px 15px;padding:8px;'
                                f'background-color:rgba(43,91,132,0.03);border-radius:3px;">'
                                f'<p style="color:#2b5b84;"><b>{s["book"]} {s["chapter"]}:{s["verse"]}</b></p>'
                                f'<p>{self.quotes_display._markdown_to_html(s["text"])}</p>'
                                f'</div>'
                            )
                        chunk.append('</div>')
                    if greek_verses:
                        chunk.append('<div style="margin:10px 0;"><p style="color:#2b5b84;"><i>Similar in Greek Structure:</i></p>')
                        for s in greek_verses:
                            chunk.append(
                                f'<div style="margin:5px 0 10px 15px;padding:8px;'
                                f'background-color:rgba(43,91,132,0.03);border-radius:3px;">'
                                f'<p style="color:#2b5b84;"><b>{s["book"]} {s["chapter"]}:{s["verse"]}</b></p>'
                                f'<p>{self.quotes_display._markdown_to_html(s["text"])}</p>'
                                f'</div>'
                            )
                        chunk.append('</div>')
                    chunk.append('</div>')
                chunk.append('</div>')
            chunk.append('</div>')
            html_chunks.append(''.join(chunk))
        
        self.quotes_display.setHtml(''.join(html_chunks))
        self.quotes_display.moveCursor(QTextCursor.Start)

    def on_query_error(self, error):
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.status_bar.showMessage("Error processing query", 3000)
        QMessageBox.critical(self, "Error", str(error))

    @pyqtSlot(QUrl)
    def on_anchor_clicked(self, url: QUrl):
        if url.scheme() == "graph":
            try:
                path = url.toString().replace("graph://", "")
                # Use a regex to ensure we can parse the book/chapter/verse correctly
                match = re.match(r'^(.*?)\:(\d+)\:(\d+)$', path)
                if not match:
                    QMessageBox.warning(self, "Error", "Invalid verse reference format.")
                    return
                book = match.group(1)
                chapter = int(match.group(2))
                verse = int(match.group(3))

                r = requests.get(f"{self.api_url}/complexity_graph", 
                               params={"book": book, "chapter": chapter, "verse": verse})
                if r.status_code == 200:
                    img_data = r.content
                    dlg = ImageDialog(img_data, f"Complexity Graph for {book} {chapter}:{verse}", self)
                    dlg.exec_()
                else:
                    QMessageBox.warning(self, "Error", "Could not retrieve complexity graph.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error processing graph request: {str(e)}")
        else:
            QDesktopServices.openUrl(url)

def run_server():
    try:
        base_port = settings.PORT
        max_retries = 5
        current_port = base_port
        
        while max_retries > 0:
            try:
                logger.info(f"Attempting to start server on port {current_port}...")
                config = uvicorn.Config(
                    app=app,
                    host=settings.HOST,
                    port=current_port,
                    log_level="info",
                    access_log=False
                )
                server = uvicorn.Server(config)
                server.run()
                break
            except OSError as e:
                if e.errno:
                    logger.warning(f"Port {current_port} is in use, trying next port...")
                    current_port += 1
                    max_retries -= 1
                else:
                    logger.error(f"Server error: {str(e)}")
                    raise
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Start the server in a background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Give the server time to start
        time.sleep(3)

        app_qt = QApplication(sys.argv)
        app_qt.setStyle(QStyleFactory.create('Fusion'))
        
        server_ready = False
        base_port = settings.PORT
        max_port = base_port + 5
        
        for port in range(base_port, max_port):
            try:
                response = requests.get(f"http://{settings.HOST}:{port}/progress", timeout=1)
                if response.status_code == 200:
                    settings.PORT = port
                    server_ready = True
                    break
            except:
                continue

        ex = BibleAIGUI()
        if not server_ready:
            QMessageBox.warning(
                ex,
                "Server Warning",
                "Could not connect to the server. Please ensure Ollama is running and try again."
            )
        
        ex.show()
        sys.exit(app_qt.exec_())
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        QMessageBox.critical(None, "Error", f"Failed to start application: {str(e)}")
        sys.exit(1)
