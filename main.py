# Optimized FOSS Bible AI Program with FastAPI Backend using Ollama Inference Engine

import json
import faiss
import numpy as np
import asyncio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import logging
from tqdm import tqdm
import os
from typing import List, Optional, Tuple, Dict, Any, Set
import requests
import time
from datetime import datetime
import zlib
import torch
from enum import Enum
import re
from fastapi.middleware.cors import CORSMiddleware
import threading
from fastapi.responses import JSONResponse
from pathlib import Path

from config import settings

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"bible_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging with both file and console handlers
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# File handler with detailed formatting
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('''
==============================================
%(asctime)s - %(name)s - %(levelname)s
==============================================
%(message)s
''')
file_handler.setFormatter(file_formatter)

# Console handler with simpler formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Get logger and add handlers
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Set OpenMP environment variable to avoid runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize FAISS for CPU only
try:
    # Ensure we're using CPU version
    logger.info("Initializing CPU-only FAISS")
    if not hasattr(faiss, 'IndexFlatL2'):
        raise ImportError("FAISS CPU version not properly initialized")
except Exception as e:
    logger.error(f"Error initializing FAISS: {e}")
    raise

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

class QueryIntent:
    PATTERNS = {
        ToolType.VERSE_LOOKUP: [
            r'(?i)what does ([\w\s]+) (\d+):(\d+) say',
            r'(?i)show me ([\w\s]+) (\d+):(\d+)',
            r'(?i)([\w\s]+) (\d+):(\d+)',
        ],
        ToolType.GREEK_ANALYSIS: [
            r'(?i)greek',
            r'(?i)original language',
            r'(?i)translation of',
            r'(?i)meaning of.*in greek',
        ],
        ToolType.COMPLEXITY_ANALYSIS: [
            r'(?i)complex(ity)?',
            r'(?i)similar structure',
            r'(?i)similar verses',
            r'(?i)pattern',
            r'(?i)compare.*verse',
        ]
    }
    
    @staticmethod
    def analyze(question: str) -> Set[ToolType]:
        tools = {ToolType.SEMANTIC_SEARCH}  # Always include semantic search
        
        for tool_type, patterns in QueryIntent.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    tools.add(tool_type)
                    break
                    
        return tools

    @staticmethod
    def extract_verse_reference(question: str) -> Optional[Tuple[str, int, int]]:
        for pattern in QueryIntent.PATTERNS[ToolType.VERSE_LOOKUP]:
            match = re.search(pattern, question)
            if match:
                try:
                    book = match.group(1).strip()
                    chapter = int(match.group(2))
                    verse = int(match.group(3))
                    return (book, chapter, verse)
                except (IndexError, ValueError):
                    continue
        return None

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
            
        try:
            logger.info("=== LLM Generation Request ===")
            logger.info(f"Model: {model}")
            logger.info(f"System Prompt: {system}")
            logger.info(f"User Prompt: {prompt}")
            
            response = await asyncio.to_thread(
                self.session.post,
                self.generate_url,
                json=payload
            )
            response.raise_for_status()
            result = response.json()["response"]
            
            logger.info("=== LLM Generation Response ===")
            logger.info(result)
            return result
        except Exception as e:
            logger.error(f"Ollama API error in generate: {e}")
            raise

    async def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        logger.info("=== Embedding Generation Request ===")
        logger.info(f"Number of texts: {len(texts)}")
        logger.info(f"Batch size: {batch_size}")
        
        all_embeddings = []
        retries = 3  # Number of retries for failed requests
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings_batch = []
            
            for text in batch:
                for attempt in range(retries):
                    try:
                        payload = {
                            "model": settings.EMBEDDING_MODEL,
                            "prompt": text
                        }
                        response = await asyncio.to_thread(
                            self.session.post,
                            self.embeddings_url,
                            json=payload
                        )
                        response.raise_for_status()
                        embedding = response.json().get("embedding")
                        
                        if not embedding:
                            raise ValueError("No embedding returned from Ollama API")
                            
                        if len(embedding) != settings.EMBEDDING_DIMENSION:
                            raise ValueError(
                                f"Embedding dimension mismatch. Expected {settings.EMBEDDING_DIMENSION}, "
                                f"got {len(embedding)}"
                            )
                            
                        embeddings_batch.append(embedding)
                        break  # Success, break retry loop
                        
                    except Exception as e:
                        if attempt == retries - 1:  # Last attempt
                            logger.error(f"Failed to get embedding after {retries} attempts: {e}")
                            raise
                        else:
                            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying...")
                            await asyncio.sleep(1)  # Wait before retry
                            
            all_embeddings.extend(embeddings_batch)
            logger.info(f"Processed embeddings batch {i // batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
        if not all_embeddings:
            raise RuntimeError("Failed to generate any embeddings")
            
        # Verify all embeddings have the same dimension
        if not all(len(emb) == settings.EMBEDDING_DIMENSION for emb in all_embeddings):
            raise ValueError("Inconsistent embedding dimensions")
            
        return all_embeddings

class ExecutionPlan:
    def __init__(self, steps: List[Dict[str, Any]], total_weight: int = 100):
        self.steps = steps
        self.current_step = 0
        self.total_weight = total_weight
        self.step_weights = self._calculate_step_weights()
        self.progress = 0
        self.thoughts = []
        self.completed_steps = set()

    def _calculate_step_weights(self) -> List[int]:
        weights = []
        remaining_weight = self.total_weight
        
        for step in self.steps:
            weight = step.get('weight', 1)
            total_steps = len(self.steps)
            step_weight = round((weight / sum(s.get('weight', 1) for s in self.steps)) * self.total_weight)
            weights.append(step_weight)
            remaining_weight -= step_weight
            
        # Distribute any remaining weight
        if remaining_weight > 0:
            weights[-1] += remaining_weight
            
        return weights

    def add_thought(self, thought: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_thought = f"[{timestamp}] {thought}"
        self.thoughts.append(formatted_thought)
        logger.info(f"Thought: {formatted_thought}")

    def update_progress(self, step_progress: float):
        if self.current_step < len(self.steps):
            completed_weight = sum(self.step_weights[:self.current_step])
            current_step_contribution = self.step_weights[self.current_step] * step_progress
            self.progress = (completed_weight + current_step_contribution) / self.total_weight * 100

    def next_step(self) -> Optional[Dict[str, Any]]:
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.current_step += 1
            return step
        return None

    def mark_step_complete(self, step_type: str):
        self.completed_steps.add(step_type)
        self.add_thought(f"Completed step: {step_type}")

    def get_completion_status(self) -> Dict[str, bool]:
        return {step['type']: step['type'] in self.completed_steps for step in self.steps}

class KolmogorovAnalyzer:
    def __init__(self):
        self.complexity_cache = {}
        
    def calculate_complexity(self, text: str) -> float:
        """Calculate normalized Kolmogorov complexity using multiple compression algorithms."""
        if text in self.complexity_cache:
            return self.complexity_cache[text]

        # Normalize text
        normalized_text = self._normalize_text(text)
        
        # Calculate complexities using different methods
        complexities = []
        
        # zlib compression
        zlib_compressed = zlib.compress(normalized_text.encode())
        complexities.append(len(zlib_compressed) / len(normalized_text))
        
        # Character frequency analysis
        char_entropy = self._calculate_entropy(normalized_text)
        complexities.append(char_entropy / 8.0)  # Normalize by max entropy per byte
        
        # Word-level complexity
        word_complexity = self._calculate_word_complexity(normalized_text)
        complexities.append(word_complexity)
        
        # Combine metrics (weighted average)
        weights = [0.5, 0.25, 0.25]  # Weights for different complexity measures
        complexity = sum(c * w for c, w in zip(complexities, weights))
        
        # Normalize to 0-1 range
        complexity = max(0.0, min(1.0, complexity))
        
        self.complexity_cache[text] = complexity
        return complexity

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent complexity analysis."""
        # Remove extra whitespace and convert to lowercase
        text = ' '.join(text.lower().split())
        # Remove punctuation except periods and spaces
        text = re.sub(r'[^\w\s.]', '', text)
        return text

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the text."""
        if not text:
            return 0.0
        
        # Count character frequencies
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
            
        # Calculate entropy
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            probability = count / length
            entropy -= probability * np.log2(probability)
            
        return entropy

    def _calculate_word_complexity(self, text: str) -> float:
        """Calculate complexity based on word patterns."""
        words = text.split()
        if not words:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Unique words ratio
        unique_ratio = len(set(words)) / len(words)
        
        # Combine metrics
        return (avg_word_length / 15 + unique_ratio) / 2  # Normalize word length by expected max

    def find_most_similar_verse(self, verse: dict, all_verses: List[dict], greek_data: dict) -> Tuple[List[dict], List[dict]]:
        """Find the two most similar verses by complexity in both English and Greek."""
        target_eng_complexity = self.calculate_complexity(verse['text'])
        
        # Get Greek text for the target verse if available
        target_greek = greek_data.get(verse['book'], {}).get(str(verse['chapter']), {}).get(str(verse['verse']))
        target_greek_complexity = self.calculate_complexity(target_greek) if target_greek else None
        
        # Find most similar verses by English complexity
        eng_similar = []
        eng_differences = []
        
        # Find most similar verses by Greek complexity
        greek_similar = []
        greek_differences = []
        
        for v in all_verses:
            # Skip the target verse itself
            if (v['book'] == verse['book'] and 
                v['chapter'] == verse['chapter'] and 
                v['verse'] == verse['verse']):
                continue
            
            # Calculate English complexity difference
            eng_complexity = self.calculate_complexity(v['text'])
            eng_diff = abs(eng_complexity - target_eng_complexity)
            
            # Store English verse and difference
            eng_differences.append((eng_diff, v))
            
            # If we have Greek text for the target verse, check Greek similarity
            if target_greek_complexity is not None:
                v_greek = greek_data.get(v['book'], {}).get(str(v['chapter']), {}).get(str(v['verse']))
                if v_greek:
                    greek_complexity = self.calculate_complexity(v_greek)
                    greek_diff = abs(greek_complexity - target_greek_complexity)
                    greek_differences.append((greek_diff, v))
        
        # Sort by difference and get top 2 for each language
        eng_differences.sort(key=lambda x: x[0])
        eng_similar = [v for _, v in eng_differences[:2]]
        
        if greek_differences:
            greek_differences.sort(key=lambda x: x[0])
            greek_similar = [v for _, v in greek_differences[:2]]
        
        return eng_similar, greek_similar

class VerseScoringSystem:
    """A general scoring system for ranking Bible verses based on relevance and context."""
    
    def calculate_verse_score(self, verse: dict, semantic_score: float) -> float:
        """Calculate a score for a verse based on semantic relevance and context."""
        # Base score from semantic search
        final_score = semantic_score
        
        # Slight preference for verses that appear earlier in chapters
        # This helps maintain natural reading order when multiple verses are relevant
        verse_position_weight = 1.0 / (1.0 + 0.01 * verse['verse'])
        final_score *= verse_position_weight
        
        return final_score
        
    def get_context_verses(self, verse: dict, all_verses: List[dict], window: int = 2) -> List[dict]:
        """Get surrounding verses for context."""
        context_verses = []
        book, chapter, verse_num = verse['book'], verse['chapter'], verse['verse']
        
        for v in all_verses:
            if (v['book'] == book and v['chapter'] == chapter and
                abs(v['verse'] - verse_num) <= window and
                v['verse'] != verse_num):
                context_verses.append(v)
                
        return context_verses

class BibleAgent:
    def __init__(self):
        self.model_name = settings.MODEL_NAME
        self.ollama = OllamaAPI()
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.complexity_analyzer = KolmogorovAnalyzer()
        self.verse_scorer = VerseScoringSystem()
        
        # Book abbreviation to full name mapping
        self.book_names = {
            'gn': 'Genesis', 'ex': 'Exodus', 'lv': 'Leviticus', 'nm': 'Numbers',
            'dt': 'Deuteronomy', 'js': 'Joshua', 'jud': 'Judges', 'rt': 'Ruth',
            '1sm': '1 Samuel', '2sm': '2 Samuel', '1kgs': '1 Kings', '2kgs': '2 Kings',
            '1ch': '1 Chronicles', '2ch': '2 Chronicles', 'ezr': 'Ezra', 'ne': 'Nehemiah',
            'et': 'Esther', 'job': 'Job', 'ps': 'Psalms', 'prv': 'Proverbs',
            'ec': 'Ecclesiastes', 'so': 'Song of Solomon', 'is': 'Isaiah',
            'jr': 'Jeremiah', 'lm': 'Lamentations', 'ez': 'Ezekiel', 'dn': 'Daniel',
            'ho': 'Hosea', 'jl': 'Joel', 'am': 'Amos', 'ob': 'Obadiah', 'jn': 'Jonah',
            'mi': 'Micah', 'na': 'Nahum', 'hk': 'Habakkuk', 'zp': 'Zephaniah',
            'hg': 'Haggai', 'zc': 'Zechariah', 'ml': 'Malachi', 'mt': 'Matthew',
            'mk': 'Mark', 'lk': 'Luke', 'jo': 'John', 'act': 'Acts', 'rm': 'Romans',
            '1co': '1 Corinthians', '2co': '2 Corinthians', 'gl': 'Galatians',
            'eph': 'Ephesians', 'ph': 'Philippians', 'cl': 'Colossians',
            '1ts': '1 Thessalonians', '2ts': '2 Thessalonians', '1tm': '1 Timothy',
            '2tm': '2 Timothy', 'tt': 'Titus', 'phm': 'Philemon', 'hb': 'Hebrews',
            'jm': 'James', '1pe': '1 Peter', '2pe': '2 Peter', '1jo': '1 John',
            '2jo': '2 John', '3jo': '3 John', 'jd': 'Jude', 're': 'Revelation'
        }
        self.index = None
        self.bible_data = []
        self.greek_data = {}
        self._initialized = False
        self._initializing = False

    async def initialize(self):
        """Initialize the agent asynchronously."""
        if self._initialized or self._initializing:
            return
        
        self._initializing = True
        try:
            await self._initialize_data()
            self._initialized = True
        finally:
            self._initializing = False

    async def ensure_initialized(self):
        """Ensure the agent is initialized before processing queries."""
        if not self._initialized:
            await self.initialize()

    async def _initialize_data(self):
        """Initialize data asynchronously to improve startup time."""
        try:
            await asyncio.gather(
                self._load_bible_data(),
                self._load_greek_data(),
                self._setup_semantic_search()
            )
            logger.info("Bible Agent initialization complete")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    async def _load_bible_data(self):
        """Load Bible data asynchronously."""
        try:
            with open(settings.BIBLE_DATA_PATH, 'r', encoding='utf-8-sig') as f:
                raw_data = json.load(f)
            
            for book in raw_data:
                abbrev = book.get('abbrev', 'unknown')
                book_name = self.book_names.get(abbrev.lower(), abbrev)
                
                chapters = book.get('chapters', [])
                for chapter_idx, chapter in enumerate(chapters, start=1):
                    for verse_idx, verse_text in enumerate(chapter, start=1):
                        clean_text = re.sub(r'\{.*?\}', '', verse_text)
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                        
                        verse_entry = {
                            'book': book_name,
                            'chapter': chapter_idx,
                            'verse': verse_idx,
                            'text': clean_text
                        }
                        self.bible_data.append(verse_entry)
            
            logger.info(f"Loaded {len(self.bible_data)} Bible verses")
        except Exception as e:
            logger.error(f"Failed to load Bible data: {e}")
            raise

    async def _load_greek_data(self):
        """Load Greek Bible data asynchronously."""
        try:
            with open(settings.GREEK_DATA_PATH, 'r', encoding='utf-8-sig') as f:
                raw_data = json.load(f)
            
            for book in raw_data:
                abbrev = book.get('abbrev', 'unknown')
                book_name = self.book_names.get(abbrev.lower(), abbrev)
                
                if book_name not in self.greek_data:
                    self.greek_data[book_name] = {}
                    
                chapters = book.get('chapters', [])
                for chapter_idx, chapter in enumerate(chapters, start=1):
                    chapter_str = str(chapter_idx)
                    if chapter_str not in self.greek_data[book_name]:
                        self.greek_data[book_name][chapter_str] = {}
                        
                    for verse_idx, verse_text in enumerate(chapter, start=1):
                        clean_text = re.sub(r'\s+', ' ', verse_text).strip()
                        self.greek_data[book_name][chapter_str][str(verse_idx)] = clean_text
            
            logger.info("Greek Bible data loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Greek Bible data: {e}")
            self.greek_data = {}

    async def _setup_semantic_search(self):
        """Setup semantic search asynchronously."""
        try:
            os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
            
            if os.path.exists(settings.FAISS_INDEX_PATH):
                self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
                logger.info("Loaded existing FAISS index")
            else:
                await self._build_faiss_index()
                
        except Exception as e:
            logger.error(f"Error setting up semantic search: {e}")
            raise

    async def _build_faiss_index(self):
        """Build FAISS index asynchronously."""
        try:
            verses = [verse['text'] for verse in self.bible_data]
            embeddings_list = []
            batch_size = settings.EMBEDDING_BATCH_SIZE
            
            for i in range(0, len(verses), batch_size):
                batch = verses[i:i + batch_size]
                embeddings_batch = await self.ollama.get_embeddings(batch, batch_size=batch_size)
                embeddings_list.extend(embeddings_batch)
            
            if not embeddings_list:
                raise RuntimeError("No embeddings were generated")
            
            embeddings = np.array(embeddings_list, dtype=np.float32)
            if embeddings.shape[1] != settings.EMBEDDING_DIMENSION:
                raise ValueError(f"Embedding dimension mismatch")
            
            self.index = faiss.IndexFlatL2(settings.EMBEDDING_DIMENSION)
            self.index.add(embeddings)
            
            faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
            logger.info("FAISS index built and saved successfully")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            if os.path.exists(settings.FAISS_INDEX_PATH):
                try:
                    os.remove(settings.FAISS_INDEX_PATH)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up partial index file: {cleanup_error}")
            raise

    async def process_query(
        self, 
        question: str, 
        conversation_id: Optional[str] = None,
        include_greek: Optional[bool] = None,
        analyze_complexity: Optional[bool] = None
    ) -> Tuple[str, List[dict], str, List[str]]:
        if not conversation_id:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        logger.info(f"=== Processing Query ===")
        logger.info(f"Conversation ID: {conversation_id}")
        logger.info(f"Question: {question}")
        
        # Create execution plan based on query analysis
        plan = self._create_execution_plan(question, include_greek, analyze_complexity)
        
        verses = []
        enhanced_verses = []
        
        while (step := plan.next_step()) is not None:
            step_type = step['type']
            
            if step_type == ToolType.SEMANTIC_SEARCH.value:
                plan.add_thought("Searching for relevant verses using semantic search...")
                semantic_verses = await self._semantic_search(question)
                verses.extend(semantic_verses)
                plan.update_progress(1.0)
                plan.mark_step_complete(step_type)
                
            elif step_type == ToolType.VERSE_LOOKUP.value:
                plan.add_thought("Looking up specific verse reference...")
                verse_ref = QueryIntent.extract_verse_reference(question)
                if verse_ref:
                    book, chapter, verse = verse_ref
                    direct_verse = self._lookup_verse(book, chapter, verse)
                    if direct_verse:
                        verses.append(direct_verse)
                plan.update_progress(1.0)
                plan.mark_step_complete(step_type)
                
            elif step_type == "enhance_verses":
                plan.add_thought("Enhancing verses with additional analysis...")
                for i, verse in enumerate(verses):
                    enhanced_verse = verse.copy()
                    if include_greek:
                        plan.add_thought(f"Analyzing Greek text for verse {verse['book']} {verse['chapter']}:{verse['verse']}...")
                        greek_text = self._get_greek_text(verse)
                        if greek_text:
                            enhanced_verse['greek_text'] = greek_text
                            
                    if analyze_complexity:
                        plan.add_thought(f"Analyzing complexity patterns for verse {verse['book']} {verse['chapter']}:{verse['verse']}...")
                        complexity = self.complexity_analyzer.calculate_complexity(verse['text'])
                        enhanced_verse['complexity_score'] = complexity
                        
                        # Find similar verses by both English and Greek complexity
                        eng_similar, greek_similar = self.complexity_analyzer.find_most_similar_verse(
                            verse,
                            self.bible_data,
                            self.greek_data
                        )
                        
                        similar_verses = []
                        # Add English similar verses
                        for eng_verse in eng_similar:
                            similar_verses.append({
                                'book': eng_verse['book'],
                                'chapter': eng_verse['chapter'],
                                'verse': eng_verse['verse'],
                                'text': eng_verse['text'],
                                'match_type': 'english'
                            })
                        
                        # Add Greek similar verses
                        for greek_verse in greek_similar:
                            # Only add if not already included in English matches
                            if not any(
                                v['book'] == greek_verse['book'] and 
                                v['chapter'] == greek_verse['chapter'] and 
                                v['verse'] == greek_verse['verse'] 
                                for v in similar_verses
                            ):
                                similar_verses.append({
                                    'book': greek_verse['book'],
                                    'chapter': greek_verse['chapter'],
                                    'verse': greek_verse['verse'],
                                    'text': greek_verse['text'],
                                    'match_type': 'greek'
                                })
                        
                        enhanced_verse['similar_by_complexity'] = similar_verses
                        
                    enhanced_verses.append(enhanced_verse)
                    plan.update_progress((i + 1) / len(verses))
                plan.mark_step_complete(step_type)
                    
            elif step_type == ToolType.RESPONSE_GENERATION.value:
                plan.add_thought("Generating final response based on analysis...")
                answer = await self._generate_response(question, enhanced_verses, plan)
                plan.update_progress(1.0)
                plan.mark_step_complete(step_type)
        
        self._update_conversation(conversation_id, question, answer, enhanced_verses, plan)
        logger.info("=== Query Processing Complete ===")
        
        return answer, enhanced_verses, conversation_id, plan.thoughts

    def _create_execution_plan(
        self, 
        question: str, 
        include_greek: bool, 
        analyze_complexity: bool
    ) -> ExecutionPlan:
        """Create a general execution plan based on query analysis."""
        steps = []
        tools = QueryIntent.analyze(question)
        
        # Analyze query complexity
        query_complexity = self.complexity_analyzer.calculate_complexity(question)
        
        # Add verse lookup for specific references first
        if ToolType.VERSE_LOOKUP in tools:
            steps.append({
                'type': ToolType.VERSE_LOOKUP.value,
                'weight': 2,
                'description': 'Looking up specific verse reference'
            })
        
        # Add semantic search with standard weight
        steps.append({
            'type': ToolType.SEMANTIC_SEARCH.value,
            'weight': 3,
            'description': 'Performing semantic search for relevant verses'
        })
        
        # Add verse enhancement step if needed
        if include_greek or analyze_complexity:
            steps.append({
                'type': 'enhance_verses',
                'weight': 3,
                'description': 'Enhancing verses with analysis'
            })
        
        # Add response generation
        steps.append({
            'type': ToolType.RESPONSE_GENERATION.value,
            'weight': 2,
            'description': 'Generating response'
        })
        
        return ExecutionPlan(steps)

    async def _generate_response(self, question: str, verses: List[dict], plan: ExecutionPlan) -> str:
        system_prompt = (
            "You are a knowledgeable Bible study assistant focused on providing accurate, "
            "contextual answers. Consider all provided verses equally, analyze their "
            "relationships and context, and provide a complete understanding while "
            "maintaining biblical accuracy."
        )
        
        # Sort verses by semantic relevance
        verses_by_score = sorted(verses, key=lambda v: self.verse_scorer.calculate_verse_score(v, 1.0), reverse=True)
        
        verses_text = "\n".join([
            f"{v['book']} {v['chapter']}:{v['verse']} - {v['text']}" +
            (f"\nGreek: {v.get('greek_text', '')}" if v.get('greek_text') else "") +
            (f"\nComplexity: {v.get('complexity_score', '')}" if v.get('complexity_score') is not None else "")
            for v in verses_by_score
        ])
        
        prompt = (
            f"Question: {question}\n\n"
            f"Relevant verses:\n{verses_text}\n\n"
            "Analyze these verses in their context and provide a clear, comprehensive answer "
            "that explains their meaning and relationships. Consider all verses equally and "
            "ensure the response is accurate and complete."
        )
        
        plan.add_thought("Generating final response using LLM...")
        return await self.ollama.generate(
            model=self.model_name,
            prompt=prompt,
            system=system_prompt
        )

    def _lookup_verse(self, book: str, chapter: int, verse: int) -> Optional[dict]:
        try:
            book_lower = book.lower()
            for v in self.bible_data:
                if (v['book'].lower() == book_lower and 
                    v['chapter'] == chapter and 
                    v['verse'] == verse):
                    return v
            return None
        except Exception as e:
            logger.error(f"Error looking up verse: {e}")
            return None

    def _update_conversation(self, conv_id: str, question: str, answer: str, verses: List[dict], plan: ExecutionPlan):
        """Update conversation history with the latest interaction."""
        if conv_id not in self.conversations:
            self.conversations[conv_id] = []
            
        # Get completed tools from plan
        completed_tools = list(plan.completed_steps)
        
        # Create conversation entry
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "verses": verses,
            "tools_used": completed_tools,
            "thoughts": plan.thoughts
        }
        
        self.conversations[conv_id].append(conversation_entry)
        logger.info(f"Updated conversation {conv_id} with latest interaction")

    async def _semantic_search(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
            
        try:
            # Get embeddings and initial matches
            query_embedding = await self.ollama.get_embeddings([query])
            if not query_embedding or len(query_embedding[0]) != settings.EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Invalid query embedding dimension. Expected {settings.EMBEDDING_DIMENSION}, "
                    f"got {len(query_embedding[0]) if query_embedding else 0}"
                )
                
            query_vector = np.array(query_embedding, dtype=np.float32)
            distances, indices = self.index.search(query_vector, top_k * 2)  # Get more initial matches
            
            # Score verses based on semantic relevance only
            scored_verses = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.bible_data):
                    verse = self.bible_data[idx].copy()
                    semantic_score = 1.0 / (1.0 + distance)  # Convert distance to similarity score
                    verse['score'] = semantic_score
                    scored_verses.append(verse)
            
            # Sort by semantic score and get top_k
            scored_verses.sort(key=lambda x: x['score'], reverse=True)
            top_verses = scored_verses[:top_k]
            
            # Add context verses
            final_verses = []
            seen_verses = set()
            
            for verse in top_verses:
                if (verse['book'], verse['chapter'], verse['verse']) not in seen_verses:
                    final_verses.append(verse)
                    seen_verses.add((verse['book'], verse['chapter'], verse['verse']))
                    
                    # Add context verses if they're not already included
                    context_verses = self.verse_scorer.get_context_verses(verse, self.bible_data)
                    for ctx_verse in context_verses:
                        ctx_key = (ctx_verse['book'], ctx_verse['chapter'], ctx_verse['verse'])
                        if ctx_key not in seen_verses:
                            final_verses.append(ctx_verse)
                            seen_verses.add(ctx_key)
            
            return final_verses
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise RuntimeError(f"Semantic search failed: {str(e)}")
        
    def _get_greek_text(self, verse: dict) -> Optional[str]:
        try:
            return self.greek_data.get(verse['book'], {}) \
                                .get(str(verse['chapter']), {}) \
                                .get(str(verse['verse']))
        except Exception:
            return None

agent = BibleAgent()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await agent.initialize()

@app.post("/query", response_model=ResponseResult)
async def query_bible_ai(q: Query):
    try:
        await agent.ensure_initialized()
        answer, verses, conv_id, thoughts = await agent.process_query(
            q.question,
            q.conversation_id,
            q.include_greek,
            q.analyze_complexity
        )
        
        verse_results = []
        for verse in verses:
            verse_dict = {
                "book": verse["book"],
                "chapter": verse["chapter"],
                "verse": verse["verse"],
                "text": verse["text"],
            }
            if "greek_text" in verse:
                verse_dict["greek_text"] = verse["greek_text"]
            if "complexity_score" in verse:
                verse_dict["complexity_score"] = verse["complexity_score"]
                verse_dict["similar_by_complexity"] = verse.get("similar_by_complexity", [])
            verse_results.append(VerseResult(**verse_dict))
            
        return ResponseResult(
            answer=answer,
            verses=verse_results,
            conversation_id=conv_id,
            thoughts=thoughts
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress")
async def get_progress():
    with agent.progress_lock:
        current_progress = agent.progress
    return {"progress": current_progress}

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Initialize agent
    loop.run_until_complete(agent.initialize())
    
    # Run the server
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, workers=settings.WORKERS)
