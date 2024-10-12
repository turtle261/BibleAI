# Optimized FOSS Bible AI Program with FastAPI Backend using Ollama Inference Engine

import json
import faiss
import numpy as np
import asyncio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
import uvicorn
import logging
from tqdm import tqdm
import os
from typing import List, Optional, Tuple
from fastapi.middleware.cors import CORSMiddleware
import gc
import subprocess
import requests
import time

# Removed caching imports
# from fastapi_cache import FastAPICache
# from fastapi_cache.backends.inmemory import InMemoryBackend

# Configure logging with reduced verbosity to save memory
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Optimized FOSS Bible AI API",
    version="5.0",
    description="An advanced, memory-optimized AI-powered Bible study and theological research tool using Ollama as the inference engine."
)

# Enable CORS for frontend interaction with restricted origins for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1:8000"],  # Restrict to trusted origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class VerseResult(BaseModel):
    book: str
    chapter: int
    verse: int
    text: str

class ResponseResult(BaseModel):
    answer: str
    verses: List[VerseResult]
    conversation_id: str

class BibleAI:
    def __init__(self, bible_data_path: str, faiss_index_path: str = "data/embeddings/faiss.index", model_name: str = "llama3.2"):
        self.model_name = model_name
        logger.info(f"Initializing BibleAI with Ollama model: {self.model_name}")

        # Ensure Ollama is installed
        if not self._check_ollama_installed():
            logger.critical("Ollama is not installed or not found in PATH. Please install Ollama and try again.")
            raise EnvironmentError("Ollama is not installed or not found in PATH.")

        # Check if the model is downloaded; if not, download it
        if not self._is_model_available():
            logger.info(f"Model '{self.model_name}' not found. Downloading...")
            self._download_model()
            logger.info(f"Model '{self.model_name}' downloaded successfully.")
        else:
            logger.info(f"Model '{self.model_name}' is already available.")

        # Load and prepare Bible data
        logger.info("Loading Bible data...")
        try:
            with open(bible_data_path, 'r', encoding='utf-8-sig') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            logger.critical(f"Bible data file '{bible_data_path}' not found.")
            raise
        except json.JSONDecodeError as e:
            logger.critical(f"Error decoding JSON from '{bible_data_path}': {e}")
            raise

        # Transform the JSON structure into a flat list of verses with required fields
        self.bible_data = []
        for book in raw_data:
            abbrev = book.get('abbrev', '')
            book_name = book.get('name', 'Unknown')  # Changed from 'book' to 'name'
            chapters = book.get('chapters', [])
            for chapter_idx, chapter in enumerate(chapters, start=1):
                for verse_idx, verse_text in enumerate(chapter, start=1):
                    verse_entry = {
                        'book': book_name,  # Ensure 'book' uses the correct key
                        'chapter': chapter_idx,
                        'verse': verse_idx,
                        'text': verse_text
                    }
                    self.bible_data.append(verse_entry)

        logger.info(f"Loaded {len(self.bible_data)} Bible verses.")

        # Ensure the embeddings directory exists
        embeddings_dir = os.path.dirname(faiss_index_path)
        os.makedirs(embeddings_dir, exist_ok=True)

        # Initialize the embedder
        self.embedder = SentenceTransformerWrapper()

        # Build or load FAISS index
        if os.path.exists(faiss_index_path):
            logger.info(f"Loading FAISS index from {faiss_index_path}...")
            self.index = faiss.read_index(faiss_index_path)
        else:
            logger.info("Building FAISS index...")
            self._build_faiss_index(faiss_index_path)

        # Conversation management
        self.conversations = {}  # Stores conversation histories

    def _build_faiss_index(self, faiss_index_path: str):
        embeddings = self.embedder.encode(
            [verse['text'] for verse in self.bible_data],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        embeddings = embeddings.astype('float32')

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, faiss_index_path)
        logger.info(f"FAISS index built and saved to {faiss_index_path}.")

    def generate_response(self, question: str, conversation_id: Optional[str] = None) -> Tuple[str, List[dict], str]:
        if conversation_id is None:
            conversation_id = self._generate_conversation_id()
        conversation = self.conversations.get(conversation_id, [])

        # Append user question to conversation
        conversation.append(f"User: {question}")

        # Step 1: Retrieve relevant verses using semantic search
        relevant_verses = self._retrieve_relevant_verses(question, top_k=10)

        # Step 2: Generate answer using LLM with relevant verses
        answer = self._generate_answer_with_quotes(question, relevant_verses)

        # Append AI answer and relevant verses to conversation
        conversation.append(f"AI: {answer}")
        for verse in relevant_verses:
            conversation.append(f"{verse['book']} {verse['chapter']}:{verse['verse']} - {verse['text']}")

        self.conversations[conversation_id] = conversation

        return answer, self.conversations[conversation_id], conversation_id

    def _retrieve_relevant_verses(self, query: str, top_k: int = 10) -> List[dict]:
        query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        relevant_verses = [self.bible_data[idx] for idx in indices[0] if idx < len(self.bible_data)]
        return relevant_verses

    def _generate_answer_with_quotes(self, question: str, verses: List[dict]) -> str:
        verses_text = "\n".join(
            [f"{verse['book']} {verse['chapter']}:{verse['verse']} - {verse['text']}" for verse in verses]
        )

        prompt = (
            f"You are a Christian AI assistant. Consider the following Bible verses:\n\n"
            f"{verses_text}\n\n"
            f"Using these verses as reference, provide a comprehensive answer to the user's question. "
            f"Include relevant quotes from the verses in your answer, citing the book, chapter, and verse. "
            f"If a verse is particularly relevant, explain its significance in the context of the question.\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        answer = self._query_ollama(prompt)
        return answer

    def _query_ollama(self, prompt: str) -> str:
        """
        Query the Ollama inference engine with the given prompt.
        """
        try:
            env = os.environ.copy()
            # Adjust PATH to include likely Ollama installation directory
            env["PATH"] = "/usr/bin:" + env.get("PATH", "")
            response = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            answer = response.stdout.strip()
            return answer
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama inference error: {e.stderr}")
            raise Exception("Failed to generate response from Ollama.")

    def _generate_conversation_id(self) -> str:
        return f"conv_{len(self.conversations) + 1}"

    def _check_ollama_installed(self) -> bool:
        try:
            subprocess.run(["ollama", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _is_model_available(self) -> bool:
        try:
            env = os.environ.copy()
            # Adjust PATH to include likely Ollama installation directory
            env["PATH"] = "/usr/bin:" + env.get("PATH", "")
            result = subprocess.run(
                ["ollama", "list"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            return self.model_name in result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Error checking Ollama models: {e.stderr}")
            return False

    def _download_model(self):
        try:
            env = os.environ.copy()
            env["PATH"] = "/usr/bin:" + env.get("PATH", "")
            subprocess.run(["ollama", "pull", self.model_name], check=True, env=env)
            # Wait briefly to ensure the model is fully downloaded
            time.sleep(5)
        except subprocess.CalledProcessError as e:
            logger.critical(f"Failed to download model '{self.model_name}': {e.stderr}")
            raise

class SentenceTransformerWrapper:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device='cpu')  # Ensure CPU usage for minimal memory

    def encode(self, sentences: List[str], show_progress_bar: bool = False, convert_to_numpy: bool = False) -> np.ndarray:
        return self.model.encode(
            sentences, 
            show_progress_bar=show_progress_bar, 
            convert_to_numpy=convert_to_numpy
            # Removed 'clean_up_tokenization_spaces' to fix the TypeError
        )

# Initialize BibleAI instance with error handling and minimal memory usage
try:
    ai = BibleAI('data/bible.json')
except Exception as e:
    logger.critical(f"Failed to initialize BibleAI: {e}")
    exit(1)

@app.post("/query", response_model=ResponseResult)
async def query_bible_ai(q: Query):
    logger.debug(f"Received query: '{q.question}' with conversation_id: '{q.conversation_id}'")
    try:
        # Offload the blocking generate_response to a separate thread
        answer, conversation_history, conversation_id = await asyncio.to_thread(
            ai.generate_response, q.question, q.conversation_id
        )

        # Extract the relevant verses
        relevant_quotes = []
        for item in conversation_history:
            if item.startswith("AI:") or item.startswith("User:"):
                continue
            parts = item.split(" - ", 1)
            if len(parts) == 2:
                reference, text = parts
                try:
                    # Split from the right to handle book names with spaces
                    book_and_chapter_verse = reference.rsplit(" ", 1)
                    if len(book_and_chapter_verse) == 2:
                        book = book_and_chapter_verse[0]
                        chapter_verse = book_and_chapter_verse[1]
                        chapter, verse = map(int, chapter_verse.split(":"))
                        relevant_quotes.append({
                            "book": book,
                            "chapter": chapter,
                            "verse": verse,
                            "text": text
                        })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse verse reference '{reference}': {e}")
                    continue

        # Limit to top 5 quotes
        relevant_quotes = relevant_quotes[:5]

        logger.debug(f"Returning response for conversation_id: '{conversation_id}'")
        return ResponseResult(answer=answer, verses=relevant_quotes, conversation_id=conversation_id)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error for request {request.url.path}: {exc}")
    raise HTTPException(status_code=422, detail=exc.errors())

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException for request {request.url.path}: {exc.detail}")
    raise exc

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Optimized FOSS Bible AI API using Ollama as the inference engine. Use the /query endpoint to interact."}

if __name__ == "__main__":
    # Reduce number of workers to limit RAM usage
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
