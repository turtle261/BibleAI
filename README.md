# Bible AI - Intelligent Bible Study Assistant

## Overview
Bible AI is a modern, intelligent Bible study assistant that combines semantic search capabilities with advanced language models. It features both a FastAPI backend and a user-friendly PyQt5 GUI interface, offering:

- Semantic verse search using FAISS indexing
- Greek text analysis and translation insights
- Verse complexity analysis with pattern matching
- Modern, responsive GUI with dark/light theme support
- Conversation memory and context awareness

## Requirements
- Python 3.8+
- Ollama (for LLM inference)
- 4GB+ RAM recommended
- CUDA-capable GPU (optional, for improved performance)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/turtle261/BibleAI
cd BibleAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Ollama:
```bash
# Install Ollama from https://ollama.com
ollama serve
```

## Usage
1. Start the FastAPI backend:
```bash
python main.py
```

2. Launch the GUI:
```bash
python gui.py
```

The application will be available at `http://localhost:8000` for API access, or use the GUI for a better experience.

## License
GNU General Public License v3.0
