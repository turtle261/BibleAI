# Optimized FOSS Bible AI

## Overview
The Optimized FOSS Bible AI is an advanced, memory-optimized AI-powered tool for Bible study. It uses a FastAPI backend with the Ollama inference engine to provide relevant Bible verses and facilitate conversations with a language model. It uses semantic search to retrieve relevant verses, and a language model to generate answers, using the context of the conversation and the retrieved verses. 

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/turtle261/BibleAI
   cd BibleAI
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama is installed and run `ollama serve`

## Usage
1. Start the FastAPI server:
   ```bash
    python main.py
   ```

2. Access the API at `http://localhost:8000`. `python gui.py`

## License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.
