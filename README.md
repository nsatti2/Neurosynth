# NeuroSynth

A research intelligence tool for neuroscience literature. Built out of frustration with how long literature reviews take — upload your PDFs, ask questions in plain language, and get cited answers grounded in your specific corpus.

## Features (V1)
- Upload and index neuroscience PDFs
- Ask questions → get answers with inline citations
- Conversation memory across follow-up questions
- Auto-extract methodology (study design, sample size, brain regions, stats)
- Filter questions to specific papers

<img width="2994" height="1851" alt="image" src="https://github.com/user-attachments/assets/6c4b6de9-c56b-48b1-a751-bb419151a16e" />



## Tech stack
- **LLM**: Llama 3.3 70B via Groq
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector store**: ChromaDB
- **PDF parsing**: PyMuPDF
- **Frontend**: Streamlit

## Setup

```bash
# 1. Clone and enter the project
git clone <your-repo>
cd neurosynth

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 5. Run the app
streamlit run app/main.py
```

## Usage

1. Add your Groq API key to a `.env` file in the project root: `GROQ_API_KEY=your_key_here` (get one free at console.groq.com)
2. Upload PDFs using the sidebar uploader
3. Click "Index" for each paper — this parses and embeds the text
4. Ask questions in the "Ask papers" tab
5. Use "Extract methods" to pull structured methodology from any paper

> **Note:** Indexed papers persist across sessions via ChromaDB — you only need to index a paper once. If you change chunking settings in `pdf_parser.py`, delete `data/chroma_db/` and re-index.


## Roadmap

- **V2**: Contradiction detector, research gap finder, cross-paper comparison table
- **V3**: Literature review writer, citation graph, paper recommendations

