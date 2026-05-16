# NeuroSynth 🧠

A research intelligence tool for neuroscience papers. Upload PDFs, ask questions, and get cited answers grounded in your specific corpus.

## Features (V1)
- Upload and index neuroscience PDFs
- Ask questions → get answers with inline citations
- Auto-extract methodology (study design, sample size, brain regions, stats)
- Filter questions to specific papers

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

# 4. Run the app
streamlit run app/main.py
```

## Usage

1. Enter your OpenAI API key in the sidebar (get one at platform.openai.com)
2. Upload PDFs using the sidebar uploader
3. Click "Index" for each paper — this parses and embeds the text
4. Ask questions in the "Ask papers" tab
5. Use "Extract methods" to pull structured methodology from any paper

## Project structure

```
neurosynth/
├── app/
│   └── main.py          # Streamlit frontend
├── src/
│   ├── pdf_parser.py    # PDF text extraction + chunking
│   ├── vector_store.py  # ChromaDB embeddings + retrieval
│   └── rag_pipeline.py  # LLM answer generation + methodology extraction
├── data/
│   ├── uploads/         # Temporary PDF storage
│   └── chroma_db/       # Persisted vector database
├── notebooks/           # Experiments and analysis
├── requirements.txt
└── README.md
```

## Roadmap

- **V2**: Contradiction detector, research gap finder, cross-paper comparison table
- **V3**: Literature review writer, citation graph, paper recommendations
