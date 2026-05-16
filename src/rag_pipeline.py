import os
from groq import Groq
from src.vector_store import query as vector_query


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No Groq API key found. Check your .env file.")
    return Groq(api_key=api_key)


SYSTEM_PROMPT = """You are NeuroSynth, a research assistant specialized in neuroscience literature.

You answer questions strictly based on the provided paper excerpts.

RULES:
1. Only use information from the provided excerpts — never add outside knowledge
2. After every claim, cite the source using [Author Year · section] format
3. If the excerpts don't contain enough information, say so clearly
4. Be precise about brain regions, statistical methods, and study designs
5. When papers conflict, note the contradiction explicitly

Your tone is academic but clear. Avoid jargon unless it appears in the source material."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        source_label = f"{chunk['paper_id']} ({chunk['section']})"
        context_parts.append(
            f"[Excerpt {i+1} — {source_label}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def answer_question(
    question: str,
    history: list[dict] | None = None,  
    paper_ids: list[str] | None = None,
    n_chunks: int = 5,
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from vector store
    2. Format as context
    3. Generate cited answer via Groq LLM
    """
    chunks = vector_query(question, n_results=n_chunks, paper_ids=paper_ids)

    if not chunks:
        return {
            "answer": "No relevant content found in the uploaded papers for this question.",
            "sources": [],
        }

    context = format_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Add conversation history
    if history:
        messages.extend(history)

# Add current question with context
    messages.append({
    "role": "user",
    "content": f"PAPER EXCERPTS:\n\n{context}\n\nQUESTION: {question}",
    })

    response = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
        "tokens_used": response.usage.total_tokens,
    }


def extract_methodology(paper_id: str, model: str = "llama-3.3-70b-versatile") -> dict:
    """
    Extract structured methodology from a specific paper.
    """
    chunks = vector_query(
        "study design methods participants sample size brain regions statistical analysis",
        n_results=6,
        paper_ids=[paper_id],
    )

    if not chunks:
        return {"error": f"No content found for paper {paper_id}"}

    context = format_context(chunks)

    extract_prompt = """Extract the methodology from these paper excerpts. Return ONLY a JSON object with these keys:
{
  "study_design": "e.g. fMRI, lesion study, computational model, EEG",
  "n_participants": "sample size as number or null",
  "brain_regions": ["list", "of", "regions"],
  "statistical_methods": ["list", "of", "methods"],
  "key_measures": ["dependent variables or outcomes measured"],
  "species": "human / rat / mouse / etc"
}

If a field is not mentioned, use null. Return only valid JSON, no other text."""

    messages = [
        {"role": "system", "content": extract_prompt},
        {"role": "user", "content": f"PAPER EXCERPTS:\n\n{context}"},
    ]

    response = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    import json
    raw = response.choices[0].message.content
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"error": "Could not parse methodology JSON", "raw": raw}
