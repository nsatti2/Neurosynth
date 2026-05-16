import fitz  # PyMuPDF
import re
from pathlib import Path


SECTION_HEADERS = [
    "abstract", "introduction", "background", "methods", "methodology",
    "materials and methods", "results", "discussion", "conclusion",
    "references", "limitations", "acknowledgements"
]


def extract_metadata(pdf_path: str) -> dict:
    """
    Extract title, authors, and affiliations from the first page.
    Uses font size to identify the title (largest text) and
    captures the header block before the abstract.
    """
    doc = fitz.open(pdf_path)
    first_page = doc[0]

    blocks = first_page.get_text("dict")["blocks"]
    lines_by_size = []

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                size = span["size"]
                if text:
                    lines_by_size.append((size, text))

    if not lines_by_size:
        doc.close()
        return {"title": "", "header": ""}

    # Title = largest font on the first page
    max_font_size = max(s for s, _ in lines_by_size)
    title_parts = [t for s, t in lines_by_size if s >= max_font_size * 0.95]
    title = " ".join(title_parts)

    first_page_text = first_page.get_text()
    doc.close()

    # Isolate header block (everything before "Abstract")
    abstract_match = re.search(r'\babstract\b', first_page_text, re.IGNORECASE)
    header_text = (
        first_page_text[:abstract_match.start()]
        if abstract_match
        else first_page_text[:1500]
    )

    return {
        "title": title,
        "header": header_text.strip(),
    }


def extract_text_by_section(pdf_path: str) -> dict[str, str]:
    """
    Extract text from a PDF organized by section.
    Injects a 'metadata' section with title + authors from page 1.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    sections = _split_into_sections(full_text)

    meta = extract_metadata(pdf_path)
    if meta["header"]:
        sections["metadata"] = (
            f"Title: {meta['title']}\n\n"
            f"Authors and affiliations:\n{meta['header']}"
        )

    return sections


def _split_into_sections(text: str) -> dict[str, str]:
    """Split raw text into sections based on common header patterns."""
    pattern = r'\n(' + '|'.join(SECTION_HEADERS) + r')[^\n]*\n'
    splits = re.split(pattern, text, flags=re.IGNORECASE)

    sections = {"full_text": text}
    current_section = "preamble"
    buffer = ""

    for part in splits:
        if part.strip().lower() in SECTION_HEADERS:
            if buffer.strip():
                sections[current_section] = buffer.strip()
            current_section = part.strip().lower()
            buffer = ""
        else:
            buffer += part

    if buffer.strip():
        sections[current_section] = buffer.strip()

    return sections


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def parse_paper(pdf_path: str, paper_id: str, original_name: str = None) -> list[dict]:
    """
    Full pipeline: parse PDF -> split sections -> chunk each section.
    Returns a list of chunk dicts ready for embedding.
    """
    sections = extract_text_by_section(pdf_path)
    chunks = []

    for section_name, section_text in sections.items():
        if section_name == "full_text":
            continue
        for i, chunk in enumerate(chunk_text(section_text)):
            chunks.append({
                "paper_id": paper_id,
                "section": section_name,
                "chunk_index": i,
                "text": chunk,
                "source": original_name or Path(pdf_path).name,
            })

    return chunks


if __name__ == "__main__":
    import json
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    try:
        meta = extract_metadata(test_path)
        print("=== METADATA ===")
        print(f"Title: {meta['title']}")
        print(f"Header:\n{meta['header'][:500]}")
        print("\n=== CHUNKS ===")
        result = parse_paper(test_path, paper_id="test_paper")
        print(f"Extracted {len(result)} chunks")
        print(json.dumps(result[0], indent=2))
    except Exception as e:
        print(f"Error: {e}")
