def fixed_chunks(pages: list[dict], size: int = 600, overlap: int = 100) -> list[dict]:
    """Slide a fixed-size character window over each document, ignoring structure."""
    by_doc: dict[tuple, list[dict]] = {}
    for p in pages:
        by_doc.setdefault((p["course"], p["source"]), []).append(p)

    chunks = []
    for doc_pages in by_doc.values():
        # join the pages, remembering where each page starts so chunks can cite a page
        text, starts = "", []
        for p in doc_pages:
            starts.append((len(text), p["page"]))
            text += p["text"] + "\n\n"

        pos = 0
        while pos < len(text):
            piece = text[pos:pos + size].strip()
            if piece:
                page_no = max(pg for start, pg in starts if start <= pos)
                chunks.append({**doc_pages[0], "text": piece, "page": page_no})
            if pos + size >= len(text):
                break
            pos += size - overlap
    return chunks


def page_chunks(pages: list[dict], max_chars: int = 1000) -> list[dict]:
    """One chunk per page; split on line breaks only if a page is too long."""
    chunks = []
    for p in pages:
        buf = ""
        for line in p["text"].split("\n"):
            if buf and len(buf) + len(line) + 1 > max_chars:
                chunks.append({**p, "text": buf.strip()})
                buf = line
            else:
                buf = f"{buf}\n{line}" if buf else line
        if buf.strip():
            chunks.append({**p, "text": buf.strip()})
    return chunks


CHUNKERS = {"fixed": fixed_chunks, "page": page_chunks}
