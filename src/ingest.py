from pathlib import Path
import pymupdf


def load_pdf(path: Path, course: str) -> list[dict]:
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({
                    "text": text,
                    "source": path.name,
                    "page": i,
                    "course": course,
                })
    return pages


def load_folder(root: Path) -> list[dict]:
    pages = []
    for pdf in sorted(root.glob("*/*.pdf")):
        pages.extend(load_pdf(pdf, course=pdf.parent.name))
    return pages
