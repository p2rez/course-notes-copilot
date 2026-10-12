import sys
from pathlib import Path

from src.ingest import load_folder

term = " ".join(sys.argv[1:]).lower()
for p in load_folder(Path("data")):
    if term in p["text"].lower():
        print(p["source"], "p.", p["page"], "|", p["text"][:80].replace("\n", " "))
