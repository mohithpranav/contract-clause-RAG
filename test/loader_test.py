import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.loader import LegalDocumentLoader

loader = LegalDocumentLoader("data/contracts")
docs = loader.load_pdfs()

print(f"Loaded {len(docs)} pages")
print(docs[0]["metadata"])
print(docs[0]["text"][:300])

