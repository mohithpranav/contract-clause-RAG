from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


class LegalDocumentLoader:
    def __init__(self, contracts_dir: str):
        self.contracts_dir = Path(contracts_dir)

        if not self.contracts_dir.exists():
            raise FileNotFoundError(
                f"Contracts directory not found: {self.contracts_dir}"
            )

    def load_pdfs(self) -> List[Dict]:
        
        documents = []

        pdf_files = list(self.contracts_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError("No PDF files found in contracts directory")

        for pdf_path in pdf_files:
            reader = PdfReader(pdf_path)

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()

                if text and text.strip():
                    documents.append({
                        "text": text.strip(),
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_number
                        }
                    })

        return documents
