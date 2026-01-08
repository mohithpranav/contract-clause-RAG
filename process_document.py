"""
Main script to process a legal PDF document
Usage: python process_document.py <path_to_pdf>
"""
import sys
from pathlib import Path
from app.document_processor import DocumentProcessor
from app.utils import is_valid_pdf, extract_clause_metadata
from loguru import logger


def process_legal_document(pdf_path: str, output_dir: str = None):
    """
    Process a legal PDF document and extract clauses
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional directory to save processed chunks
    """
    # Validate PDF
    if not is_valid_pdf(pdf_path):
        logger.error(f"Invalid PDF file: {pdf_path}")
        return None
    
    # Initialize processor with optimal settings for legal documents
    processor = DocumentProcessor(
        chunk_size=1000,      # Larger chunks to preserve legal context
        chunk_overlap=200     # Overlap to avoid splitting clauses
    )
    
    # Process the PDF
    logger.info(f"Processing legal document: {pdf_path}")
    documents = processor.process_pdf(
        pdf_path,
        metadata={
            "document_type": "legal_contract",
            "processed_by": "clause-insight"
        }
    )
    
    # Extract clause metadata for each chunk
    for doc in documents:
        clause_meta = extract_clause_metadata(doc.page_content)
        doc.metadata.update(clause_meta)
    
    # Display statistics
    stats = processor.get_chunk_statistics(documents)
    logger.info("Processing complete!")
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Average chunk size: {stats['avg_chunk_size']:.0f} characters")
    
    # Optionally save to file
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save chunks to text file
        output_file = output_path / f"{Path(pdf_path).stem}_chunks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, doc in enumerate(documents, 1):
                f.write(f"{'='*80}\n")
                f.write(f"CHUNK {idx}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Metadata: {doc.metadata}\n\n")
                f.write(doc.page_content)
                f.write(f"\n\n")
        
        logger.info(f"Chunks saved to: {output_file}")
    
    return documents


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        logger.info("Usage: python process_document.py <path_to_pdf> [output_dir]")
        logger.info("\nExample:")
        logger.info("  python process_document.py contract.pdf")
        logger.info("  python process_document.py contract.pdf ./output")
        return
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    documents = process_legal_document(pdf_path, output_dir)
    
    if documents:
        # Show first chunk as preview
        logger.info("\n" + "="*80)
        logger.info("PREVIEW - First Chunk:")
        logger.info("="*80)
        logger.info(f"Metadata: {documents[0].metadata}")
        logger.info(f"\nContent:\n{documents[0].page_content[:500]}...")


if __name__ == "__main__":
    main()
