import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """
    Takes a path to a PDF file.
    Returns a list of Document objects — one per page.
    Each Document has two things:
        .page_content  → the raw text of that page
        .metadata      → dict with 'source' and 'page' number
    """

    # Check the file actually exists before trying to open it
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at: {file_path}")

    # Check it's actually a PDF
    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Expected a .pdf file, got: {file_path}")

    # Load it
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Check we actually got text back
    # If this returns empty, it's almost certainly a scanned PDF
    if not pages:
        raise ValueError(
            "PDF loaded but no text was extracted. "
            "This is usually a scanned PDF (image-based). "
            "This project requires text-based PDFs."
        )

    # Check for pages that have very little text (common with scanned docs)
    empty_pages = [i for i, p in enumerate(pages) if len(p.page_content.strip()) < 20]
    if empty_pages:
        print(f"Warning: {len(empty_pages)} page(s) had very little text "
              f"(pages: {empty_pages}). These may be images or blank pages.")

    print(f"✓ Loaded {len(pages)} page(s) from '{os.path.basename(file_path)}'")
    return pages


def get_pdf_stats(pages):
    """
    Prints a readable summary of what was loaded.
    Useful for debugging and understanding your document.
    """
    total_chars = sum(len(p.page_content) for p in pages)
    '''total_chars = 0

for p in pages:
    total_chars += len(p.page_content)'''
    avg_chars = total_chars // len(pages)

    print(f"\n--- PDF Stats ---")
    print(f"Pages loaded   : {len(pages)}")
    print(f"Total chars    : {total_chars:,}")
    print(f"Avg per page   : {avg_chars:,} chars")
    print(f"Source file    : {pages[0].metadata.get('source', 'unknown')}")
    print(f"\nFirst 300 chars of page 1:")
    print("-" * 40)
    print(pages[0].page_content[:300])
    print("-" * 40)