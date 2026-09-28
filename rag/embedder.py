import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def chunk_documents(pages, chunk_size=800, chunk_overlap=100):
    
    """Takes the list of pages from load_pdf().
    Splits them into smaller chunks for embedding.
    
    Paramenters:
    pages -> output from load_pdf()
    chunk_size -> max characters per chunk (default 500)
    chunk_overlap -> characters repeated between chunks (default 50)
    
    Returns:
    A list of Document objects (chunks), each with:
    .page_content -> the chunk text
    .metadata -> inherited from its original page (source file, page number)
    
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap,
        #This is the order it tries to split on.
        #It tries paragraph breaks first, then line breaks,
        #then spaces. Only splits mid-word as a last resort.
        separators= ["\n\n", "\n", " ",""],
        length_function=len,
    
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError ("Chunking produced no output. Check your pages list.")
    
    #Print a summary so you can see what happened
    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chunk = total_chars // len(chunks)
    min_chunk = min(len(c.page_content) for c in chunks)
    max_chunk = max(len(c.page_content) for c in chunks)
    
    print(f"✓ Chunking complete")
    print(f"Pages in : {len(pages)}")
    print(f"Chunks out : {len(chunks)}")
    print(f"Avg chunk size : {avg_chunk} chars")
    print(f"Smallest chunk : {min_chunk} chars")
    print(f"Largest chunk : {max_chunk} chars")
    
    return chunks

def deduplicate_chunks(chunks):
    """
    Removes identical or near-identical chunks before embedding.
    A TOC page split into 5 pieces becomes 1 piece.
    Compares page_content strings exactly.
    """
    seen    = set()
    unique  = []
    removed = 0

    for chunk in chunks:
        # Strip whitespace before comparing so minor
        # spacing differences don't create false duplicates
        content = chunk.page_content.strip()

        if content not in seen:
            seen.add(content)
            unique.append(chunk)
        else:
            removed += 1

    if removed > 0:
        print(f"  Removed {removed} duplicate chunks")

    print(f"  Unique chunks remaining: {len(unique)}")
    return unique


def is_toc_page(text):
    """
    Detects table of contents pages.
    TOC entries have patterns like:
    'Section Name ............... 19'
    We detect this by counting dotted leaders.
    If more than 3 lines contain '....', it's likely a TOC page.
    """
    lines      = text.split('\n')
    toc_lines  = sum(1 for line in lines if '....' in line)
    return toc_lines >= 3


def filter_toc_pages(pages):
    """
    Removes table of contents pages from the document
    before chunking so they don't pollute the index.
    """
    filtered = []
    removed  = []

    for page in pages:
        if is_toc_page(page.page_content):
            removed.append(page.metadata.get('page', '?'))
        else:
            filtered.append(page)

    if removed:
        print(f"  Removed TOC pages: {removed}")

    return filtered
def preview_chunks(chunks, n=3):
    """Shows the first n chunks in readable form.
    Use this to sanity-check that splits look sensible.
    """
    print(f"\n --- Previewing first {n} chunks ---")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i+ 1}"
              f"|page {chunk.metadata.get('page', '?')}"
              f"{len(chunk.page_content)} chars")
        print("-"* 40)
        print(chunk.page_content)
        print("-" * 40)
        
        


def get_embedder():
    """
    Creates and returns the embedding model.
    Kept as a separate function so we reuse the
    same embedder everywhere without rewriting config.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview"
    )


def build_vector_store(chunks, persist_directory="./chroma_db"):
    """
    Embeds chunks and stores in ChromaDB.
    Processes in small batches with delays to avoid hitting
    Gemini's 100-requests-per-minute embedding rate limit.
    """
    import time

    BATCH_SIZE = 20   # embed 20 chunks at a time
    DELAY      = 15   # wait 15 seconds between batches

    # This is well within the 100/minute free tier limit:
    # 20 chunks per batch, pause 15s, next 20 chunks.
    # Even for a 100-chunk document = 5 batches = 60s total.

    embedder = get_embedder()
    vectordb = None
    total    = len(chunks)

    print(f"Embedding {total} chunks in batches of {BATCH_SIZE}...")
    print(f"Estimated time: ~{((total // BATCH_SIZE) * DELAY)}s\n")

    for i in range(0, total, BATCH_SIZE):
        batch     = chunks[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        end_idx   = min(i + BATCH_SIZE, total)

        print(f"  Batch {batch_num}: chunks {i+1}–{end_idx} of {total}...")

        if vectordb is None:
            # First batch — create the database
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embedder,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            # Subsequent batches — add to existing database
            vectordb.add_documents(batch)

        # Pause between batches — skip pause after the last batch
        if end_idx < total:
            print(f"  Pausing {DELAY}s to stay within rate limits...")
            time.sleep(DELAY)

    count = vectordb._collection.count()
    print(f"\n✓ {count} vectors stored in ChromaDB")
    print(f"✓ Saved to disk at '{persist_directory}/'")

    return vectordb

def load_vector_store(persist_directory="./chroma_db"):
    """
    Loads an existing ChromaDB from disk.

    Use this on every run AFTER the first one.
    No API calls needed — just reads from disk.
    Much faster than rebuilding from scratch.

    Parameters:
        persist_directory → same folder used in build_vector_store()

    Returns:
        vectordb → a live ChromaDB object ready to search
    """

    embedder = get_embedder()

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedder,
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = vectordb._collection.count()

    if count == 0:
        raise ValueError(
            f"Vector store at '{persist_directory}' is empty. "
            "Run build_vector_store() first to index a document."
        )

    print(f"✓ Loaded vector store from disk ({count} vectors)")
    return vectordb


def search(vectordb, query, k=3):
    """
    Runs a similarity search against the vector store.

    Takes your question as a plain string.
    Embeds the question, finds the k closest chunks.

    Parameters:
        vectordb → from build_vector_store() or load_vector_store()
        query    → your question as a string
        k        → how many chunks to return (default 3)

    Returns:
        List of (Document, score) tuples.
        Score is between 0 and 1. Higher = more similar.
    """

    results = vectordb.similarity_search_with_relevance_scores(
        query, k=k
    )

    return results

def build_vector_store(chunks, persist_directory="./chroma_db", collection_name="langchain"):
    import time
    BATCH_SIZE = 20
    DELAY      = 15
    embedder   = get_embedder()
    vectordb   = None
    total      = len(chunks)

    print(f"Embedding {total} chunks in batches of {BATCH_SIZE}...")
    print(f"Estimated time: ~{((total // BATCH_SIZE) * DELAY)}s\n")

    for i in range(0, total, BATCH_SIZE):
        batch   = chunks[i : i + BATCH_SIZE]
        end_idx = min(i + BATCH_SIZE, total)
        print(f"  Batch {(i//BATCH_SIZE)+1}: chunks {i+1}–{end_idx} of {total}...")

        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embedder,
                persist_directory=persist_directory,
                collection_name=collection_name,
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            vectordb.add_documents(batch)

        if end_idx < total:
            print(f"  Pausing {DELAY}s to stay within rate limits...")
            time.sleep(DELAY)

    count = vectordb._collection.count()
    print(f"\n✓ {count} vectors stored in ChromaDB")
    print(f"✓ Saved to disk at '{persist_directory}/'")
    return vectordb


def load_vector_store(persist_directory="./chroma_db", collection_name="langchain"):
    embedder = get_embedder()
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedder,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )
    count = vectordb._collection.count()
    if count == 0:
        raise ValueError(
            f"Vector store at '{persist_directory}' "
            f"collection '{collection_name}' is empty."
        )
    print(f"✓ Loaded vector store from disk ({count} vectors)")
    return vectordb