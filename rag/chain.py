import time
import warnings
warnings.filterwarnings("ignore")
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

PROMPT_TEMPLATE = """You are a helpful document Q&A assistant.
Answer the question using ONLY the information in the context below.
Use all relevant parts of the context to give the most complete
answer possible.

Rules:
- If the context contains the answer directly, state it clearly.
- If the context contains related information that partially answers
  the question, share what you found and note what is missing.
- Only say "I don't know based on the provided document" if the
  context has NO relevant information at all.
- Never guess or use outside knowledge.

Context:
{context}

Question:
{question}

Answer:"""


def get_llm():
    # Primary model — recommended by Google for new users
    primary = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

    # Fallbacks tried in order if primary hits rate limits or errors
    fallbacks = [
        ChatGoogleGenerativeAI(model="gemini-3.7-flash"),
        ChatGoogleGenerativeAI(model="gemini-3.5-flash"),
        ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite"),
        ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite"),
    ]

    # .with_fallbacks() is LangChain's built-in fallback chain.
    # If primary throws any exception, it automatically tries
    # fallbacks[0], then fallbacks[1], and so on.
    # exceptions_to_handle tells it which errors trigger a fallback.
    return primary.with_fallbacks(
        fallbacks,
        exceptions_to_handle=(Exception,)
    )


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(vectordb):
    llm       = get_llm()
    base_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    # MultiQueryRetriever generates multiple phrasings of your question
    # and merges the results — catches things a single search misses
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=get_llm()
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # RunnableParallel runs two things at once:
    # - retrieves relevant chunks and formats them as context
    # - passes the question through unchanged
    # Both outputs feed into the prompt template together
    setup = RunnableParallel(
        context  = retriever | format_docs,
        question = RunnablePassthrough()
    )

    chain = setup | prompt | llm | StrOutputParser()

    print("✓ QA chain built")
    return chain, retriever


def ask(chain, retriever, question):
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            answer  = chain.invoke(question)
            sources = retriever.invoke(question)
            return answer, sources

        except Exception as e:
            error_str = str(e)

            if "503" in error_str:
                wait = 20 * (attempt + 1)
                print(f"Server busy, retrying in {wait}s...")
                time.sleep(wait)

            elif "429" in error_str:
                wait = 40 * (attempt + 1)
                print(f"Rate limited, retrying in {wait}s...")
                time.sleep(wait)

            else:
                raise

    raise Exception("Failed after 3 attempts. Check API quota.")