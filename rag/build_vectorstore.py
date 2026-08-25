"""
Phase 5: chunk and embed the AML typology PDFs into a Chroma vector store.

Embedding model choice: a local HuggingFace sentence-transformer
(all-MiniLM-L6-v2) via langchain-huggingface, not an OpenAI/hosted API.
No API key required, and it keeps the RAG layer usable offline -- consistent
with Phase 7's requirement that the demo work "even without live API
credits." The investigation agent's LLM calls (Phase 6) still need a real
LLM API; only the embedding step is local.

Documents (rag/documents/, all verified as genuine downloaded PDFs, not
placeholder/blocked pages -- FATF's site sits behind Cloudflare bot
detection that blocks plain HTTP clients, so the two FATF PDFs were sourced
via the Internet Archive's cached copies of the same public FATF URLs, not
mirrors from a different/unverified source):
  - FATF: Virtual Assets Red Flag Indicators of ML/TF (Sept 2020)
  - FATF: Updated Guidance for a Risk-Based Approach to Virtual Assets
    and VASPs (Oct 2021)
  - FinCEN: Advisory on Illicit Activity Involving Convertible Virtual
    Currency, FIN-2019-A003 (May 2019)
  - FinCEN: Guidance on Application of Regulations to CVC Business Models,
    FIN-2019-G001 (May 2019)
  - FinCEN: Notice on CVC Kiosks for Scam Payments, FIN-2025-NTC1 (Aug 2025)
"""
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "rag/documents"
PERSIST_DIR = "rag/chroma_db"
COLLECTION_NAME = "aml_typologies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def build_vectorstore() -> Chroma:
    pdf_files = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf"))
    print(f"found {len(pdf_files)} PDFs in {DOCS_DIR}: {pdf_files}")
    assert len(pdf_files) >= 5, f"expected >=5 AML typology PDFs, found {len(pdf_files)}"

    all_docs = []
    for fname in pdf_files:
        path = os.path.join(DOCS_DIR, fname)
        loader = PyPDFLoader(path)
        pages = loader.load()
        for p in pages:
            p.metadata["source_document"] = fname
        all_docs.extend(pages)
        print(f"  {fname}: {len(pages)} pages loaded")

    print(f"\ntotal pages loaded: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    print(f"total chunks after splitting: {len(chunks)}")

    print(f"\nembedding with {EMBEDDING_MODEL} (local, no API key) ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"\nvector store built and persisted to {PERSIST_DIR} ({len(chunks)} chunks)")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
