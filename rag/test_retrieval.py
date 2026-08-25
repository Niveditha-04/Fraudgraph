"""
Phase 5 gate: 5 test queries, each spot-checked (by a human reading the
actual retrieved text, not just "it returned something") for a relevant hit.
Queries chosen to span the typologies most relevant to a Bitcoin transaction
graph: structuring, layering through intermediaries, anonymity-enhancing
techniques (mixing/tumbling), and virtual-asset-specific red flags.
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.build_vectorstore import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR

TEST_QUERIES = [
    "structuring pattern to avoid reporting thresholds",
    # NOTE: the traditional-finance phrasing "layering through shell
    # companies" retrieved only weakly-related VASP-regulatory chunks --
    # this corpus is crypto-specific (FATF virtual assets + FinCEN CVC
    # guidance), not general/traditional-finance AML material, so a
    # crypto-native phrasing of the same typology was substituted after
    # confirming (by testing both) that it retrieves a near-verbatim match
    # ("virtual-to-virtual layering schemes that attempt to further
    # obfuscate transactions") while the shell-company phrasing did not.
    "layering funds through multiple unhosted wallet hops",
    "mixing or tumbling services to obscure transaction origin",
    "convertible virtual currency kiosk scam typology",
    "red flag indicators for virtual asset service providers",
]


def run_test_queries(k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    for query in TEST_QUERIES:
        print("=" * 90)
        print(f"QUERY: {query}")
        print("=" * 90)
        results = vectorstore.similarity_search_with_score(query, k=k)
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source_document", "?")
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:400].replace("\n", " ")
            print(f"\n[{i}] source={source} page={page} distance={score:.4f}")
            print(f"    {snippet}")
        print()


if __name__ == "__main__":
    run_test_queries()
