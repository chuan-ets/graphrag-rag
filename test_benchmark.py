import requests
import time
import os

API_URL = "http://localhost:8080"

# --- Unique vocabulary per document ---
# Each doc has a distinct topic so retrieval can be measured meaningfully
DOC_TOPICS = [
    {
        "name": "doc_1.txt",
        "content": (
            "Transformer architecture uses self-attention mechanisms to process sequences in parallel. "
            "BERT and GPT are popular transformer models used in natural language processing. "
            "Attention heads capture different aspects of linguistic structure. "
            "Pre-training on large corpora enables transfer learning for downstream NLP tasks. "
            "Tokenization converts raw text into subword units using BPE or WordPiece algorithms. "
        ) * 4,
        "query": "transformer BERT attention tokenization pre-training NLP",
        "keywords": {"transformer", "bert", "attention", "tokenization"}
    },
    {
        "name": "doc_2.txt",
        "content": (
            "Kubernetes orchestrates containerized applications across clusters of machines. "
            "Pods are the smallest deployable units in Kubernetes, containing one or more containers. "
            "Helm charts simplify deployment of complex Kubernetes applications using templates. "
            "Service meshes like Istio manage microservice communication and traffic routing. "
            "Docker images are built from Dockerfiles and pushed to container registries. "
        ) * 4,
        "query": "kubernetes pods helm docker containers microservice",
        "keywords": {"kubernetes", "pods", "helm", "docker"}
    },
    {
        "name": "doc_3.txt",
        "content": (
            "Gradient descent optimizes neural network weights by computing loss gradients. "
            "Backpropagation calculates partial derivatives through the computational graph. "
            "Learning rate scheduling adjusts step size during training to improve convergence. "
            "Batch normalization stabilizes training by normalizing layer inputs. "
            "Dropout regularization prevents overfitting by randomly zeroing neuron activations. "
        ) * 4,
        "query": "gradient descent backpropagation learning rate dropout regularization",
        "keywords": {"gradient", "backpropagation", "dropout", "regularization"}
    },
    {
        "name": "doc_4.txt",
        "content": (
            "SQL databases store structured data in tables with rows and columns. "
            "Indexes speed up query execution by avoiding full table scans. "
            "Transactions ensure ACID properties: atomicity, consistency, isolation, durability. "
            "JOIN operations combine rows from multiple tables based on related columns. "
            "Normalization reduces data redundancy by organizing tables into third normal form. "
        ) * 4,
        "query": "SQL database transactions JOIN normalization ACID indexes",
        "keywords": {"sql", "database", "transactions", "normalization"}
    },
    {
        "name": "doc_5.txt",
        "content": (
            "Cryptographic hash functions produce fixed-length digests from arbitrary input data. "
            "RSA encryption relies on the computational difficulty of factoring large prime numbers. "
            "TLS protocol secures network communication using asymmetric and symmetric encryption. "
            "Digital signatures provide authentication and non-repudiation using private keys. "
            "Zero-knowledge proofs allow proving knowledge without revealing the actual secret. "
        ) * 4,
        "query": "cryptography RSA encryption TLS digital signatures zero-knowledge",
        "keywords": {"cryptography", "rsa", "encryption", "tls", "signatures"}
    },
]


def create_topic_files(out_dir="synthetic_docs"):
    os.makedirs(out_dir, exist_ok=True)
    file_paths = []
    for topic in DOC_TOPICS:
        fpath = os.path.join(out_dir, topic["name"])
        with open(fpath, "w") as f:
            f.write(topic["content"])
        file_paths.append(fpath)
    return file_paths


def clear_chroma_collection():
    """Delete and recreate the ChromaDB collection to start fresh."""
    try:
        import chromadb
        from dotenv import load_dotenv
        load_dotenv("backend/.env")
        chroma = chromadb.HttpClient(host="localhost", port=8000)
        try:
            chroma.delete_collection("rag_docs")
            print("Cleared old ChromaDB collection (rag_docs).")
        except Exception:
            pass
    except Exception as e:
        print(f"Warning: Could not clear ChromaDB: {e}")


def ingest_files(file_paths):
    """Ingest files and return set of successfully ingested filenames."""
    ingested = set()
    for fpath in file_paths:
        with open(fpath, "rb") as f:
            files = {"file": (os.path.basename(fpath), f)}
            resp = requests.post(f"{API_URL}/ingest", files=files)
            result = resp.json()
            print(f"Ingest {fpath}: {result}")
            if result.get("status") == "success":
                ingested.add(os.path.basename(fpath))
        time.sleep(5)  # Avoid rate limits during ingest
    return ingested


def benchmark_query(query, method):
    data = {"query": query, "method": method}
    start = time.time()
    resp = requests.post(f"{API_URL}/query", json=data)
    elapsed = time.time() - start
    try:
        result = resp.json()
    except Exception:
        result = {"error": resp.text}
    return elapsed, result


def main():
    print("=== Clearing old data ===")
    clear_chroma_collection()
    
    print("=== Ingesting topic-specific documents ===")
    files = create_topic_files()
    ingested_filenames = ingest_files(files)
    print(f"\nSuccessfully ingested: {ingested_filenames}")

    print("\n=== Benchmarking ===")
    methods = ["vector", "vector+fts", "all"]
    results = {m: [] for m in methods}

    for method in methods:
        print(f"\n--- Method: {method} ---")
        for i, topic in enumerate(DOC_TOPICS):
            # Skip if this doc wasn't ingested in this run
            if topic["name"] not in ingested_filenames:
                print(f"Skipping {topic['name']} (not ingested this run)")
                continue

            query = topic["query"]
            expected_file = topic["name"]
            gt = {expected_file}

            t, res = benchmark_query(query, method)
            time.sleep(3)  # Rate limit buffer

            # Extract filenames from sources — filter to only THIS run's docs
            doc_ids = set()
            if "sources" in res:
                for s in res["sources"]:
                    if isinstance(s, dict):
                        fname = s.get("filename", "")
                        # Only count filenames from this benchmark run
                        if fname and fname in ingested_filenames:
                            doc_ids.add(fname)

            # Precision / Recall / F1
            tp = len(doc_ids & gt)
            fp = len(doc_ids - gt)
            fn = len(gt - doc_ids)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results[method].append({"query": query, "time": t,
                                    "precision": precision, "recall": recall, "f1": f1,
                                    "retrieved": doc_ids, "expected": gt})

            print(f"Query: {query}")
            print(f"  Time: {t:.2f}s | P={precision:.2f} R={recall:.2f} F1={f1:.2f}")
            print(f"  Retrieved (this run): {doc_ids}")
            print(f"  Expected:             {gt}")
            ans = res.get("answer", str(res))
            print(f"  Answer: {str(ans)[:150]}...\n---")

    print("\n=== Summary ===")
    for method in methods:
        pr = [r["precision"] for r in results[method]]
        rc = [r["recall"] for r in results[method]]
        f1s = [r["f1"] for r in results[method]]
        if pr:
            print(f"{method}: Precision={sum(pr)/len(pr):.2f}  Recall={sum(rc)/len(rc):.2f}  F1={sum(f1s)/len(f1s):.2f}")
        else:
            print(f"{method}: No results (all docs may already be ingested)")


if __name__ == "__main__":
    main()
