import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import jsonlines
import numpy as np
from ollama import Client

# Make project modules importable when running as a script
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from My_RAG.config import load_config  # noqa: E402


class OllamaEmbedder:
    """Minimal wrapper around Ollama's embedding endpoint."""

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self.host = host
        self.client = Client(host=host)

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text is empty; cannot embed.")

        response = self.client.embeddings(model=self.model, prompt=text)
        embedding = response.get("embedding")
        if embedding is None:
            raise RuntimeError("Ollama returned no embedding.")

        vec = np.asarray(embedding, dtype=np.float32)
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec


def resolve_ollama_config(model: str | None, host: str | None) -> Tuple[str, str]:
    """Pull defaults from configs if CLI args are not provided."""
    config = load_config()
    retrieval_cfg = config.get("retrieval", {})
    ollama_cfg = config.get("ollama", {})

    resolved_model = (
        model
        or retrieval_cfg.get("embedding_model_path")
        or ollama_cfg.get("model")
        or "qwen-embedding:0.6b"
    )
    resolved_host = host or retrieval_cfg.get("ollama_host") or ollama_cfg.get("host") or "http://localhost:11434"
    return resolved_model, resolved_host


def load_centroids(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError(f"Centroid file at {path} is not a dict.")
    # Re-normalize defensively
    centroids: Dict[str, np.ndarray] = {}
    for domain, vec in data.items():
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        centroids[domain] = arr
    return centroids


def classify(query: str, embedder: OllamaEmbedder, centroids: Dict[str, np.ndarray]) -> str:
    q = embedder.embed(query, normalize=True)
    best_domain = None
    best_score = -1.0
    for domain, centroid in centroids.items():
        score = float(np.dot(q, centroid))
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain or ""


def evaluate(
    queries_path: Path,
    centroids_path: Path,
    model: str | None,
    host: str | None,
    output_path: Path,
) -> None:
    model, host = resolve_ollama_config(model, host)
    print(f"Using Ollama embedding model '{model}' at '{host}'")

    centroids = load_centroids(centroids_path)
    embedder = OllamaEmbedder(model=model, host=host)

    total = 0
    correct = 0
    mistakes: list[dict] = []

    with jsonlines.open(queries_path, "r") as reader:
        for obj in reader:
            total += 1
            true_domain = obj.get("domain")
            query_obj = obj.get("query", {})
            query_text = query_obj.get("content", "")
            query_id = query_obj.get("query_id")
            pred_domain = classify(query_text, embedder, centroids)

            if pred_domain == true_domain:
                correct += 1
            else:
                mistakes.append(
                    {
                        "query_id": query_id,
                        "language": obj.get("language"),
                        "true_domain": true_domain,
                        "pred_domain": pred_domain,
                        "query": query_text,
                    }
                )

    accuracy = correct / total if total else 0.0
    print(f"Evaluated {total} queries | accuracy={accuracy:.4f} | mistakes={len(mistakes)}")

    if mistakes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(output_path, "w") as writer:
            writer.write_all(mistakes)
        print(f"Saved misclassified queries to {output_path}")
    else:
        print("No misclassifications detected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate domain classifier against labeled queries.")
    parser.add_argument(
        "--queries",
        default="dragonball_dataset/dragonball_queries.jsonl",
        help="Path to labeled queries JSONL.",
    )
    parser.add_argument(
        "--centroids",
        default="database/domain_centroids.npy",
        help="Path to saved centroids (.npy).",
    )
    parser.add_argument(
        "--output",
        default="database/domain_misclassified.jsonl",
        help="Where to write misclassified queries.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama embedding model (default: from configs or qwen-embedding:0.6b).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama host URL (default: from configs or http://localhost:11434).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        queries_path=Path(args.queries),
        centroids_path=Path(args.centroids),
        model=args.model,
        host=args.host,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
