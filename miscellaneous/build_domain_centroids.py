import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jsonlines
import numpy as np
from ollama import Client

# Make project modules importable when running as a script
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from My_RAG.config import load_config


class OllamaEmbedder:
    """Minimal wrapper around Ollama's embedding endpoint."""

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self.host = host
        self.client = Client(host=host)

    def verify_connection(self) -> None:
        try:
            _ = self.client.list()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Cannot reach Ollama at {self.host}: {exc}") from exc

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


def resolve_ollama_config(model: Optional[str], host: Optional[str]) -> Tuple[str, str]:
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


def load_queries(paths: Sequence[Path]) -> List[Tuple[str, str]]:
    """Return (domain, text) tuples from JSONL query files."""
    examples: List[Tuple[str, str]] = []
    for path in paths:
        with jsonlines.open(path, "r") as reader:
            for obj in reader:
                domain = obj.get("domain")
                content = obj.get("query", {}).get("content")
                if domain and content:
                    examples.append((domain, content))
    return examples


def build_centroids(
    examples: Iterable[Tuple[str, str]], embedder: OllamaEmbedder, normalize: bool = True
) -> Dict[str, np.ndarray]:
    grouped: defaultdict[str, List[np.ndarray]] = defaultdict(list)

    for domain, text in examples:
        try:
            vec = embedder.embed(text, normalize=normalize)
            grouped[domain].append(vec)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Skipping example in domain '{domain}': {exc}")

    centroids: Dict[str, np.ndarray] = {}
    for domain, vectors in grouped.items():
        if not vectors:
            continue
        stack = np.vstack(vectors)
        centroid = stack.mean(axis=0)
        if normalize:
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
        centroids[domain] = centroid
    return centroids


def save_centroids(centroids: Dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, centroids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build domain centroids from query sets.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "dragonball_dataset/queries_en.jsonl",
            "dragonball_dataset/queries_zh.jsonl",
        ],
        help="Paths to query JSONL files.",
    )
    parser.add_argument(
        "--output",
        default="database/domain_centroids.npy",
        help="Where to save the centroids (.npy).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama embedding model to use (default: pull from configs or qwen-embedding:0.6b).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama host URL (default: pull from configs or http://localhost:11434).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization on embeddings and centroids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    normalize = not args.no_normalize

    model, host = resolve_ollama_config(args.model, args.host)
    print(f"Using Ollama embedding model '{model}' at '{host}'")

    print("📥 Loading queries...")
    examples = load_queries(input_paths)
    print(f"Loaded {len(examples)} query texts from {len(input_paths)} files.")

    embedder = OllamaEmbedder(model=model, host=host)
    try:
        embedder.verify_connection()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ollama connection check failed: {exc}") from exc

    print("🧠 Building centroids...")
    centroids = build_centroids(examples, embedder, normalize=normalize)

    if not centroids:
        raise RuntimeError("No centroids were produced. Check inputs and embeddings.")

    output_path = Path(args.output)
    save_centroids(centroids, output_path)
    domains = ", ".join(sorted(centroids))
    print(f"✅ Saved centroids for domains [{domains}] to {output_path}")


if __name__ == "__main__":
    main()
