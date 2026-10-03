"""Descriptive statistics and clustering for an Assignment 1 corpus.

Your job with this module is to *read* the output, not to compute it. The memo
is graded on what you conclude, not on whether you can call numpy.

    from schema import load_corpus
    from corpus_stats import describe, print_report, cluster

    corpus = load_corpus("corpus.jsonl")
    print_report(describe(corpus.records))

    result = cluster(corpus.records, k=6, plot_path="clusters.png")
    for doc in result.sample(cluster_id=3, n=5):
        print(doc["doc_id"], doc["raw_text"][:200])

Embeddings default to TF-IDF with a truncated SVD projection: no download, no
GPU, runs in seconds on any laptop. If your machine can take it, pass
`backend="sbert"` for sentence-transformer embeddings. Either answers the
assignment's question; if you try both, whether the clusters change is itself
worth a sentence in the memo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from schema import MIN_TABULAR_SHARE, is_tabular

__all__ = ["describe", "print_report", "near_duplicates", "embed", "cluster", "ClusterResult"]

_NUMERIC = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*\.?\d*%?")
_WORD = re.compile(r"\b\w+\b")
_DOMAIN = re.compile(r"https?://([^/]+)")


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------

def _percentiles(values: Sequence[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _numeric_density(text: str) -> float:
    """Share of tokens that are numbers. High density means tabular-ish prose."""
    words = _WORD.findall(text)
    return len(_NUMERIC.findall(text)) / len(words) if words else 0.0


def _table_cells(table_json: Any) -> list[Any] | None:
    """Flatten the table shapes we accept into a list of cells, or None."""
    if isinstance(table_json, list) and table_json:
        if all(isinstance(row, dict) for row in table_json):
            keys = {k for row in table_json for k in row}
            return [row.get(k) for row in table_json for k in keys]
        if all(isinstance(row, list) for row in table_json):
            return [cell for row in table_json for cell in row]
    if isinstance(table_json, dict) and table_json:
        if isinstance(table_json.get("rows"), list):
            return _table_cells(table_json["rows"])
        if all(isinstance(v, list) for v in table_json.values()):
            return [cell for column in table_json.values() for cell in column]
    return None


def _table_shape(table_json: Any) -> tuple[int, int] | None:
    """(rows, cols) for the table shapes we accept."""
    if isinstance(table_json, list) and table_json:
        if all(isinstance(row, dict) for row in table_json):
            return len(table_json), len({k for row in table_json for k in row})
        if all(isinstance(row, list) for row in table_json):
            return len(table_json), max(len(row) for row in table_json)
    if isinstance(table_json, dict) and table_json:
        if isinstance(table_json.get("rows"), list):
            header = table_json.get("columns") or table_json.get("header") or []
            shape = _table_shape(table_json["rows"])
            return (shape[0], len(header) or shape[1]) if shape else (len(table_json["rows"]), len(header))
        if all(isinstance(v, list) for v in table_json.values()):
            return max(len(v) for v in table_json.values()), len(table_json)
    return None


def near_duplicates(texts: Sequence[str], threshold: float = 0.90) -> list[tuple[int, int, float]]:
    """Pairs of near-identical documents, by word-shingle cosine similarity.

    Catches the common failure: paginated sources fetched twice, or a portal
    that serves the same record under several URLs. Word 3-to-5-grams rather
    than character n-grams, because public records share boilerplate sentences
    that character n-grams score as near-identity.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if sum(1 for t in texts if t and t.strip()) < 2:
        return []
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(3, 5), max_features=50000, min_df=1)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:  # every document too short to produce a 3-gram
        return []
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)
    rows, cols = np.where(similarity >= threshold)
    pairs = [(int(i), int(j), float(similarity[i, j])) for i, j in zip(rows, cols) if i < j]
    return sorted(pairs, key=lambda p: -p[2])


def describe(records: Sequence[dict], dup_threshold: float = 0.90) -> dict:
    """Everything the memo needs about corpus shape, in one dictionary."""
    n = len(records)
    if n == 0:
        return {"n_documents": 0}

    texts = [r.get("raw_text") or "" for r in records]
    tabular = [r for r in records if is_tabular(r)]
    shapes = [s for s in (_table_shape(r.get("table_json")) for r in tabular) if s]
    missingness = []
    for r in tabular:
        cells = _table_cells(r.get("table_json"))
        if cells:
            empty = sum(1 for c in cells if c is None or (isinstance(c, str) and not c.strip()))
            missingness.append(empty / len(cells))

    def count(values):
        out: dict[str, int] = {}
        for v in values:
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def domain(url):
        match = _DOMAIN.match(url or "")
        return match.group(1).lower() if match else "(unparsed)"

    modalities = count(str(r.get("modality")) for r in records)
    domains = count(domain(r.get("source_url")) for r in records)
    licenses = count((r.get("license_note") or "").strip()[:60] or "(empty)" for r in records)
    dates = sorted(d for d in (str(r.get("retrieved_at") or "")[:10] for r in records)
                   if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d))

    duplicate_pairs = near_duplicates(texts, dup_threshold)
    duplicated_docs = {i for pair in duplicate_pairs for i in pair[:2]}

    return {
        "n_documents": n,
        "tabular_share": round(len(tabular) / n, 4),
        "tabular_share_ok": (len(tabular) / n) >= MIN_TABULAR_SHARE,
        "modality_counts": modalities,
        "char_length": _percentiles([len(t) for t in texts]),
        "word_length": _percentiles([len(_WORD.findall(t)) for t in texts]),
        "numeric_density": _percentiles([_numeric_density(t) for t in texts]),
        "table_rows": _percentiles([s[0] for s in shapes]) if shapes else {},
        "table_cols": _percentiles([s[1] for s in shapes]) if shapes else {},
        "table_missingness": _percentiles(missingness) if missingness else {},
        "n_sources": len(domains),
        "documents_per_source": domains,
        "source_concentration": round(max(domains.values()) / n, 3),
        "license_notes": licenses,
        "distinct_license_notes": len(licenses),
        "retrieved_span": (dates[0], dates[-1]) if dates else None,
        "near_duplicate_pairs": len(duplicate_pairs),
        "near_duplicate_rate": round(len(duplicated_docs) / n, 4),
        "near_duplicate_examples": [
            (records[i].get("doc_id"), records[j].get("doc_id"), round(score, 3))
            for i, j, score in duplicate_pairs[:10]
        ],
        "documents_without_metadata": sum(1 for r in records if not r.get("metadata")),
    }


def _fmt(stats: dict) -> str:
    if not stats:
        return "  (none)"
    return (f"  min {stats['min']:>10,.1f}   p25 {stats['p25']:>10,.1f}   "
            f"median {stats['median']:>10,.1f}\n"
            f"  p75 {stats['p75']:>10,.1f}   p95 {stats['p95']:>10,.1f}   "
            f"max    {stats['max']:>10,.1f}")


def print_report(stats: dict) -> None:
    """Human-readable version of `describe`. Paste-able into the memo appendix."""
    if not stats.get("n_documents"):
        print("Empty corpus.")
        return

    print("=" * 68)
    print("CORPUS CHARACTERIZATION")
    print("=" * 68)
    print(f"\nDocuments: {stats['n_documents']}")
    print(f"Sources:   {stats['n_sources']}  (largest contributes {stats['source_concentration']:.0%})")
    if stats.get("retrieved_span"):
        print(f"Retrieved: {stats['retrieved_span'][0]} to {stats['retrieved_span'][1]}")

    mark = "OK" if stats["tabular_share_ok"] else f"BELOW {MIN_TABULAR_SHARE:.0%} REQUIREMENT"
    print(f"\nTabular share: {stats['tabular_share']:.1%}  [{mark}]")
    print(f"Modalities:    {stats['modality_counts']}")

    print("\nDocument length, characters")
    print(_fmt(stats["char_length"]))
    print("\nDocument length, words")
    print(_fmt(stats["word_length"]))
    print("\nNumeric density (share of tokens that are numbers)")
    print(_fmt(stats["numeric_density"]))
    if stats.get("table_rows"):
        print("\nTable rows")
        print(_fmt(stats["table_rows"]))
        print("\nTable columns")
        print(_fmt(stats["table_cols"]))
    if stats.get("table_missingness"):
        print("\nTable cell missingness")
        print(_fmt(stats["table_missingness"]))

    print(f"\nNear-duplicate pairs: {stats['near_duplicate_pairs']}  "
          f"({stats['near_duplicate_rate']:.1%} of documents involved)")
    for a, b, score in stats["near_duplicate_examples"][:5]:
        print(f"  {score:.3f}  {a}  ~  {b}")

    print(f"\nDistinct license notes: {stats['distinct_license_notes']}")
    for note, n in list(stats["license_notes"].items())[:5]:
        print(f"  {n:>6}  {note}")
    print("\nDocuments per source")
    for domain, n in list(stats["documents_per_source"].items())[:10]:
        print(f"  {n:>6}  {domain}")
    if stats["documents_without_metadata"]:
        print(f"\nDocuments with empty metadata: {stats['documents_without_metadata']}")

    print("\n" + "-" * 68)
    print("Questions the memo has to answer, which this output does not:")
    print("  * Is the source concentration above a defensible level?")
    print("  * Do the near-duplicates represent real reissued records, or a")
    print("    collection bug? Open two pairs and say which.")
    print("  * What kinds of questions can this corpus not answer?")
    print("-" * 68)


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

def embed(texts: Sequence[str], backend: str = "tfidf", dims: int = 128) -> np.ndarray:
    """Document vectors. Default backend needs no download and no GPU."""
    if backend == "tfidf":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import Normalizer

        vectorizer = TfidfVectorizer(max_features=30000, stop_words="english", ngram_range=(1, 2),
                                     min_df=2 if len(texts) > 20 else 1, sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
        n_components = int(min(dims, max(2, min(matrix.shape) - 1)))
        reduced = TruncatedSVD(n_components=n_components, random_state=0).fit_transform(matrix)
        return Normalizer().fit_transform(reduced)

    if backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("pip install sentence-transformers, or use backend='tfidf'") from exc
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return np.asarray(model.encode(list(texts), show_progress_bar=True, normalize_embeddings=True))

    raise ValueError(f"Unknown backend {backend!r}. Use 'tfidf' or 'sbert'.")


@dataclass
class ClusterResult:
    labels: np.ndarray
    coords: np.ndarray
    records: list[dict]
    k: int
    top_terms: dict[int, list[str]] = field(default_factory=dict)
    silhouette: float | None = None

    def sizes(self) -> dict[int, int]:
        return {int(c): int((self.labels == c).sum()) for c in sorted(set(self.labels.tolist()))}

    def sample(self, cluster_id: int, n: int = 5, seed: int = 0) -> list[dict]:
        """A few documents from one cluster -- the assignment asks you to read them."""
        rng = np.random.default_rng(seed)
        idx = np.where(self.labels == cluster_id)[0]
        if idx.size == 0:
            return []
        chosen = rng.choice(idx, size=min(n, idx.size), replace=False)
        return [self.records[i] for i in sorted(chosen.tolist())]

    def crosstab(self, field_path: str = "metadata.document_type") -> dict:
        """Cross-tabulate clusters against a (possibly nested) record field."""
        parts = field_path.split(".")
        table: dict[int, dict[str, int]] = {}
        for label, record in zip(self.labels.tolist(), self.records):
            value: Any = record
            for part in parts:
                value = value.get(part) if isinstance(value, dict) else None
            row = table.setdefault(int(label), {})
            row[str(value)] = row.get(str(value), 0) + 1
        return table

    def print_summary(self) -> None:
        print(f"\n{self.k} clusters"
              + (f", silhouette {self.silhouette:.3f}" if self.silhouette is not None else ""))
        for cluster_id, size in self.sizes().items():
            terms = ", ".join(self.top_terms.get(cluster_id, [])[:8])
            print(f"  cluster {cluster_id:>2}  n={size:>4}   {terms}")
        print("\nName each cluster yourself after reading a few documents from it:")
        print("    for doc in result.sample(cluster_id=0, n=5): print(doc['raw_text'][:300])")


def cluster(records: Sequence[dict], k: int = 6, *, backend: str = "tfidf",
            plot_path: str | None = None, seed: int = 0) -> ClusterResult:
    """K-means over document embeddings, with an optional 2-D plot.

    The plot is a projection, so distances in it are indicative, not exact.
    Treat overlapping clusters as a prompt to read documents, not as a verdict.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score

    texts = [r.get("raw_text") or "" for r in records]
    if len(texts) < k:
        raise ValueError(f"Cannot make {k} clusters from {len(texts)} documents.")

    vectors = embed(texts, backend=backend)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vectors)

    score = None
    if 1 < k < len(texts):
        try:
            score = float(silhouette_score(vectors, labels))
        except ValueError:
            score = None

    coords = PCA(n_components=2, random_state=seed).fit_transform(vectors)

    # The terms most distinctive to each cluster -- a starting point for naming it.
    top_terms: dict[int, list[str]] = {}
    try:
        vec = TfidfVectorizer(max_features=20000, stop_words="english",
                              min_df=2 if len(texts) > 20 else 1)
        matrix = vec.fit_transform(texts)
        vocabulary = np.array(vec.get_feature_names_out())
        for cluster_id in sorted(set(labels.tolist())):
            centroid = np.asarray(matrix[labels == cluster_id].mean(axis=0)).ravel()
            top_terms[int(cluster_id)] = vocabulary[centroid.argsort()[::-1][:10]].tolist()
    except ValueError:
        pass

    result = ClusterResult(labels=labels, coords=coords, records=list(records), k=k,
                           top_terms=top_terms, silhouette=score)
    if plot_path:
        _plot(result, plot_path)
    return result


def _plot(result: ClusterResult, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    palette = plt.cm.tab10(np.linspace(0, 1, max(result.k, 2)))
    for cluster_id in sorted(set(result.labels.tolist())):
        mask = result.labels == cluster_id
        terms = ", ".join(result.top_terms.get(int(cluster_id), [])[:3])
        ax.scatter(result.coords[mask, 0], result.coords[mask, 1], s=16, alpha=0.75,
                   color=palette[int(cluster_id) % len(palette)],
                   label=f"{cluster_id}: {terms}"[:44])
    ax.set_title(f"{result.k} clusters, PCA projection"
                 + (f" (silhouette {result.silhouette:.2f})" if result.silhouette is not None else ""))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")
