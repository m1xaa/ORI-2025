from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import networkx as nx


def align_segments_with_user_scores(
        segments: List[Dict[str, Any]],
        user_scores: np.ndarray,
        fps: float
):
    average_frame_importance = user_scores.mean(axis=1)

    labels = []
    number_of_frames = len(average_frame_importance)
    for segment in segments:
        start = segment['start']
        end = segment['end']
        start_idx = max(0, round(start * fps))
        end_idx = min(round(end * fps), number_of_frames - 1)
        segment_window = average_frame_importance[start_idx:end_idx]
        if segment_window.size <= 0:
            labels.append(0.0)
        else:
            labels.append(segment_window.mean())

    return np.array(labels, dtype=float)


def compute_tfidf_scores(texts: List[str]) -> np.ndarray:
    model = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    x = model.fit_transform(texts)
    return np.asarray(x.sum(axis=1)).ravel()


def compute_textrank_scores(texts: List[str]) -> np.ndarray:
    model = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    x = model.fit_transform(texts)
    sim = (x * x.T).toarray()
    np.fill_diagonal(sim, 0.0)
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(graph, max_iter=200)
    return np.array([scores[i] for i in range(len(texts))])


def embed_sentences(texts: List[str], model_name: str):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts)
    return np.asarray(embs)


def make_feature_matrix(segments: List[Dict[str, Any]], embedding_model_name: str):
    texts = [s['text'] for s in segments]

    tfidf = compute_tfidf_scores(texts)
    textrank = compute_textrank_scores(texts)
    text_lengths = np.array([len(text) for text in texts], dtype=float)
    position = np.array([i for i in range(len(segments))])
    embeddings = embed_sentences(texts, embedding_model_name)

    features = np.column_stack([
        embeddings,
        position,
        tfidf,
        textrank,
        text_lengths
    ])

    return features
