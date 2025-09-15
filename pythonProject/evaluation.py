from typing import List, Dict, Any, Tuple
import numpy as np
import bert_score
from rouge_score import rouge_scorer


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    for h, r in zip(hypotheses, references):
        s = scorer.score(r, h)
        for k in scores:
            scores[k] += s[k].fmeasure
    for k in scores:
        scores[k] /= max(1, len(hypotheses))
    return scores


def compute_bertscore(hypotheses: List[str], references: List[str], lang: str) -> Dict[str, float]:
    P, R, F1 = bert_score.score(hypotheses, references, lang=lang)
    return {
        "precision": float(P.mean().item()),
        "recall": float(R.mean().item()),
        "f1": float(F1.mean().item())
    }

