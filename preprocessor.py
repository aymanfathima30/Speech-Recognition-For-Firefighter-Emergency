"""
UrgencyDetector
---------------
NLP module that classifies transcribed firefighter speech into urgency levels
using semantic similarity, keyword matching, and sentence embeddings.

Urgency Levels:
  3 - CRITICAL  : Immediate life threat / mayday call
  2 - HIGH      : Active fire / trapped personnel
  1 - MODERATE  : Situation update / resource request
  0 - ROUTINE   : Status check / non-emergency comms
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Urgency lexicon — domain-specific fire service terminology
# ---------------------------------------------------------------------------
URGENCY_KEYWORDS: Dict[int, List[str]] = {
    3: [
        "mayday", "firefighter down", "lost", "trapped", "running out of air",
        "structural collapse", "disoriented", "emergency evacuation", "sos",
        "man down", "no egress", "missing crew", "collapse imminent",
    ],
    2: [
        "rapid fire spread", "multiple casualties", "exposure threatened",
        "backdraft", "flashover", "victim located", "rescue in progress",
        "vent required", "fire through roof", "second alarm",
    ],
    1: [
        "water supply needed", "additional crew", "equipment malfunction",
        "smoke showing", "investigation", "defensive operations",
        "hydrant problem", "staging area", "command post update",
    ],
    0: [
        "status check", "all clear", "negative", "standing by",
        "copy that", "en route", "scene secure", "no fire found",
    ],
}

# Anchor sentences per level for semantic similarity (TF-IDF space)
ANCHOR_SENTENCES: Dict[int, str] = {
    3: "firefighter is down trapped with no air mayday emergency",
    2: "rapid fire spreading multiple victims need rescue now",
    1: "requesting additional resources water supply equipment update",
    0: "everything is clear standing by no emergency at this time",
}


@dataclass
class UrgencyResult:
    level: int
    label: str
    score: float
    matched_keywords: List[str]
    semantic_scores: Dict[int, float]

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "label": self.label,
            "score": round(self.score, 4),
            "matched_keywords": self.matched_keywords,
        }


class UrgencyDetector:
    LABELS = {3: "CRITICAL", 2: "HIGH", 1: "MODERATE", 0: "ROUTINE"}

    def __init__(self):
        # Build TF-IDF vectorizer fitted on all anchor sentences
        corpus = list(ANCHOR_SENTENCES.values())
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.vectorizer.fit(corpus)
        self._anchor_vectors = {
            level: self.vectorizer.transform([text])
            for level, text in ANCHOR_SENTENCES.items()
        }

    def _keyword_score(self, text: str) -> Dict[int, float]:
        """Return keyword match count per urgency level."""
        text_lower = text.lower()
        scores = {}
        matched = {}
        for level, keywords in URGENCY_KEYWORDS.items():
            hits = [kw for kw in keywords if kw in text_lower]
            scores[level] = len(hits)
            matched[level] = hits
        return scores, matched

    def _semantic_score(self, text: str) -> Dict[int, float]:
        """Cosine similarity between input and each urgency anchor."""
        vec = self.vectorizer.transform([text.lower()])
        return {
            level: float(cosine_similarity(vec, anchor)[0][0])
            for level, anchor in self._anchor_vectors.items()
        }

    def classify(self, transcription: str) -> dict:
        """
        Classify urgency by combining keyword matching and semantic similarity.
        Keyword matches are weighted higher (0.6) than semantic (0.4).
        """
        kw_counts, kw_matched = self._keyword_score(transcription)
        sem_scores = self._semantic_score(transcription)

        # Normalise keyword counts
        max_kw = max(kw_counts.values()) or 1
        kw_norm = {lvl: cnt / max_kw for lvl, cnt in kw_counts.items()}

        # Combined score
        combined = {
            lvl: 0.6 * kw_norm[lvl] + 0.4 * sem_scores[lvl]
            for lvl in range(4)
        }

        best_level = max(combined, key=combined.get)
        all_matched = [kw for hits in kw_matched.values() for kw in hits]

        result = UrgencyResult(
            level=best_level,
            label=self.LABELS[best_level],
            score=combined[best_level],
            matched_keywords=all_matched,
            semantic_scores=sem_scores,
        )
        return result.to_dict()
