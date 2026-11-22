"""How well do two raters agree? Used to validate an LLM judge against human labels.

Accuracy alone misleads when one class is rare: a judge that always says "no hallucination" is 80%
accurate on data with 20% hallucinations. So we also report precision/recall/F1 for the positive class,
balanced accuracy, Cohen's kappa (agreement beyond chance) and the Matthews correlation coefficient.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass(frozen=True)
class Confusion:
    """Binary confusion matrix. Positive = the property we detect (e.g. "has a hallucination")."""
    tp: int
    fp: int
    fn: int
    tn: int

    @classmethod
    def from_labels(cls, predicted: Sequence[bool], gold: Sequence[bool]) -> "Confusion":
        if len(predicted) != len(gold):
            raise ValueError("predicted and gold must have the same length")
        c = Counter(zip(map(bool, predicted), map(bool, gold)))
        return cls(tp=c[(True, True)], fp=c[(True, False)], fn=c[(False, True)], tn=c[(False, False)])

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:                       # = sensitivity, true positive rate
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def specificity(self) -> float:                  # = true negative rate
        return self.tn / (self.tn + self.fp) if self.tn + self.fp else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2

    @property
    def kappa(self) -> float:
        """Cohen's kappa: (observed agreement - chance agreement) / (1 - chance agreement)."""
        n = self.n
        if not n:
            return 0.0
        po = (self.tp + self.tn) / n
        pe = ((self.tp + self.fp) * (self.tp + self.fn) + (self.fn + self.tn) * (self.fp + self.tn)) / n ** 2
        return (po - pe) / (1 - pe) if pe < 1 else 1.0

    @property
    def mcc(self) -> float:
        """Matthews correlation coefficient: the Pearson correlation between two binary labelings."""
        den = math.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        return (self.tp * self.tn - self.fp * self.fn) / den if den else 0.0

    def summary(self) -> dict[str, float]:
        return {"n": self.n, "precision": self.precision, "recall": self.recall, "f1": self.f1,
                "accuracy": self.accuracy, "balanced_accuracy": self.balanced_accuracy,
                "specificity": self.specificity, "kappa": self.kappa, "mcc": self.mcc,
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for any number of categories (e.g. 1-5 scores treated as classes)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of equal length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
