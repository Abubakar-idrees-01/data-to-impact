"""Metrics. Each module is pure Python with no dependencies:

- text       exact match, token F1 (SQuAD), ROUGE-L, key-fact recall
- retrieval  hit@k, recall@k, precision@k, MRR, nDCG@k
- agreement  confusion matrix, precision/recall/F1, balanced accuracy, Cohen's kappa, MCC
- stats      bootstrap confidence intervals, paired tests (bootstrap, McNemar), correlations
"""
