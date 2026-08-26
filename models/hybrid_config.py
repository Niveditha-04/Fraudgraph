"""
Single source of truth for the hybrid score weighting -- previously
hardcoded separately in models/run_phase4.py and agent/case_data.py,
which is exactly how a formula fix could silently miss one of the two
places it's used.

Benford's Law deviation, evaluated directly against the ground-truth
label (models/evaluate_hybrid_score.py), predicts illicit status at or
below chance level: standalone AUC-PR 0.038 against a 4.49% base rate.
Blending it into the hybrid score at any positive weight can only pull
the score away from the GNN's own (real, above-chance) signal. The
hybrid score is therefore the GNN score alone until a genuinely
predictive second signal replaces Benford.
"""

HYBRID_WEIGHT_GNN = 1.0
HYBRID_WEIGHT_BENFORD = 0.0
