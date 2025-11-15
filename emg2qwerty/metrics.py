from collections import Counter
from typing import Any

import Levenshtein
import torch
from torchmetrics import Metric

from emg2qwerty.data import LabelData


class CharacterErrorRates(Metric):
    """Character-level error rates metrics based on Levenshtein edit-distance
    between the predicted and target sequences.

    Returns a dictionary with the following metrics:
    - ``CER``: Character Error Rate
    - ``IER``: Insertion Error Rate
    - ``DER``: Deletion Error Rate
    - ``SER``: Substitution Error Rate

    As an instance of ``torchmetric.Metric``, synchronization across all GPUs
    involved in a distributed setting is automatically performed on every call
    to ``compute()``."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.add_state("insertions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("deletions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("substitutions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("target_len", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: LabelData, target: LabelData) -> None:
        # Use Levenshtein.editops rather than Levenshtein.distance to
        # break down errors into insertions, deletions and substitutions.
        editops = Levenshtein.editops(prediction.text, target.text)
        edits = Counter(op for op, _, _ in editops)

        # Update running counts
        self.insertions += edits["insert"]
        self.deletions += edits["delete"]
        self.substitutions += edits["replace"]
        self.target_len += len(target)

        # print(self.insertions, self.deletions, self.substitutions, self.target_len)

    def compute_from_text(self, prediction: str, target: str) -> dict[str, float]:
        # Use Levenshtein.editops rather than Levenshtein.distance to
        # break down errors into insertions, deletions and substitutions.
        editops = Levenshtein.editops(prediction, target)
        edits = Counter(op for op, _, _ in editops)

        # Update running counts
        self.insertions += edits["insert"]
        self.deletions += edits["delete"]
        self.substitutions += edits["replace"]
        self.target_len += len(target)

        ret = self.compute()
        self.reset()
        return ret

    def compute(self) -> dict[str, float]:
        def _error_rate(errors: torch.Tensor) -> float:
            return float(errors.item() / self.target_len.item() * 100.0)

        return {
            "CER": _error_rate(self.insertions + self.deletions + self.substitutions),
            "IER": _error_rate(self.insertions),
            "DER": _error_rate(self.deletions),
            "SER": _error_rate(self.substitutions),
        }




def get_target_len_from_metrics(metrics):
    cer_metric = None
    for name, metric in metrics.items():
        if isinstance(metric, CharacterErrorRates):
            cer_metric = metric
            break

    if cer_metric is not None:
        return cer_metric.target_len.item()
    else:
        print("CharacterErrorRates not found in val_metrics... returning -1")
        return -1