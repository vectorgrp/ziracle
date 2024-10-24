# Copyright (C) 2024  Vector Informatik GmbH.
#  
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#  
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#  
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from ziracle.data import RankedDocumentList, RelevanceLabel

from .abstract_metric import AbstractBinaryMetric


class AveragePrecisionMetric(AbstractBinaryMetric):
    """Implements average precision
    AP = sum_over_all_ranks_k(P@k*rel(k)) / n_rel_docs_in_k
    P@r is the precision at r
    rel(k) indicates whether document@k is relevant"""

    def __init__(
        self,
        cutoff: int | None,
        relevance_threshold: RelevanceLabel | None,
    ) -> None:
        """Initializes the AveragePrecisionMetric object

        Parameters
        ----------
        cutoff : int | None
            Cutoff the ranking at this index before evaluation. Cutoff is done according
            to python list slicing, e.g. ranking[:cutoff]. If None, perform no cutoff.
        relevance_threshold : RelevanceLabel | None
            Threshold to convert numeric to binary relevance labels, by default None
        """
        super().__init__(relevance_threshold=relevance_threshold)
        self.cutoff = cutoff

    def evaluate_ranking(self, ranked_list: RankedDocumentList) -> float:
        self._warn_on_high_relevance_threshold(ranked_list)
        if (
            self.cutoff is None
            or self.cutoff == -1
            or self.cutoff > len(ranked_list.ranking)
        ):
            self.cutoff = len(ranked_list.ranking)
        rel_labels = [
            ranked_list.relevance_labels[doc_id]
            for doc_id in ranked_list.ranking[: self.cutoff]
        ]
        labels = self._convert_to_binary_labels(rel_labels)
        positive_count = 0
        pr_list = []
        for rank, label in enumerate(labels):
            if label == True:
                positive_count += 1
                _prec_at_rank = positive_count / (rank + 1)
                pr_list.append(_prec_at_rank)
        try:
            ap_score = sum(pr_list) / positive_count
        except ZeroDivisionError:
            ap_score = 0.0
        return ap_score
