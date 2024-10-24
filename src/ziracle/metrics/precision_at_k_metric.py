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


class PrecisionAtKMetric(AbstractBinaryMetric):
    """Implements precision@k metric for list ranking evaluation.
    A document with a relevance label greater than relevance_threshold is considered
    as positive.
    Precision@k is the relative amount of positives in the to k samples of the ranking.
    """

    def __init__(
        self,
        cutoff: int,
        relevance_threshold: RelevanceLabel | None = None,
    ) -> None:
        """Initializes the PrecisionAtKMetric object

        Parameters
        ----------
        cutoff : int
            Corresponds to the k in the metric name. Cutoff the ranking and assess the
            precision within the remaining ranking partition.
        relevance_threshold : RelevanceLabel | None, optional
            Threshold to convert numeric to binary relevance labels, by default None
        """
        super().__init__(relevance_threshold=relevance_threshold)
        self.cutoff = cutoff

    def evaluate_ranking(self, ranked_list: RankedDocumentList) -> float:
        self._warn_on_high_relevance_threshold(ranked_list)
        if self.cutoff == -1 or self.cutoff > len(ranked_list.ranking):
            self.cutoff = len(ranked_list.ranking)
        rel_labels = [
            ranked_list.relevance_labels[doc_id]
            for doc_id in ranked_list.ranking[: self.cutoff]
        ]
        labels = self._convert_to_binary_labels(rel_labels)
        true_count = sum(labels)
        total_count = len(labels)
        pak_score = true_count / total_count
        return pak_score
