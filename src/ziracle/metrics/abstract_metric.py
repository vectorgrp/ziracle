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

import abc
import warnings

from ziracle.data import RankedDocumentList, RelevanceLabel


class AbstractMetric(abc.ABC):
    """Base class for metrics."""

    @abc.abstractmethod
    def evaluate_ranking(self, ranked_list: RankedDocumentList) -> float:
        """Evaluate the given ranking.

        Parameters
        ----------
        ranked_list : RankedDocumentList
            document list object containing ranking as well as ground truth
            relevance labels

        Returns
        -------
        float
            Evaluated metric value
        """


class AbstractBinaryMetric(AbstractMetric):
    """Base class for binary metrics, metrics that are only evaluated on binary labels.
    Provides utility for relevance thresholding, to convert numeric labels to binary
    labels for evaluation.
    """

    def __init__(self, relevance_threshold: RelevanceLabel | None = None) -> None:
        super().__init__()
        if relevance_threshold == None:
            relevance_threshold = 1
        self.relevance_threshold = relevance_threshold

    def _warn_on_high_relevance_threshold(
        self, ranked_list: RankedDocumentList
    ) -> None:
        """Check if numeric relevance threshold is higher than all labels in list
        If true, issue a UserWarning
        """
        all_relevance_levels = set(ranked_list.relevance_labels.values())
        if max(all_relevance_levels) < self.relevance_threshold:
            warnings.warn(
                f"Relevance level {self.relevance_threshold} "
                "is higher than all available labels. Set too high in config?",
                category=UserWarning,
            )

    def _convert_to_binary_labels(
        self, labels: list[RelevanceLabel]
    ) -> list[RelevanceLabel]:
        return [rel >= self.relevance_threshold for rel in labels]
