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

from dataclasses import dataclass, field

import numpy as np

from .document import DocId, Document
from .query import Query, RelevanceLabel


@dataclass
class RankedDocumentList:
    # Unordered dict of documents
    # Keys are doc ids, values are corresponding document objects
    document_dict: dict[DocId, Document]

    # Hold query information for the ranking
    query: Query

    # Ordered lists of document ids, corresponding to ranking
    ranking: list[DocId] = field(default_factory=lambda: [])
    # Relevance labels for all documents
    relevance_labels: dict[DocId, RelevanceLabel] = field(
        default_factory=lambda: dict()
    )

    def __len__(self) -> int:
        if self.ranking is not None:
            return len(self.ranking)
        else:
            return len(self.document_dict)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, RankedDocumentList):
            return False
        return (
            self.document_dict == __o.document_dict
            and self.query == __o.query
            and self.ranking == __o.ranking
            and self.relevance_labels == __o.relevance_labels
        )

    def __repr__(self) -> str:
        return f"RankedDocumentList(len={len(self.document_dict)})"

    def get_rich_document_ranking(self) -> list[Document]:
        """Return a list of document objects in the order defined in self.ranking."""
        assert self.ranking is not None
        return [self.document_dict[id] for id in self.ranking]

    def aggregate_feature_matrix(self) -> np.ndarray:
        """Aggregate feature vectors of all documents to one matrix"""
        return np.stack([doc.feature_vector for _, doc in self.document_dict.items()])
