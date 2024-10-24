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

from dataclasses import dataclass

import numpy as np

from ziracle.utils.dataclass_equal import dc_eq

DocId = int | str


@dataclass(eq=False)
class Document:
    # UUID for the given document
    id: DocId
    # Feature vectors between documents must be comparable
    # -> a valid distance metric must be applicable
    feature_vector: np.ndarray

    def __eq__(self, __o: object) -> bool:
        return dc_eq(self, __o)

    def __repr__(self) -> str:
        return f"Document(id={self.id}, feature_vector=array.size({self.feature_vector.shape}))"
