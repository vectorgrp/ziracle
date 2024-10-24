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

from ..utils.dataclass_equal import dc_eq
from .document import DocId

QueryId = int
RelevanceLabel = int


@dataclass(eq=False)
class Query:
    # UUID for the given query
    id: QueryId
    # The query/target image of the query
    query_img: str
    # Link query to candidate images and corresponding labels
    # Tuple consists of
    #   - str - imgname: filename of the candidate image
    #   - int - relevance label: Gives relevance of document for given query
    labels: list[tuple[DocId, RelevanceLabel]]
    # Textual name of the query (short description)
    name_text: str
    # Textual detailed description of the query (long description)
    desc_text: str

    def __eq__(self, __o: object) -> bool:
        return dc_eq(self, __o)
