#!/bin/bash

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

python3 src/ziracle/processing/embed_dataset.py \
    --query_labeling \
    --model_name feat \
    --focus_coir_path data/focus_coir/queries.jsonl \
    --bdd_100k_base_path data/bdd100k/bdd100k/ \
    --emb_save_path data/emb/emb_focus_coir.pkl \
    --batch_size 32
