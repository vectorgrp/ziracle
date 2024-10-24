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

# ------------------------------------------------------------------------------
# Ziracle experiment
# ------------------------------------------------------------------------------
python3 src/ziracle/experiments/query_fusion_weighting_tempered.py \
    --model_name feat \
    --target_text name \
    --target_token cls \
    --focus_coir_path data/focus_coir/queries.jsonl \
    --emb_path data/emb/emb_focus_coir.pkl \
    --run_name exp_focus_coir_ziracle_t0.001_name_cls \
    --experiment_name ziracle_focus_coir \
    --temperature 0.001

# ------------------------------------------------------------------------------
# Image-only Baseline
# ------------------------------------------------------------------------------
python3 src/ziracle/experiments/baseline.py \
    --model_name feat \
    --focus_coir_path data/focus_coir/queries.jsonl \
    --emb_path data/emb/emb_focus_coir.pkl \
    --run_name exp_focus_coir_baseline \
    --experiment_name ziracle_focus_coir
