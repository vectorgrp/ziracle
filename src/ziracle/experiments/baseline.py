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

import argparse
import logging
from typing import Literal

import mlflow
from ziracle.experiments.base_experiment import MLFLOW_EXPERIMENT_NAME, ExpBlip2Base


class ExpBlip2Baseline(ExpBlip2Base):
    """Implements the baseline BLIP2 experiment.

    This executes the image retrieval experiment using an image only BLIP2 image
    retrieval method, that is used as a baseline for comparison.
    """

    def __init__(
        self,
        model_name: Literal["opt", "feat"],
        focus_coir_path: str,
        emb_path: str,
        run_name: str,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """Initialize the baseline experiment.

        Parameters
        ----------
        model_name : Literal[&quot;opt&quot;, &quot;feat&quot;]
            BLIP2 model variant to load.
            `opt` loads the BLIP2 Q-former from the opt caption generation model.
            `feat` loads the designated feature extraction Q-former model.
            For the Ziracle experiments, we utilize the `feat` model variant.
        focus_coir_path : str
            Path to the focus-coir dataset query file.
        emb_path : str
            Path to the precomputed embeddings file.
        run_name : str
            Name for the experiment run, used in mlflow.
        experiment_name : str, optional
            Name of the experiment used in mlflow, by default MLFLOW_EXPERIMENT_NAME
        device : Literal[&quot;cuda&quot;, &quot;cpu&quot;], optional
            Device used for the BLIP2 model, by default "cuda"
        """
        super().__init__(focus_coir_path, emb_path, run_name, experiment_name, device)
        mlflow.log_params(
            {
                "model": model_name,
                "similarity": "baseline_img_sim",
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="BLIP2 model variant to use, either `feat` or `opt`.",
    )
    parser.add_argument(
        "--focus_coir_path", help="Local path to the focus-coir `queries.jsonl` file."
    )
    parser.add_argument(
        "--emb_path", help="Local path to the precomputed embeddings file."
    )
    parser.add_argument(
        "--run_name", help="Name for the experiment run, used in mlflow."
    )
    parser.add_argument(
        "--experiment_name", help="Name of the experiment used in mlflow."
    )
    args = parser.parse_args()
    logging.info("=" * 30)
    logging.info("Baseline Experiment")
    logging.info("=" * 30)

    exp = ExpBlip2Baseline(
        model_name=args.model_name,
        focus_coir_path=args.focus_coir_path,
        emb_path=args.emb_path,
        run_name=args.run_name,
        experiment_name=args.experiment_name,
    )
    exp.run()
