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

# ==============================================================================
# Baseline experiment
# ==============================================================================
# Simple matrix multiplication similarity between query image embeddings and the
# database image embeddings.
# Base class, can be used to implement different similarity concepts.

import logging
import pickle
from collections import defaultdict
from statistics import mean
from typing import Literal

import jsonlines
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

import mlflow
from ziracle.data import DocId, Query, RankedDocumentList
from ziracle.metrics import AbstractMetric, AveragePrecisionMetric, PrecisionAtKMetric
from ziracle.model.blip2_model import get_blip2_model

MLFLOW_TRACKING_URI = "./mlflow"
MLFLOW_EXPERIMENT_NAME = "ziracle_focus_coir"

logging.basicConfig(level=logging.INFO)


class ExpBlip2Base:
    """Base implementation for the Ziracle experiments.

    This implements the core functionality of the experiments, that can be reused over
    the different experiment variants."""

    def __init__(
        self,
        # Dataset config
        focus_coir_path: str,
        emb_path: str,
        # mlflow config
        run_name: str,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """Initialize the experiment.

        This loads the experiment data and initializes a run in the mlflow experiment.

        Parameters
        ----------
        focus_coir_path : str
            Path to the focus-coir dataset query file.
        emb_path : str
            Path to the precomputed embeddings file.
        run_name : str
            Name for the run, used in mlflow.
        experiment_name : str, optional
            Name of the experiment used in mlflow, by default MLFLOW_EXPERIMENT_NAME
        device : Literal[&quot;cuda&quot;, &quot;cpu&quot;], optional
            Device used for the BLIP2 model, by default "cuda"
        """
        self.focus_coir_path = focus_coir_path
        self.emb_path = emb_path
        self.run_name = run_name
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self._load_experiment_data()
        self.metrics: dict[str, AbstractMetric] = {
            "avg_prec": AveragePrecisionMetric(None, None),
            "avg_prec_at_20": AveragePrecisionMetric(20, None),
            "prec_at_20": PrecisionAtKMetric(20),
            "prec_at_50": PrecisionAtKMetric(50),
        }
        self._start_mlflow_run(experiment_name)

    def _start_mlflow_run(self, experiment_name: str) -> None:
        """Setup the mlflow experiment and start a run."""
        logging.info("Starting mlflow experiment run")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.start_run(run_name=self.run_name)

    def _load_experiment_data(self) -> None:
        logging.info("Loading Focus-CoIR dataset")
        with open(self.focus_coir_path) as file_in:
            self.query_list = [
                Query(**d) for d in jsonlines.Reader(file_in).iter(type=dict)
            ]
        logging.info("Loading BLIP2 embeddings for dataset")
        emb_dict = np.load(self.emb_path, allow_pickle=True)
        assert isinstance(emb_dict, dict)
        self.emb_dict = emb_dict

    def run(self):
        """Execute the initialized experiment."""
        # init mlflow experiment
        logging.info("Starting experiment execution.")
        query_metrics: list[dict[str, float]] = []
        self.step = 0
        for query in tqdm(self.query_list, desc="Queries"):
            _metrics = self._run_query(query)
            query_metrics.append(_metrics)
            self.step += 1
        # Aggregate and log metrics
        metrics_dict = defaultdict(lambda: [])
        for qdict in query_metrics:
            [metrics_dict[key].append(val) for key, val in qdict.items()]
        aggr_dict = {
            f"{metric}_mean": mean(val_list)
            for metric, val_list in metrics_dict.items()
        }
        mlflow.log_metrics(aggr_dict)
        mlflow.end_run()

    def _run_query(self, query: Query) -> dict[str, float]:
        mlflow.log_metric("query_id", query.id, step=self.step)
        scores = self._get_similarity_scores(query)
        metric_dict = self._evaluate_scores(query, scores)
        mlflow.log_metrics(metric_dict, step=self.step)
        return metric_dict

    def _evaluate_scores(
        self, query: Query, scores: list[tuple[DocId, float]]
    ) -> dict[str, float]:
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked_doc_list = RankedDocumentList(
            document_dict={},
            query=query,
            ranking=[doc_id for doc_id, _ in scores],
            relevance_labels={doc_id: label for doc_id, label in query.labels},
        )
        return {
            name: obj.evaluate_ranking(ranked_doc_list)
            for name, obj in self.metrics.items()
        }

    def _get_similarity_scores(self, query: Query) -> list[tuple[DocId, float]]:
        """Return similarity score for all documents in the query. Only method that
        needs to be changed for different similarity experiments"""
        target_emb = self.emb_dict[query.query_img]
        target_db = [doc_id for doc_id, _ in query.labels]
        q_emb = F.normalize(torch.from_numpy(target_emb).to(self.device))
        scores: list[tuple[DocId, float]] = []
        for doc_id in target_db:
            doc_emb = self.emb_dict[doc_id]
            norm_doc_emb = F.normalize(torch.from_numpy(doc_emb).to(self.device))
            _score = (q_emb @ norm_doc_emb.T).mean().cpu().detach().item()
            scores.append((doc_id, _score))
        return scores


class ExpBlip2BaseModel(ExpBlip2Base):
    """Load the base experiment with BLIP2 model.

    Initializes the base experiment and additionally loads a BLIP2 model to the
    configured device.
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
        """_summary_

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
            Name for the run, used in mlflow.
        experiment_name : str, optional
            Name of the experiment used in mlflow, by default MLFLOW_EXPERIMENT_NAME
        device : Literal[&quot;cuda&quot;, &quot;cpu&quot;], optional
            Device used for the BLIP2 model, by default "cuda"
        """
        super().__init__(focus_coir_path, emb_path, run_name, experiment_name, device)
        logging.info("Loading BLIP2 model")
        self.model = get_blip2_model(model_name)
