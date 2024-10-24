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

# Additionally use query textual descriptions.
# Has different variants
#  - use name vs. use desc
#  - only use cls token vs. use all tokens

import argparse
import logging
from typing import Literal

import torch
from torch.nn import functional as F

import mlflow
from ziracle.data import DocId, Query
from ziracle.experiments.base_experiment import (
    MLFLOW_EXPERIMENT_NAME,
    ExpBlip2BaseModel,
)


class ExpBlip2QueryFusionWeighting(ExpBlip2BaseModel):
    """Implements the Ziracle experiment.

    This executes the image retrieval experiment using the Ziracle method, which employs
    our tempered-weighting query fusion method. This experiment can be used to reproduce
    the results from the Ziracle paper.
    """

    def __init__(
        self,
        model_name: Literal["opt", "feat"],
        target_text: Literal["name", "desc"],
        target_token: Literal["full", "cls"],
        focus_coir_path: str,
        emb_path: str,
        run_name: str,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        temperature: float = 1.0,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """Initialize the Ziracle experiment.

        Parameters
        ----------
        model_name : Literal[&quot;opt&quot;, &quot;feat&quot;]
            BLIP2 model variant to load.
            `opt` loads the BLIP2 Q-former from the opt caption generation model.
            `feat` loads the designated feature extraction Q-former model.
            For the Ziracle experiments, we utilize the `feat` model variant.
        target_text : Literal[&quot;name&quot;, &quot;desc&quot;]
            Target text to use, as described in the paper.
            `name` corresponds to the short description setting.
            `desc` corresponds to the long description setting.
        target_token : Literal[&quot;full&quot;, &quot;cls&quot;]
            Token embedding to use for weight computation.
            `full` uses all token embeddings and applies a rowwise mean to the result.
            `cls` uses only the CLS-token embedding.
        focus_coir_path : str
            Path to the focus-coir dataset query file.
        emb_path : str
            Path to the precomputed embeddings file.
        run_name : str
            Name for the run, used in mlflow.
        experiment_name : str, optional
            Name of the experiment used in mlflow, by default MLFLOW_EXPERIMENT_NAME
        temperature : float, optional
            Temperature for the softmax applied to the weight vector, by default 1.0
        device : Literal[&quot;cuda&quot;, &quot;cpu&quot;], optional
            Device used for the BLIP2 model, by default "cuda"
        """
        super().__init__(
            model_name, focus_coir_path, emb_path, run_name, experiment_name, device
        )
        self.target_text = target_text
        self.target_token = target_token
        self.temperature = float(temperature)
        mlflow.log_params(
            {
                "similarity": "query_fusion_weighting_tempered",
                "model": model_name,
                "target_text": target_text,
                "target_token": target_token,
                "temperature": self.temperature,
            }
        )

    def _get_query_txt_embedding(self, query: Query) -> torch.Tensor:
        match self.target_text:
            case "name":
                txt = query.name_text
            case "desc":
                txt = query.desc_text
            case other:
                msg = f"Invalid value for target_text: `{other}`"
                raise ValueError(msg)
        emb = self.model.embed_text(txt).squeeze()

        match self.target_token:
            case "full":
                pass
            case "cls":
                emb = emb[0].reshape(1, -1)
        return emb

    def _get_similarity_scores(self, query: Query) -> list[tuple[DocId, float]]:
        # Weight the matrix multiplication with the q_weight
        # q_weight weights each of the 32 img emb vectors according to the similarity
        # with the cls token -> steer focus of similarity via text embedding
        target_emb_np = self.emb_dict[query.query_img]
        target_txt_emb = self._get_query_txt_embedding(query).to(self.device)
        target_img_emb = torch.from_numpy(target_emb_np).to(self.device)
        q_emb = F.normalize(target_img_emb)
        t_emb = F.normalize(target_txt_emb)
        q_weight_logits = (q_emb @ t_emb.T).mean(dim=1).reshape(1, -1)
        q_weight = F.softmax(q_weight_logits / self.temperature, dim=1)

        target_db = [doc_id for doc_id, _ in query.labels]
        scores: list[tuple[DocId, float]] = []
        for doc_id in target_db:
            doc_emb = self.emb_dict[doc_id]
            norm_doc_emb = F.normalize(torch.from_numpy(doc_emb).to(self.device))
            _score = (q_emb * q_weight.T) @ (norm_doc_emb * q_weight.T).T
            _score = _score.mean().cpu().detach().item()
            scores.append((doc_id, _score))
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="BLIP2 model variant to use, either `feat` or `opt`.",
    )
    parser.add_argument(
        "--target_text",
        help="Target text to use, as described in the paper. "
        "`name` corresponds to the short description setting. "
        "`desc` corresponds to the long description setting.",
    )
    parser.add_argument(
        "--target_token",
        help="Token embedding to use for weight computation. "
        "`full` uses all token embeddings and applies a rowwise mean to the result. "
        "`cls` uses only the CLS-token embedding.",
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
    parser.add_argument(
        "--temperature",
        help="Temperature for the softmax applied to the weight vector.",
    )
    args = parser.parse_args()

    logging.info("=" * 30)
    logging.info("ZIRACLE Experiment")
    logging.info("=" * 30)

    exp = ExpBlip2QueryFusionWeighting(
        model_name=args.model_name,
        target_text=args.target_text,
        target_token=args.target_token,
        focus_coir_path=args.focus_coir_path,
        emb_path=args.emb_path,
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        temperature=args.temperature,
    )
    exp.run()
