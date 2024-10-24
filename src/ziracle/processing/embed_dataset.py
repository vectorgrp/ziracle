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
import os
import pickle
from typing import Literal

import jsonlines
import numpy as np
from PIL import Image
from tqdm import tqdm

from ziracle.data import Query
from ziracle.model.blip2_model import get_blip2_model
from ziracle.utils.batch_iterable import batch

logging.basicConfig(level=logging.INFO)


class DatasetEmbedderBlip2:
    """Embed imgrank datasets with Blip2"""

    def __init__(
        self,
        emb_save_path: str,
        model_name: Literal["opt", "feat"] = "feat",
        batch_size: int = 32,
        focus_coir_path: str = "data/focus_coir/queries.jsonl",
        bdd_100k_base_path: str = "data/bdd100k/bdd100k/",
    ) -> None:
        """Initialize the embedder object.

        Parameters
        ----------
        emb_save_path : str
            Desired path where to store the computed embeddings.
        model_name : Literal[&quot;opt&quot;, &quot;feat&quot;], optional
            Which model to use to create the embeddings, by default "feat"
        batch_size : int, optional
            Batch size to use during the embedding computation, by default 32
        focus_coir_path : str, optional
            Path to the focus-coir queries, by default "data/focus_coir/queries.jsonl"
        bdd_100k_base_path : str, optional
            Path to the BDD100k images, by default "data/bdd100k/bdd100k/"
        """
        self.batch_size = batch_size
        self.focus_coir_path = focus_coir_path
        self.emb_save_path = emb_save_path
        if not os.path.exists(os.path.dirname(emb_save_path)):
            os.makedirs(os.path.dirname(emb_save_path))

        self.base_path_train = os.path.join(bdd_100k_base_path, "images/100k/train/")
        self.base_path_test = os.path.join(bdd_100k_base_path, "images/100k/test/")
        self.base_path_val = os.path.join(bdd_100k_base_path, "images/100k/val/")
        self.model = get_blip2_model(model_name)

    def _load_image(self, img_name: str):
        _train_path = os.path.join(self.base_path_train, img_name)
        _test_path = os.path.join(self.base_path_test, img_name)
        _val_path = os.path.join(self.base_path_val, img_name)
        if os.path.exists(_train_path):
            img_path = _train_path
        elif os.path.exists(_test_path):
            img_path = _test_path
        elif os.path.exists(_val_path):
            img_path = _val_path
        else:
            raise ValueError("Image not found")
        img = Image.open(img_path).convert("RGB")
        return img

    def _embed_images(self, img_names: list[str]) -> dict[str, np.ndarray]:
        emb_dict = {}
        for img_name_batch in tqdm(
            batch(img_names, n=self.batch_size),
            total=round(len(img_names) / self.batch_size),
        ):
            imgdata_list = [self._load_image(img) for img in img_name_batch]
            emb_tensor = self.model.embed_image_batch(imgdata_list)
            np_arr = emb_tensor.cpu().detach().numpy()
            for i, img in enumerate(img_name_batch):
                emb_dict[img] = np_arr[i]
        return emb_dict

    def embed_focus_coir(self):
        """Embed only the images used in the focus-coir dataset."""
        # Load required files
        logging.info("Embedding images of the query labeling dataset")
        with open(self.focus_coir_path) as file_in:
            query_list = [Query(**d) for d in jsonlines.Reader(file_in).iter(type=dict)]
        img_names = set()
        for query in query_list:
            query_images, _ = zip(*query.labels)
            img_names.update(query_images)
        # Add query representation images
        img_names.update([q.query_img for q in query_list])
        emb_dict = self._embed_images(list(img_names))
        # Write embeddings to disk
        logging.info("Saving embeddings...")
        with open(self.emb_save_path, "wb") as outfile:
            pickle.dump(emb_dict, outfile)

    def embed_full_bdd100k(self):
        """Embed the full BDD100k dataset."""
        logging.info("Embedding images of the bdd100k train split")
        img_names = os.listdir(self.base_path_train)
        emb_dict = self._embed_images(img_names)
        logging.info("Saving embeddings...")
        with open(self.emb_save_path, "wb") as outfile:
            pickle.dump(emb_dict, outfile)


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create image embeddings using BLIP2.")
    parser.add_argument(
        "--bdd100k",
        action="store_true",
        help="If set, create embeddings for the whole BDD100K dataset.",
    )
    parser.add_argument(
        "--query_labeling",
        action="store_true",
        help="If set, create embeddings for the whole focus_coir dataset.",
    )
    parser.add_argument(
        "--model_name",
        default="feat",
        help="BLIP2 model variant to use, either `feat` or `opt`.",
    )
    parser.add_argument(
        "--focus_coir_path", help="Local path to the focus-coir `queries.jsonl` file."
    )
    parser.add_argument(
        "--bdd_100k_base_path", help="Local path to the BDD100K dataset images."
    )
    parser.add_argument(
        "--emb_save_path", help="Desired save location for the computed embeddings."
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        help="Batch size to use during embedding processing.",
    )
    args = parser.parse_args()

    encoder = DatasetEmbedderBlip2(
        batch_size=int(args.batch_size),
        model_name=args.model_name,
        focus_coir_path=args.focus_coir_path,
        bdd_100k_base_path=args.bdd_100k_base_path,
        emb_save_path=args.emb_save_path,
    )
    if args.query_labeling:
        encoder.embed_focus_coir()
    if args.bdd100k:
        encoder.embed_full_bdd100k()
