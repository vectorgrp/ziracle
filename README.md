# Ziracle

This repository contains the official implementation for the ZIRACLE method, as well as the Focus-CoIR dataset.
The implementation and dataset are in correspondence to our paper "ZIRACLE: Zero-shot composed Image Retrieval with Advanced Component-Level Emphasis", presented at S+SSPR 2024 Venice and released in the Springer LNCS series.

## Paper Abstract

Image retrieval systems often struggle to align with a user's search intent, particularly when the query image is composed of multiple semantic concepts.

To address this challenge, we propose a novel tempered-weighting query-fusion approach that leverages the capabilities of pre-trained vision-language models (VLMs) to focus on specific image components expressed through textual descriptions.
Notably, our method utilizes the BLIP2 model's embeddings without requiring additional training, enabling zero-shot Composed Image Retrieval (CoIR).

For systematic evaluation of our approach and comparison, we introduce the Focus-CoIR dataset, a novel resource derived from the BDD100K dataset that adopts a focusing setting and provides multiple ground truth positive labels for each target image and corresponding textual description.
Our experimental results on Focus-CoIR demonstrate the effectiveness of our approach, outperforming state-of-the-art CoIR methods across all evaluated ranking metrics.

This work highlights the potential of pre-trained VLMs for CoIR and contributes to the advancement of CoIR research. We make the Focus-CoIR dataset and our implementation publicly available to support future research in this domain.

## Usage

To use this implementation, please clone the repository to your desired location.

### Dependencies

The dependencies for this project are managed via pdm.
Please refer to the [pdm documentation](https://pdm-project.org/en/latest/#installation) for install instructions regarding pdm.

To install the project with all necessary dependencies, run the following command:

```shell
pdm install -G:all
```

This will create a new venv within the project directory and install all required dependencies according to the `pyproject.toml` and the `pdm.lock`.

### Datasets

The experiments in this repo depend on our released `Focus-CoIR` dataset, which is released together with this repository.
You can find the dataset in the [data/focus_coir](data/focus_coir/) directory.
The dataset consists of the `queries.jsonl` file in the jsonlines format, where each line corresponds to a single image query, serialized as json.
For further details, please refer to the [dataset README](data/README.md)

The experiment execution is based on BLIP2 image embeddings for the Focus-CoIR dataset images.
To get the embeddings, you can reproduce them yourself, using the scripts provided in this repository.
Additionally, we provide the precomputed embeddings on request.
For this, please contact us via [email](mailto:khanlian.chung@vector.com).

If you want to reproduce the image embeddings as well, you need the `BDD100K` dataset.
You can download the dataset from this link [http://bdd-data.berkeley.edu/download.html](http://bdd-data.berkeley.edu/download.html).
Specifically, we require the "100K Images" variant of the BDD100K dataset.
Download the dataset archives for all splits, and extract them in the `<repo_path>/data/bdd100k/` directory.
This should result in the following directory structure: `<repo_path>/data/bdd100k/bdd100k/images/100k/<train,test,val>`

### Execute Experiments

To reproduce the embeddings, you can use the [scripts/run_embed.sh](scripts/run_embed.sh) script.
Please note that this script depends on the proper setup of the BDD100K dataset.
The embeddings are required for the execution of the experiments.

To run the experiments you can use the [scripts/run_experiments.sh](scripts/run_experiments.sh) script.
This script will run two experiments using the Focus-CoIR dataset: Our method evaluation, as well as the image-only baseline evaluation

All experiment results will be logged using mlflow in the [mlflow](mlflow) directory.
To view these results you can use the [scripts/run_mlflow.sh](scripts/run_mlflow.sh) script.
This will launch a local mlflow server which you can access via local url [http://127.0.0.1:8080](http://127.0.0.1:8080).

## Citation

If you found this repository or the Focus-CoIR dataset useful, please consider citing our work.

<!-- ```bibtex

``` -->
