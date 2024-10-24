# Focus-CoIR Dataset

We base our ZIRACLE experiments on our newly created Focus-CoIR dataset, specifically designed for focusing CoIR.
To allow for further research and reproduction of our results, we release the Focus-CoIR dataset licensed in correspondence to the BDD100K dataset license, from which we derived the Focus-CoIR dataset.
The license is provided in the [focus_coir/LICENSE](focus_coir/LICENSE) file.

## Dataset Structure

The Focus-CoIR dataset consists of 102 query images with corresponding focussing query texts.
For each of these query images, 500 binary relevance labels are provided, where the label `1` is given to candidate images that are relevant to the query image and description, and the label `0` is given to irrelevant candidate images.
The combination of query image, texts and corresponding candidate image labels is called "query" in the following.

The queries are provided in the [focus_coir/queries.jsonl](focus_coir/queries.jsonl) file in the jsonlines format.
Each query is given as a json object in a single line.
A query object contains the following fields:

```json
{
    "id": "unique id representing the query.",
    "query_img": "Image filename for the query image.",
    "labels": "List of pairs of image filenames and corresponding labels",
    "name_text": "Textual name of the query (short description)",
    "desc_text": "Textual detailed description of the query (long description)",
}
```

All image filenames correspond to images in the BDD100K dataset.

## Labeling Process

In the following we describe the labeling process for our newly created Focus-CoIR dataset.
We justify the creation of a new dataset over existing CoIR benchmarks via two primary reasons:

- **Focus vs. Transformation Setting**: Existing CoIR benchmarks address the transformation setting, where text descriptions modify image components for retrieval. In contrast, Focus-CoIR aims to evaluate methods that utilize text descriptions to focus on existing image components in order to better align with a user’s search intention.
- **Limited Ground Truth Positives**: Current benchmarks often have few ground truth positive labels per reference image. This limits the robustness of evaluation metrics as only a restricted set of candidate images can contribute positively. To address this, Focus-CoIR provides multiple ground truth positive labels per reference image textual description.

The dataset leverages the publicly available [BDD100K dataset](http://bdd-data.berkeley.edu/) as its foundation.
To generate ground truth labels for image retrieval, we employed a labeling process that utilizes state-of-the-art CoIR methods to minimize manual labeling efforts.
This process is outlined below:

- **Target Image and Text Description Selection**: A random target image is chosen from BDD100K. Two textual descriptions are created: a short description used for CoIR retrieval and a longer description that specifies the manual labeling task.
- **Candidate Retrieval with CoIR Methods**: Three CoIR methods, BLIP4CIR \cite{liu2024bi}, TransAgg \cite{liu2023zeroshot}, and our proposed method, are employed to retrieve the top 500 candidate images for each target image and its corresponding short description. We use three different methods to reduce the bias in the initial retrieval.
- **Candidate List Combination**: A round-robin approach is used to combine and deduplicate the retrieved images from the three methods, resulting in a combined list of 500 unique candidate images.
- **Manual Labeling**: All 500 candidate images are manually labeled as "relevant" or "irrelevant" with respect to the target image and long description.

Following the manual labeling process, a filtering step is implemented to ensure a sufficient number of ground truth positive candidate images per target image.
Target images with less than 30 positive candidates are excluded.
This threshold was chosen to align with the dataset objective outlined earlier.
The filtering process results in a final dataset containing 102 target images.
Each target image has 500 labeled candidate images associated with it.
On average, each target image has 107 positive candidate images, representing approximately  21.4\% of the total candidate images per target.
The maximum number of positive candidate images for a single target image is 271.
