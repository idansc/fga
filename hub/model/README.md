---
license: mit
library_name: transformers
pipeline_tag: visual-question-answering
tags:
  - visual-dialog
  - visdial
  - multimodal
  - attention
  - factor-graph-attention
datasets:
  - idansc/visdial-fga-preprocessed
---

# Factor Graph Attention

* A general multimodal attention approach inspired by probabilistic graphical models.
* Achieves a state-of-the-art performance (MRR) on visual dialog task.

This is the official implementation of [Factor Graph Attention](https://arxiv.org/abs/1904.05880).
(Appeared in CVPR'19). Code: [idansc/fga](https://github.com/idansc/fga).

* Part of 2020 visual dialog challenge winning submission (https://github.com/idansc/mrr-ndcg)

Use cases of FGA:
* Video dialog, spatial interactions between frames, can be found here (https://github.com/idansc/simple-avsd)
* Spatial navigation, can be found here (https://github.com/barmayo/spatial_attention)
* Video retrieval, between text query and clips, can be found here (https://github.com/AmeenAli/VideoMatch)

> ### ⚠️ Architecture only — no trained weights yet
>
> This repository currently holds the **configuration and model definition**, so the
> architecture can be instantiated and inspected. It does **not** contain trained
> parameters. Anything you load from here is randomly initialized and will produce
> chance-level rankings.
>
> The original checkpoints were hosted on an institutional account that has since
> expired. Trained weights will be published here once they are recovered or retrained,
> together with re-measured metrics.

## How it works

Every modality is a *utility*: a set of entities with an embedding each — the 100
candidate answers, the question words, the caption words, the image regions, and the
question and answer of each history round. Attention over a utility is the softmax of a
sum of learned potentials, exactly as in a factor graph:

* **unary** — how salient an entity is on its own,
* **self** — how an entity relates to the other entities of the same utility,
* **pairwise** — how an entity relates to the entities of every other utility,
* **prior** — an external bias, such as sentence-length cues.

The potentials are stacked and combined by a learned, bias-free `Conv1d`, so the model
learns how much to weight each factor. The nine history rounds share one set of factor
weights, which is what keeps the interaction affordable.

The model can easily run on a single GPU :) — and also on CPU or Apple Silicon.

## Usage

```python
from fga import FGAConfig, FGAForVisualDialog

model = FGAForVisualDialog(FGAConfig(hidden_img_dim=2048))

outputs = model(
    question_input_ids=...,          # (batch, 21)
    option_input_ids=...,            # (batch, 100, 21)
    history_question_input_ids=...,  # (batch, 9, 21)
    history_answer_input_ids=...,    # (batch, 9, 21)
    caption_input_ids=...,           # (batch, 41)
    question_lengths=...,
    option_lengths=...,
    caption_lengths=...,
    image_features=...,              # (batch, 37, 2048)
    labels=...,                      # optional, gives outputs.loss
)
ranking = outputs.logits.argsort(dim=-1, descending=True)
```

Install with `pip install git+https://github.com/idansc/fga.git`.

The attention block is the reusable part of the paper and knows nothing about Visual
Dialog — give it any set of utilities:

```python
from fga.attention import Atten, Utility

attention = Atten.from_utilities([
    Utility("text",    dim=512,  size=20),
    Utility("image",   dim=2048, size=36),
    Utility("history", dim=128,  size=21, repeats=9, connected_to=("text", "image")),
], use_prior=True)
```

Pass `output_attentions=True` to get the per-utility attention distributions for
visualization.

## Data

Preprocessed dialogs: [`idansc/visdial-fga-preprocessed`](https://huggingface.co/datasets/idansc/visdial-fga-preprocessed).

Image features are not distributed — the model expects an h5 with `{split}_features` of
shape `(num_images, 37, 2048)`. See the original paper for performance differences.
I recommend using the FRCNN features, mainly because it is finetuned on the relevant
VisualGenome dataset.

## Results reported in the paper

Evaluation is done on [VisDialv1.0](https://visualdialog.org/data).

VisDial v1.0 contains 1 dialog with 10 question-answer pairs (starting from an image
caption) on ~130k images from COCO-trainval and Flickr, totalling ~1.3 million
question-answer pairs.

| Model name | R@1 | MRR |
| --- | --- | --- |
| FGA | 53% | 66 |
| 5×FGA | 56% | 69 |

These are the published numbers on the validation set, with similar results on
test-std/test-challenge. **They describe the original trained model, not this
repository**, which ships no weights. Note also that the paper results may slightly vary
from the results of the current code, since it is a refactored version.

## Citation

Please cite Factor Graph Attention if you use this work in your research:
```
@inproceedings{schwartz2019factor,
  title={Factor graph attention},
  author={Schwartz, Idan and Yu, Seunghak and Hazan, Tamir and Schwing, Alexander G},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2039--2048},
  year={2019}
}
```
