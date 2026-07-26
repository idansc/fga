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
  - Idan/visdial-fga-preprocessed
metrics:
  - recall
  - mrr
  - ndcg
model-index:
  - name: fga
    results:
      - task:
          type: visual-question-answering
          name: Visual Dialog
        dataset:
          type: visdial
          name: VisDial v1.0 val
        metrics:
          - type: mrr
            value: 66.01
            name: MRR
          - type: recall
            value: 52.46
            name: R@1
          - type: recall
            value: 82.95
            name: R@5
          - type: recall
            value: 90.97
            name: R@10
          - type: ndcg
            value: 56.46
            name: NDCG
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

The model can easily run on a single GPU :) — 52.4M parameters.

## Results

Evaluation is done on [VisDialv1.0](https://visualdialog.org/data).

VisDial v1.0 contains 1 dialog with 10 question-answer pairs (starting from an image
caption) on ~130k images from COCO-trainval and Flickr, totalling ~1.3 million
question-answer pairs.

These are the measured numbers for the weights in this repository, on the v1.0
validation split:

| Metric | R@1 | R@5 | R@10 | MRR | Mean rank | NDCG |
| --- | --- | --- | --- | --- | --- | --- |
| **This checkpoint** (epoch 5) | 52.46 | 82.95 | 90.97 | **66.01** | 3.92 | 56.46 |
| Paper | 53 | — | — | 66 | — | — |
| 5×FGA ensemble (paper) | 56 | — | — | 69 | — | — |

Trained for 10 epochs on 8×L40S, ~3 hours. MRR peaks at epoch 5 while NDCG keeps
improving to epoch 10 (58.19) — the two metrics disagree, which is the tension the
[2020 challenge submission](https://github.com/idansc/mrr-ndcg) addressed.

Two things differ from the original run and are folded into the small R@1 gap. The
image features were re-extracted, because every published copy of the originals has
gone offline; the detector is the same one the paper used (Faster R-CNN, ResNeXt-101,
fine-tuned on Visual Genome, 36 proposals). And evaluation no longer applies dropout:
the original called `F.dropout` without forwarding `self.training`, so scoring was
mildly stochastic even under `model.eval()`.

## Usage

```python
from fga import FGAForVisualDialog

model = FGAForVisualDialog.from_pretrained("Idan/fga")

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
)
ranking = outputs.logits.argsort(dim=-1, descending=True)
```

Install with `pip install git+https://github.com/idansc/fga.git`.

Pass `output_attentions=True` to get the per-utility attention distributions for
visualization.

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

## Data

Preprocessed dialogs: [`Idan/visdial-fga-preprocessed`](https://huggingface.co/datasets/Idan/visdial-fga-preprocessed).

Image features are not distributed with the model — it expects an h5 with
`{split}_features` of shape `(num_images, 37, 2048)`. See the original paper for
performance differences. I recommend using the FRCNN features, mainly because it is
finetuned on the relevant VisualGenome dataset. The repository has scripts to
regenerate them from the images.

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
