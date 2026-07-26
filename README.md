# Factor Graph Attention

* A general multimodal attention approach inspired by probabilistic graphical models.
* Achieves a state-of-the-art performance (MRR) on visual dialog task.

This repository is the official implementation of [Factor Graph Attention](https://arxiv.org/abs/1904.05880).
(Appeared in CVPR'19)

<p float="left">
  <img src="imgs/fga.png" height="40%" width="40%"/>
  <img src="/imgs/model.png"" height="50%" width="50%"/>
</p>

* Part of 2020 visual dialog challenge winning submission (https://github.com/idansc/mrr-ndcg)

Use cases of FGA:
* Video dialog, spatial interactions between frames, can be found here (https://github.com/idansc/simple-avsd)
* Spatial navigation,  can be found here (https://github.com/barmayo/spatial_attention)
* Video retrieval, between text query and clips, can be found here (https://github.com/AmeenAli/VideoMatch)

## Installation

```bash
pip install -e ".[train]"
```

The model runs on a single GPU, and also on CPU or Apple Silicon (MPS) — the
device is chosen for you.

## Quick start

FGA follows the standard `transformers` interfaces, so it composes with the rest
of the ecosystem:

```python
from fga import FGAConfig, FGAForVisualDialog

model = FGAForVisualDialog(FGAConfig(hidden_img_dim=2048))
outputs = model(**batch, labels=labels)   # outputs.loss, outputs.logits
ranking = outputs.logits.argsort(dim=-1, descending=True)

model.save_pretrained("models/fga")       # config.json + model.safetensors
model = FGAForVisualDialog.from_pretrained("models/fga")
```

`AutoConfig` / `AutoModel` resolve `model_type="fga"` once `fga` has been
imported, and `push_to_hub` / `from_pretrained("<user>/fga")` work as usual.

## The attention block on its own

The factor-graph attention is the reusable part of the paper and knows nothing
about Visual Dialog — hand it any list of `(batch, num_entities, dim)` utilities:

```python
from fga.attention import Atten

attention = Atten(util_e=[512, 2048], sizes=[20, 36])
pooled_text, pooled_image = attention([text_states, image_regions])
```

Describe the graph by name rather than by parallel index-aligned lists. These two
are equivalent, but only one is readable:

```python
# indexed: "utility 4 repeats 9 times and connects to utilities 0 and 1"
Atten(util_e=[512, 512, 128], sizes=[100, 21, 21], sharing_factor_weights={2: (9, [0, 1])})

# named
from fga.attention import Atten, Utility

attention = Atten.from_utilities([
    Utility("answer",   dim=512, size=100),
    Utility("question", dim=512, size=21),
    Utility("history",  dim=128, size=21, repeats=9, connected_to=("answer", "question")),
], use_prior=True)

print(attention.describe())
```

`repeats` is what the paper calls factor-weight sharing: the utility arrives as
`(batch, repeats, entities, dim)` and one set of factor weights serves all
repeats, which is how nine history rounds stay affordable. `connected_to` is the
efficiency constraint — a shared utility only interacts with the utilities it
names. Both are validated, so a typo now raises instead of silently building a
different graph.

`FGAConfig` takes the same readable form:

```python
FGAConfig(shared_utilities=[
    {"name": "history_question", "repeats": 9, "connected_to": ["answer", "question"]},
    {"name": "history_answer",   "repeats": 9, "connected_to": ["answer", "question"]},
])
```

The indexed `sharing_factor_weights` stays the serialized field, so old configs
and checkpoints keep loading, and `config.shared_utilities` renders it by name.
The `prior_flag` / `pairwise_flag` / `unary_flag` / `self_flag` arguments are
still accepted alongside the clearer `use_prior` / `use_pairwise` / `use_unary` /
`use_self`.

To get the attention distributions for a visualization, pass
`output_attentions=True` (or `return_weights=True` to `Atten`).

## Data

Add the following files under `data/`:

| File | What it is |
| --- | --- |
| `visdial_data.h5` | Tokenized dialogs, options and captions for train/val/test |
| `visdial_params.json` | Vocabulary (`word2ind`) and per-split image filenames |
| `visdial_1.0_val_dense_annotations.json` | Dense relevance judgements, required for NDCG — [visualdialog.org/data](https://visualdialog.org/data) |
| `frcnn_features_new.h5` | Image features: `{split}_features` of shape `(num_images, 37, 2048)` |

> **Note:** the SharePoint links previously listed here have expired. The dialog
> files can be rebuilt from the official VisDial v1.0 JSONs; the image features
> must be re-extracted (see below).

### On the HuggingFace Hub copies

`HuggingFaceM4/VisDial` and `jxu124/visdial` mirror VisDial v1.0 with the right
split sizes (123,287 / 2,064 / 8,000), but their schema is
`caption`, `dialog` as `[[question, answer], ...]`, `image_path`, `image` —
**no `answer_options`, no `gt_index`, no dense relevance**. FGA *ranks* 100
candidate answers, so those columns are exactly the ones it needs; the Hub copies
cannot replace `visdial_data.h5`. Dense annotations are only distributed by
[visualdialog.org/data](https://visualdialog.org/data).

They are, however, a convenient source of the **images** themselves
(`HuggingFaceM4/VisDial` ships them inline, ~21.8 GB), which is what you need to
re-extract the F-RCNN features.

Pretrained image features:
- **F-RCNN** — object detector with a ResNeXt-101 backbone, 37 proposals, fine-tuned on
  [Visual Genome](https://visualgenome.org/). *Achieves SOTA.* Recommended, mainly
  because it is finetuned on the relevant Visual Genome data.
- **VGG** — a grid image feature based on the VGG model pretrained on ImageNet
  (*faster*). Note the h5 uses slightly different dataset keys, so the loader
  needs adapting.

If you have the bottom-up-attention LMDB published with the
[VisDial challenge starter code](https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch),
convert it into the expected layout:

```python
from fga.image_features import lmdb_to_h5
from fga.data import image_ids_from_params, load_visdial_params

params = load_visdial_params("data/visdial_params.json")
lmdb_to_h5("features_val.lmdb", "data/frcnn_features_new.h5", "val", image_ids_from_params(params, "val"))
```

To exercise the pipeline before the real features are in place:

```bash
python scripts/make_dev_image_features.py --splits val --num_images 64 --output data/dev_features.h5
```

## Training

```bash
python scripts/run_visual_dialog.py \
    --image_features_path data/frcnn_features_new.h5 \
    --output_dir models/baseline \
    --do_train --do_eval \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 10 \
    --learning_rate 1e-3 \
    --word_embed_dim 200 \
    --hidden_ques_dim 512 --hidden_ans_dim 512 \
    --hidden_hist_dim 128 --hidden_cap_dim 128 \
    --eval_strategy epoch --save_strategy epoch \
    --load_best_model_at_end --metric_for_best_model mrr \
    --bf16 --seed 0
```

Training runs on [`transformers.Trainer`], so mixed precision, gradient
accumulation, checkpoint resumption, early stopping and W&B/TensorBoard logging
are all available as standard flags. For multi-GPU, launch with
`torchrun --nproc_per_node=N` — the script needs no changes.

A ready-made SLURM job is in [`slurm/train_fga.sbatch`](slurm/train_fga.sbatch).

## Evaluation

Evaluate on the val split, reporting R@{1,5,10}, mean rank, MRR and NDCG:

```bash
python scripts/run_visual_dialog.py \
    --image_features_path data/frcnn_features_new.h5 \
    --model_name_or_path models/baseline --output_dir models/baseline \
    --do_eval --write_submission
```

Produce a test submission for the [EvalAI](https://evalai.cloudcv.org/) challenge server:

```bash
python scripts/run_visual_dialog.py \
    --image_features_path data/frcnn_features_new.h5 \
    --model_name_or_path models/baseline --output_dir models/baseline \
    --do_predict --write_submission
```

## Pre-trained models

Original `.pth.tar` checkpoints convert to the HuggingFace format with:

```bash
python scripts/convert_legacy_checkpoint.py \
    --checkpoint models/baseline/best_model_mrr.pth.tar \
    --output_dir models/fga-frcnn-hf \
    --image_feature_dim 2048
```

The converter strips `DataParallel` prefixes, renames the parameters to the
current module layout, and writes `config.json` + `model.safetensors`. Pass
`--push_to_hub <user>/fga` to publish. `tests/test_equivalence.py` checks that a
converted checkpoint reproduces the original model's scores exactly.

## Results

Evaluation is done on [VisDialv1.0](https://visualdialog.org/data).

VisDial v1.0 contains 1 dialog with 10 question-answer pairs (starting from an image caption) on ~130k images
from COCO-trainval and Flickr, totalling ~1.3 million question-answer pairs.

Our model achieves the following performance on the validation set, and similar results on test-std/test-challenge.

| Model name | R@1 | MRR |
| --- | --- | --- |
| FGA | 53% | 66 |
| 5×FGA | 56% | 69 |

### Reproduced with this code

Trained from scratch with the commands above — 10 epochs on 8×L40S, about 3 hours —
using image features re-extracted with the same Visual-Genome-finetuned detector,
since every published copy of the originals is now offline.

| Epoch | R@1 | R@5 | R@10 | MRR | Mean rank | NDCG |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 44.09 | 73.67 | 83.39 | 57.85 | 5.95 | 49.16 |
| 3 | 50.66 | 81.36 | 89.75 | 64.48 | 4.18 | 55.28 |
| **5** | **52.46** | **82.95** | **90.97** | **66.01** | **3.92** | 56.46 |
| 7 | 52.23 | 83.05 | 90.70 | 65.79 | 3.98 | 57.44 |
| 10 | 51.23 | 81.48 | 89.81 | 64.68 | 4.31 | **58.19** |

MRR peaks at epoch 5 and matches the published 66; R@1 comes in 0.54 lower. NDCG keeps
climbing after MRR has turned over, which is the metric tension the
[2020 challenge submission](https://github.com/idansc/mrr-ndcg) dealt with — checkpoints
are saved every epoch so you can select per metric.

Weights for the epoch-5 checkpoint: [idansc/fga](https://huggingface.co/idansc/fga).

Note, the paper results may slightly vary from the results of this repo, since it is a refactored version.
For the legacy version, please contact via email.

## Notes on this refactor

The math is unchanged and covered by an equivalence test against the original
code, which is preserved verbatim in `tests/legacy_reference/`. Three fixes do
change behaviour:

* **Unary dropout at evaluation.** The original called `F.dropout(...)` without
  forwarding `self.training`, so activations were dropped during evaluation and
  scores were non-deterministic. Set `legacy_unary_dropout=True` on the config to
  reproduce the old numbers exactly.
* **Padding embedding.** The padding row was randomly initialized and, because
  `padding_idx` zeroes its gradient, stayed random. It is now zero.
* **Option lengths without `--astop`.** That branch indexed the answer-length
  table by round instead of by option, producing a scalar where 100 lengths were
  needed. The default path (`astop=True`) was unaffected.

Six `margin_Y` convolutions in the self-interaction factors were computed and
discarded on every forward pass; they are gone, and the converter drops the
corresponding untrained weights.

Run the tests with `pytest`. The data tests skip automatically when
`data/visdial_data.h5` is absent.

## Contributing

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
