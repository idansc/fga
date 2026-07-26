---
license: cc-by-4.0
task_categories:
  - visual-question-answering
language:
  - en
tags:
  - visual-dialog
  - visdial
  - multimodal
  - factor-graph-attention
pretty_name: VisDial v1.0 preprocessed for Factor Graph Attention
size_categories:
  - 1M<n<10M
---

# VisDial v1.0, preprocessed for Factor Graph Attention

The preprocessed VisDial v1.0 files used by [Factor Graph Attention](https://arxiv.org/abs/1904.05880)
(CVPR'19) — code at [idansc/fga](https://github.com/idansc/fga).

Evaluation is done on [VisDialv1.0](https://visualdialog.org/data).

Short description:

VisDial v1.0 contains 1 dialog with 10 question-answer pairs (starting from an image caption) on ~130k images
from COCO-trainval and Flickr, totalling ~1.3 million question-answer pairs.

These are the tokenized, integer-indexed versions of those dialogs: every question,
answer, caption and candidate answer is stored as vocabulary ids, which is the form
the model consumes. They are published here because the links previously given in the
repository have expired.

## Files

| File | Contents |
| --- | --- |
| `visdial_data.h5` | Tokenized dialogs, answer options and captions for train/val/test |
| `visdial_params.json` | Vocabulary (`word2ind`, `ind2word`) and the image filename list per split |
| `visdial_1.0_val_dense_annotations.json` | Dense relevance judgements for the val split, used for NDCG |

### `visdial_data.h5` layout

Per split (`train` / `val` / `test`):

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `ques_{split}` | `(num_images, 10, 20)` | Question token ids, zero-padded after the last word |
| `ques_length_{split}` | `(num_images, 10)` | Words per question |
| `ans_{split}` | `(num_images, 10, 20)` | Ground-truth answer token ids |
| `ans_length_{split}` | `(num_images, 10)` | Words per answer |
| `opt_{split}` | `(num_images, 10, 100)` | 1-based indices into `opt_list_{split}` |
| `opt_list_{split}` | `(num_answers, 20)` | The pool of unique candidate answers |
| `opt_length_{split}` | `(num_answers,)` | Words per candidate answer |
| `ans_index_{split}` | `(num_images, 10)` | 1-based index of the correct option (train/val only) |
| `cap_{split}` | `(num_images, 40)` | Caption token ids |
| `cap_length_{split}` | `(num_images,)` | Words per caption |
| `num_rounds_{split}` | `(num_images,)` | Rounds actually present; the test split is cut at a random round |
| `img_pos_{split}` | `(num_images,)` | Image position index |

Splits are 123,287 / 2,064 / 8,000 images. The vocabulary holds 11,319 words; token
id 0 is padding, and the loader appends a `<stop>` and an `<empty>` symbol, so an
embedding table needs 11,322 rows.

Image filenames live in `visdial_params.json` under `unique_img_{split}`, e.g.
`VisualDialog_val2018/VisualDialog_val2018_000000185565.jpg`.

## Usage

```python
from huggingface_hub import snapshot_download
from fga import VisDialDataset
from fga.data import load_visdial_params, vocab_size_from_params

path = snapshot_download("Idan/visdial-fga-preprocessed", repo_type="dataset")
params = load_visdial_params(f"{path}/visdial_params.json")

dataset = VisDialDataset(
    visdial_data_path=f"{path}/visdial_data.h5",
    image_features_path="frcnn_features_new.h5",   # see below
    split="val",
    vocab_size=vocab_size_from_params(params),
)
```

## What is not here

**Image features.** The model reads pre-extracted region features, not pixels — an h5
with `{split}_features` of shape `(num_images, 37, 2048)`. Those are not included here.

Pretrained features:
- VGG: a grid image feature based on the VGG model pretrained on ImageNet *(Faster)*.
  Note, the h5 databases has slightly different dataset keys, therefore the code needs
  to be adapted accordingly.
- F-RCNN: based on object detector with ResNeXt-101 backbone, 37 proposals, fine-tuned
  on [Visual Genome](https://visualgenome.org/). *Achieves SOTA*. The file includes boxes
  and classes information.

See the original paper for performance differences. I recommend using the FRCNN features,
mainly because it is finetuned on the relevant VisualGenome dataset.

The images themselves can be obtained from [visualdialog.org/data](https://visualdialog.org/data)
or from the [`HuggingFaceM4/VisDial`](https://huggingface.co/datasets/HuggingFaceM4/VisDial)
mirror. Note that the Hub mirrors of VisDial carry only `[question, answer]` pairs — they
do not include the answer options or the ground-truth index, so they cannot be used to
train or evaluate a ranking model on their own.

## Licensing and credit

Released under CC BY 4.0, matching the upstream VisDial v1.0 annotations. This is a
preprocessed derivative; the underlying dialog data is by Das et al., and the images come
from COCO and Flickr under their own terms.

```
@inproceedings{das2017visual,
  title={Visual Dialog},
  author={Das, Abhishek and Kottur, Satwik and Gupta, Khushi and Singh, Avi and Yadav, Deshraj
          and Moura, Jos\'e M.F. and Parikh, Devi and Batra, Dhruv},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

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
