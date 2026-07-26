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
  - ndcg
datasets:
  - Idan/visdial-fga-preprocessed
metrics:
  - recall
  - mrr
  - ndcg
model-index:
  - name: fga-ndcg
    results:
      - task:
          type: visual-question-answering
          name: Visual Dialog
        dataset:
          type: visdial
          name: VisDial v1.0 val
        metrics:
          - type: ndcg
            value: 69.07
            name: NDCG
          - type: mrr
            value: 49.03
            name: MRR
          - type: recall
            value: 34.27
            name: R@1
          - type: recall
            value: 66.15
            name: R@5
          - type: recall
            value: 80.13
            name: R@10
---

# Factor Graph Attention — NDCG variant

* A general multimodal attention approach inspired by probabilistic graphical models.
* Achieves a state-of-the-art performance (MRR) on visual dialog task.

This is [Factor Graph Attention](https://arxiv.org/abs/1904.05880) (CVPR'19) finetuned
on the **dense relevance annotations**, which optimizes NDCG. Code:
[idansc/fga](https://github.com/idansc/fga).

For the MRR-oriented model, see [`Idan/fga`](https://huggingface.co/Idan/fga).

## Why there are two models

The sparse training label calls one of the 100 candidates correct and the other 99
equally wrong. That is what MRR measures. But for "is it daytime?" the candidate list
contains *yes*, *yeah*, *it is* and *yes it is*, and the label picks one arbitrarily.
NDCG instead scores against the graded relevance five annotators assigned to every
candidate.

Training on the graded signal moves NDCG a long way and costs MRR, because the two
metrics genuinely disagree about what a good ranking is. That tension is the subject of
the 2020 challenge submission, https://github.com/idansc/mrr-ndcg.

## Results

On VisDial v1.0 val. The baseline is the MRR model this was finetuned from:

| Model | NDCG | MRR | R@1 | R@5 | R@10 | Mean rank |
| --- | --- | --- | --- | --- | --- | --- |
| [`Idan/fga`](https://huggingface.co/Idan/fga) (sparse) | 56.46 | **66.01** | 52.46 | 82.95 | 90.97 | 3.92 |
| **This model** (dense only) | **69.07** | 49.03 | 34.27 | 66.15 | 80.13 | 6.68 |
| dense + sparse, weight 0.5 | 68.00 | 61.93 | 49.55 | 76.59 | 86.14 | 5.07 |
| dense + sparse, weight 1.0 | 66.61 | 63.27 | 50.69 | 78.33 | 87.38 | 4.78 |

This checkpoint is the pure-dense end of that trade-off: the best NDCG, and the worst
MRR. If you want one model that is respectable at both, the `sparse_weight=1.0` row
gives up 2.5 NDCG to recover 14 MRR, and is reproducible with `--sparse_weight 1.0`.

Trained for 5 epochs at lr 1e-4 on the 2,000-image dense subset of **train**. The val
annotations are never trained on — they are the evaluation set.

### On the objective

The loss is a soft-label cross entropy: the relevance vector is normalized into a
distribution and matched against the model's log-softmax over the candidates,

```
loss = -sum_i p_i log q_i,     p = relevance / sum(relevance)
```

which is the cross entropy the sparse loss already computes, generalized from a one-hot
target to a graded one.

A smooth approximation of NDCG itself (ApproxNDCG, Qin et al.) was also tried and is
implemented in the repo. It is worse on this task — +6.0 NDCG against +12.6, while
losing more MRR — most likely because it concentrates its gradient on the few positions
the discount rewards, whereas the soft cross entropy draws signal from all 100
candidates, which matters when there are only 2,000 training examples.

## Usage

```python
from fga import FGAForVisualDialog

model = FGAForVisualDialog.from_pretrained("Idan/fga-ndcg")
outputs = model(**batch)
ranking = outputs.logits.argsort(dim=-1, descending=True)
```

Install with `pip install git+https://github.com/idansc/fga.git`.

Reproduce with:

```bash
python scripts/finetune_dense.py \
    --model_name_or_path <sparse checkpoint> \
    --train_dense_path data/visdial_1.0_train_dense_annotations.json \
    --output_dir models/fga-ndcg \
    --loss soft_ce --sparse_weight 0.0 \
    --learning_rate 1e-4 --num_train_epochs 5
```

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
