# Factor Graph Attention

This repository is the official implementation of Factor Graph Attention (https://arxiv.org/abs/1904.05880).
(Appeared in CVPR'19)


<img src="imgs/fga.png">
<img src="imgs/model.png">

## Requirements

The model can easily run on a single GPU :)

To install requirements:

```setup
conda env create -f fga.yml
```

following with:
```
 conda env create -f fga.yml
 conda activate fga
```

## Preprocessed data:

Add the following files under data dir:

[visdial_params.json](https://technionmail-my.sharepoint.com/:u:/g/personal/idansc_campus_technion_ac_il/EUbM8rne2_BItNT8fR6MJXsBnzAMzUB7ssEc4Xt1Wza0lA?e=xCrKx8)

[visdial_data.h5](https://technionmail-my.sharepoint.com/:u:/g/personal/idansc_campus_technion_ac_il/EczwqWX5lBVAizIfpLauq1kB-s8oaZpF93drq5y8TOiRjQ?e=8NqsoY)

Pretrained features:
Soon uploading
- [VGG]() A grid image features based on the VGG model pretrained on ImageNet.
- [F-RCNN]() based on object detector, 37 proposals, based on ResNetx101, fine-tuned on VisualGnome.


## Training

To train the model in the paper, run this command:

```train
python train.py --batch-size  128 \
             --image_data "data/frcnn_features_new" \
             --test-batch-size 64 \
             --epochs 10 \
             --lr 1e-3 \
             --opt 0 \
             --folder-prefix "baseline" \
             --mode "FGA" \
             --initialization "he" \
             --lstm-initialization "he" \
             --log-interval 3000 \
             --test-after-every 1 \
             --word-embed-dim 256 \
             --hidden-img-dim -1 \
             --hidden-ans-dim 512 \
             --hidden-hist-dim 128 \
             --hidden-cap-dim 128 \
             --hidden-ques-dim 512 \
             --seed 0
```

## Evaluation

To evaluate on the val split, provide a path using the model-pathname arg.
The path should contain a model file, `best_model_mrr.pth.tar`.

Call example:
```val eval
python train.py --batch-size  128 \
             --image_data "data/frcnn_features_new" \
             --test-batch-size 64 \
             --epochs 10 \
             --lr 1e-3 \
             --opt 0 \
             --only_val T \
             --model-pathname "models/baseline"
             --folder-prefix "baseline" \
             --mode "FGA" \
             --initialization "he" \
             --lstm-initialization "he" \
             --log-interval 3000 \
             --test-after-every 1 \
             --word-embed-dim 256 \
             --hidden-img-dim -1 \
             --hidden-ans-dim 512 \
             --hidden-hist-dim 128 \
             --hidden-cap-dim 128 \
             --hidden-ques-dim 512 \
             --seed 0
```

If you wish to create a test submission file (can be submitted to the challenge servers @ EvalAI)
replace only_val, with submission arg, i.e.:

```test eval
python train.py --batch-size  128 \
             --image_data "data/frcnn_features_new" \
             --test-batch-size 64 \
             --epochs 10 \
             --lr 1e-3 \
             --opt 0 \
             --submission T \
             --model-pathname "models/baseline"
             --folder-prefix "baseline" \
             --mode "FGA" \
             --initialization "he" \
             --lstm-initialization "he" \
             --log-interval 3000 \
             --test-after-every 1 \
             --word-embed-dim 256 \
             --hidden-img-dim -1 \
             --hidden-ans-dim 512 \
             --hidden-hist-dim 128 \
             --hidden-cap-dim 128 \
             --hidden-ques-dim 512 \
             --seed 0
```

## Pre-trained Models

You can download pertained models here:

- [uploading]() trained on VisDial1.0 using F-RCNN features


## Results

Evaluation is done on ### [VisDialv1.0](https://visualdialog.org/data).

Short description:

VisDial v1.0 contains 1 dialog with 10 question-answer pairs (starting from an image caption) on ~130k images
from COCO-trainval and Flickr, totalling ~1.3 million question-answer pairs. The v1.0 training set consists
of dialogs on ~120k images from COCO-trainval, while the validation and test sets consist of dialogs on an additional
~10k COCO-like images from Flickr. We have worked closely with the COCO team to ensure that these additional images
match the distribution of images and captions of the training set.



Our model achieves the following performance on the validation set, and similar results on test-std/test-challenge.

| Model name                   | R@1 |  MRR  |
| ---------------------------- |---- |------ |
| FGA                          | 53% |  66   |
| FGA                          | 53% |  66   |
| 5×FGA                        | 56% |  69   |

Note, the paper results may slightly vary from the results of this repo, since it is a refactored version.
For the legacy version, please contact via email

## Contributing

Please cite Factor Graph Attention if you use this work in your research:

@inproceedings{schwartz2019factor,
  title={Factor graph attention},
  author={Schwartz, Idan and Yu, Seunghak and Hazan, Tamir and Schwing, Alexander G},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2039--2048},
  year={2019}
}
