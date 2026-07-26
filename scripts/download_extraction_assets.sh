#!/bin/bash
# Fetch everything needed to regenerate the bottom-up image features.
#
#   bash scripts/download_extraction_assets.sh val          # 316 MB of images
#   bash scripts/download_extraction_assets.sh test         # 1.2 GB
#   bash scripts/download_extraction_assets.sh train        # ~19 GB (COCO)
#   bash scripts/download_extraction_assets.sh detector     # 652 MB
#
# The pre-extracted feature dumps that used to be published are all gone
# (visual-dialog and visdial-bert S3 buckets, up-down-attention GCS). The
# detector that produced them is still up, so the features can be regenerated
# rather than approximated: Faster R-CNN, ResNeXt-101 backbone, fine-tuned on
# Visual Genome, which is what the paper used.
#
# VisDial train images are COCO train2014 + val2014 (123,287 images). The val2018
# and test2018 splits are Flickr images distributed by visualdialog.org.

set -euo pipefail

WHAT="${1:-}"
DATA_DIR="${DATA_DIR:-data}"
IMAGE_DIR="${IMAGE_DIR:-$DATA_DIR/images}"
DETECTOR_DIR="${DETECTOR_DIR:-$DATA_DIR/detector}"

fetch() {  # fetch <url> <destination>
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "already present: $dest"
        return
    fi
    echo "downloading $(basename "$dest")"
    mkdir -p "$(dirname "$dest")"
    curl -fL --retry 3 --continue-at - -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

unzip_once() {  # unzip_once <zip> <marker-dir>
    local zip="$1" marker="$2"
    if [ -d "$marker" ]; then
        echo "already extracted: $marker"
        return
    fi
    unzip -q "$zip" -d "$IMAGE_DIR"
}

case "$WHAT" in
    val)
        fetch "https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=1" \
              "$IMAGE_DIR/VisualDialog_val2018.zip"
        unzip_once "$IMAGE_DIR/VisualDialog_val2018.zip" "$IMAGE_DIR/VisualDialog_val2018"
        ;;
    test)
        fetch "https://www.dropbox.com/s/mwlrg31hx0430mt/VisualDialog_test2018.zip?dl=1" \
              "$IMAGE_DIR/VisualDialog_test2018.zip"
        unzip_once "$IMAGE_DIR/VisualDialog_test2018.zip" "$IMAGE_DIR/VisualDialog_test2018"
        ;;
    train)
        fetch "http://images.cocodataset.org/zips/train2014.zip" "$IMAGE_DIR/train2014.zip"
        unzip_once "$IMAGE_DIR/train2014.zip" "$IMAGE_DIR/train2014"
        fetch "http://images.cocodataset.org/zips/val2014.zip" "$IMAGE_DIR/val2014.zip"
        unzip_once "$IMAGE_DIR/val2014.zip" "$IMAGE_DIR/val2014"
        ;;
    detector)
        fetch "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth" \
              "$DETECTOR_DIR/detectron_model.pth"
        fetch "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml" \
              "$DETECTOR_DIR/detectron_model.yaml"
        ;;
    *)
        echo "usage: $0 {val|test|train|detector}" >&2
        exit 1
        ;;
esac

echo "done: $WHAT"
