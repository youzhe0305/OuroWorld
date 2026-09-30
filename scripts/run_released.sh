#!/usr/bin/env bash
# Rerun one released scene of the paper from its static 3DGS to its loop videos.
# The paper's motion prompt and reference video are reused, so no API key is needed.
#
#   bash scripts/run_released.sh <dataset> <scene>     e.g. Marble world
#
# Outputs go to output/<dataset>/<scene>/; the loop videos to render/iter_XXXXX/.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <dataset> <scene>   (the scenes are under data/3dgs/)" >&2
    exit 2
fi
DATASET=$1
SCENE=$2
OUT=output/$DATASET/$SCENE

ouroworld-import released --source "data/3dgs/$DATASET/$SCENE" --out "$OUT/scene"
mkdir -p "$OUT/reference_video"
cp "data/refvideo/$DATASET/$SCENE/prompt_used.txt" "$OUT/reference_video/prompt.txt"
cp "data/refvideo/$DATASET/$SCENE/raw.mp4" "$OUT/reference_video/raw.mp4"

for stage in reference-video multiview train render; do
    "ouroworld-$stage" "dataset=$DATASET" "scene=$SCENE"
done
