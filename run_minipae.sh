#!/bin/bash
target_dir="/Users/jbard/Library/CloudStorage/SynologyDrive-Shared/Data/KexinP/250925_STRINGDB_testset/colabfold_batch/positive/batch_predictions"
pixi run python AF2_MiniPAE_JB_v2.py --unzipped-dir "$target_dir" \
  --rolling --window-size 5 --window-avg 7 -o positive_minipae.csv