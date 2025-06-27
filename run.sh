#!/bin/bash
python main.py viz_model_preds mini --modelf=/workspace/lss1/lss/model525000.pt --dataroot=/workspace/lss1/dateset/nuscenes --map_folder=/workspace/lss1/dateset/nuscenes
python main.py eval_model_iou mini --modelf=/workspace/lss1/lss/model525000.pt --dataroot=/workspace/lss1/dateset/nuscenes