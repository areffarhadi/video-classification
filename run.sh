#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python3 video_classification_finetune.py

CUDA_VISIBLE_DEVICES=0 python3 video_classification_test.py

