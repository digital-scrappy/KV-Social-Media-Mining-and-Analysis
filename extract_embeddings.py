#!/usr/bin/env python
"""Extract and cache embeddings for train/dev/test (run once, ideally on a GPU).

Usage: python extract_embeddings.py configs/multimodal_gated_lambda0.json
Writes embeddings_cache/<model keys>/{train,dev,test}.npy
"""
import argparse
import json

from main import get_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    get_embeddings(config, splits=("train", "dev", "test"))
