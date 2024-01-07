#!/usr/bin/env python3

"""SARFish reference model evaluation script using the SARFish_metric.py
"""

from pathlib import Path
import yaml

import pandas as pd

import SARFish_metric

def main():
    environment_path = Path("environment.yaml")
    with open(str(environment_path), "r") as f:
        config = yaml.safe_load(f)

    SARFish_root_directory = config["SARFish_root_directory"]
    product_type = Path(config["product_type"])
    partition = 'validation'
    xView3_SLC_GRD_correspondences_path = Path(
        config["xView3_SLC_GRD_correspondences_path"]
    )
    xView3_SLC_GRD_correspondences = pd.read_csv(
        str(xView3_SLC_GRD_correspondences_path)
    )

    predictions = pd.read_csv("3_test_output.csv")
    groundtruth_path = Path(
        SARFish_root_directory, product_type, partition, 
        f"{product_type}_{partition}.csv"
    )
    groundtruth = pd.read_csv(str(groundtruth_path))

    SARFish_metric.score(
        predictions, groundtruth, xView3_SLC_GRD_correspondences, 
        SARFish_root_directory, product_type = "SLC", shoreline_type = None, 
        score_all = False, drop_low_detect = True, costly_dist = True, 
        evaluation_mode = False
    )

if __name__ == "__main__":
    main()
