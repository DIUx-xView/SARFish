#!/usr/bin/env python3

import os
from pathlib import Path
import yaml

import pandas as pd

def concatenate_scene_predictions():
    """Concatenate the scene predictions from 5_reference.py into a merged .csv
    file for evaluation using the SARFish_metric.py script.
    """

    environment_path = Path("environment.yaml")
    with open(str(environment_path), "r") as f:
        config = yaml.safe_load(f)

    partition = config["TEST"]["partition"]
    product_type = Path(config["product_type"])
    scene_prediction_dataframes = []
    reference_model_predictions_path = Path(
        config["TEST"]["referencePredictionsPath"]
    )

    xView3_SLC_GRD_correspondences = pd.read_csv(
        config["xView3_SLC_GRD_correspondences_path"]
    )
    xView3_SLC_GRD_correspondences= xView3_SLC_GRD_correspondences[
        xView3_SLC_GRD_correspondences['DATA_PARTITION'] == partition
    ]
    scene_predictions_dataframes = []
    for index, correspondence in xView3_SLC_GRD_correspondences.iterrows():
        SLC_product_identifier = correspondence[
            f'{product_type}_product_identifier'
        ] 
        print(f"processing: {SLC_product_identifier}")
        scene_predictions_path = Path(
            reference_model_predictions_path, 
            f"reference_predictions_{product_type}_{partition}_{SLC_product_identifier}.csv"
        )
        scene_predictions = pd.read_csv(str(scene_predictions_path))
        scene_prediction_dataframes.append(scene_predictions)

    reference_predictions = pd.concat(scene_prediction_dataframes)
    print(f'submission .csv format:\n{reference_predictions.dtypes.to_markdown()}')
    reference_predictions_path = Path(
        reference_model_predictions_path, 
        f"reference_predictions_{product_type}_{partition}.csv"
    )
    reference_predictions.to_csv(str(reference_predictions_path), index = False) 

def main():
    concatenate_scene_predictions()

if __name__ == "__main__":
    main()
