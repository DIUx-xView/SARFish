#!/usr/bin/env python3

import os
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import torch

from GeoTiff import load_GeoTiff
import SARModel
import torchvision

def inference():
    """Run the trained model over full-size SLC products from the public partition
    of the SARFish dataset.
    """

    environment_path = Path("environment.yaml")
    with open(str(environment_path), "r") as f:
        config = yaml.safe_load(f)

    SARFish_root_directory = Path(config["SARFish_root_directory"])
    partition = config["TEST"]["partition"]
    product_type = Path(config["product_type"])
    tileSize = int(config["CREATE_TILE"]["TileSize"])
    tileOverlap = int(config["CREATE_TILE"]["TileOverlap"])
    trainedModelPath = config["TRAIN"]["TrainedModelPath"]
    scoreThreshold = float(config["TEST"]["ScoreThreshold"])
    epochToUse = config["TEST"]["EpochToUse"]
    inference_model_path = Path(trainedModelPath, f"epoch_{epochToUse}.pth")
    reference_model_predictions_path = Path(
        config["TEST"]["referencePredictionsPath"]
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = SARModel.SARFishModel(num_classes = 4)
    model.load_state_dict(
        torch.load(str(inference_model_path), map_location = device)
    )
    model.to(device)
    model.eval()
    torch.no_grad()

    myTransform = torchvision.transforms.transforms.Compose(
        [torchvision.transforms.transforms.ToTensor(),
        torchvision.transforms.transforms.Normalize(
            mean = [0, 0, 0], std = [1, 1, 1] 
        )]
    )

    scene_prediction_dataframes = []
    xView3_SLC_GRD_correspondences = pd.read_csv(
        config["xView3_SLC_GRD_correspondences_path"]
    )
    xView3_SLC_GRD_correspondences_public = xView3_SLC_GRD_correspondences[
        xView3_SLC_GRD_correspondences['DATA_PARTITION'] == partition
    ]
    for index, correspondence in xView3_SLC_GRD_correspondences_public.iterrows():
        SLC_product_identifier = correspondence[
            f'{product_type}_product_identifier'
        ] 
        swath_prediction_dataframes = []
        for swath_index in [1, 2, 3]:
            print(f"processing {SLC_product_identifier}, swath {swath_index}")
            measurement_directory = Path(
                SARFish_root_directory, product_type, 
                correspondence["DATA_PARTITION"], f"{SLC_product_identifier}.SAFE", 
                "measurement", 
            )
            vh_FN = Path(
                measurement_directory, correspondence[f"SLC_swath_{swath_index}_vh"]
            )
            vv_FN = Path(
                measurement_directory, correspondence[f"SLC_swath_{swath_index}_vv"]
            )
            vh_data, vh_mask, _, _ = load_GeoTiff(str(vh_FN))
            vh_ysize, vh_xsize = vh_data.shape
            vh_amplitude = np.abs(vh_data)
            vh_phase = np.angle(vh_data)
            vh_data = None

            vv_data, vv_mask, _, _ = load_GeoTiff(str(vv_FN))
            vv_amplitude = np.abs(vv_data)
            vv_phase = np.angle(vv_data)
            vv_data = None
            
            image = np.concatenate(
                [vh_amplitude[..., None], vh_phase[..., None], 
                vv_amplitude[..., None], vv_phase[..., None]], axis = -1
            )
            tile_predictions_dataframes = []
            xmax = 1 + (vh_xsize - tileSize) // (tileSize - tileOverlap)
            ymax = 1 + (vh_ysize - tileSize) // (tileSize - tileOverlap)
            for x in range(0, xmax):
                xmin = x * (tileSize - tileOverlap) if x < xmax-1 else vh_xsize - tileSize
                for y in range(0, ymax):
                    ymin = y * (tileSize - tileOverlap) if y < ymax-1 else vh_ysize - tileSize
                    print(f'processing tile (y, x): {y, x}')
                    #print(f'region bounds: {ymin}:{ymin + tileSize}, {xmin}: {xmin + tileSize}')
                    image_region = image[ymin: ymin + tileSize, xmin: xmin + tileSize, :3]
                    image_region = myTransform(image_region)[None, ...]
                    image_region = image_region.to(device)
                    if torch.all(image_region == 0):
                        continue

                    out = model(image_region)
                    image_region = None

                    boxesTensor = out[0]['boxes']
                    numDetect = len(boxesTensor)
                    if numDetect == 0:
                        continue
                    #
                    # format the output to Numpy
                    boxesNumpy = boxesTensor.cpu().detach().numpy()
                    scoresNumpy = out[0]['scores'].cpu().detach().numpy()
                    labelsNumpy = out[0]['labels'].cpu().detach().numpy()
                    #
                    # Only output scores larger than threshold. (Is this score always sorted?)
                    #isGoodScore = scoresNumpy > scoreThreshold
                    isGoodScore = scoresNumpy > 0.5
                    scoresNumpy = scoresNumpy[isGoodScore]
                    labelsNumpy = labelsNumpy[isGoodScore]
                    boxesNumpy = boxesNumpy[isGoodScore, :]
                    numDetect = len(scoresNumpy)
                    #
                    
                    prediction_dicts = []
                    for bId in range(numDetect):
                        prediction_row = {}
                        # parse the test file information, take one label
                        prediction_row['scene_id'] = correspondence['scene_id']
                        prediction_row[f'{product_type}_product_identifier'] = SLC_product_identifier
                        prediction_row['swath_index'] = swath_index
                        prediction_row['detect_scene_column'] = (
                            np.mean(boxesNumpy[bId,[0,2]]) + 
                            x * (tileSize - tileOverlap)
                        )
                        prediction_row['detect_scene_row'] = (
                            np.mean(boxesNumpy[bId,[1,3]]) + 
                            y * (tileSize - tileOverlap)
                        )
                        prediction_row['score'] = scoresNumpy[bId]
                        (is_vessel, is_fishing) = (
                            SARModel.convertVesselClassToValues(labelsNumpy[bId])
                        )
                        prediction_row['is_vessel'] = is_vessel
                        prediction_row['is_fishing'] = is_fishing
                        #
                        prediction_dicts.append(prediction_row)

                    tile_predictions = pd.DataFrame.from_records(
                        prediction_dicts
                    )
                    tile_predictions_dataframes.append(tile_predictions)

            swath_predictions = pd.concat(tile_predictions_dataframes)
            swath_predictions = swath_predictions.groupby(
                ['detect_scene_column', 'detect_scene_row']
            ).first().reset_index()
            swath_predictions['vessel_length_m'] = 12.3
            swath_predictions['partition'] = partition
            swath_predictions['product_type'] = product_type
            swath_predictions = swath_predictions[
                ['partition', 'scene_id', 'product_type', 
                'SLC_product_identifier', 'swath_index', 'detect_scene_column', 
                'detect_scene_row', 'scene_id', 'is_vessel', 'is_fishing', 
                'vessel_length_m']
            ]
            swath_prediction_dataframes.append(swath_predictions)

        scene_predictions = pd.concat(swath_prediction_dataframes)
        scene_predictions_path = Path(
            reference_model_predictions_path, 
            f"reference_predictions_{product_type}_{partition}_{SLC_product_identifier}.csv"
        )
        scene_predictions.to_csv(str(scene_predictions_path))
        scene_predictions.to_csv(str(scene_predictions_path))

def main():
    inference()

if __name__ == "__main__":
    main()
