#!/usr/bin/env python3

"""SARFish reference testing script, This file applies the trained model 
to the extracted tiles (for testing), and generates the detected locations 
and classification.
"""

import os
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import torchvision.models.detection
import torch

import SARModel

environment_path = Path("environment.yaml")
with open(str(environment_path), "r") as f:
    config = yaml.safe_load(f)

#os.environ['TORCH_HOME'] = config["TORCH_HOME"]

def main():
    #=========================
    # Config
    #=========================

    # FOLD
    foldCSV = config["FOLD"]["FoldCSV"]
    whichFold = config["FOLD"]["WhichFold"]

    # CREATE_TILE
    tileSize = int(config["CREATE_TILE"]["TileSize"])
    tileOverlap = int(config["CREATE_TILE"]["TileOverlap"])

    # TRAIN
    trainedModelPath = config["TRAIN"]["TrainedModelPath"]
    xView3_SLC_GRD_correspondences_path = Path(
        config["xView3_SLC_GRD_correspondences_path"]
    )

    # TES
    epochToUse = config["TEST"]["EpochToUse"]
    scoreThreshold = float(config["TEST"]["ScoreThreshold"])

    #
    # Pass the config across to the SARModel
    SARModel.config = config

    #=========================
    # MAIN
    #=========================

    (testFileList, testLabelDF) = SARModel.obtainLabel( 3 )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load model
    modelFN = Path(trainedModelPath, f"epoch_{epochToUse}.pth")
    model = SARModel.SARFishModel( num_classes = 4 )
    model.load_state_dict(torch.load(str(modelFN), map_location = device))
    model.to(device)
    model.eval()

    #
    test_data = SARModel.SarfishDataset(testFileList, testLabelDF)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size = 1, sampler = test_sampler, num_workers = 0
    )

    #
    allOut = []
    useCache = False
    cacheFN = '3_test_cache.csv'
    if useCache:
        cacheDF = pd.read_csv(cacheFN)
    else:
        torch.no_grad()
        for testId, (image, targets) in enumerate(test_data_loader):
            #print(f'image: {image}')
            print(f'shape: {image.shape}')
            print(f'type: {type(image)}')
            out = model(image)
            #
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
            isGoodScore = scoresNumpy > scoreThreshold
            scoresNumpy = scoresNumpy[isGoodScore]
            labelsNumpy = labelsNumpy[isGoodScore]
            boxesNumpy = boxesNumpy[isGoodScore, :]
            numDetect = len(scoresNumpy)
            #
            for bId in range( numDetect ):
                print( "Test Id: ", testId, " Batch Id: ", bId )
                oneRow = {}
                # parse the test file information, take one label
                oneLabelIndex = int( testFileList[testId][1].split(',')[0] )
                oneLabel = testLabelDF.iloc[oneLabelIndex]
                oneRow['scene_id'] = oneLabel['scene_id']
                oneRow['swath_index'] = oneLabel['swath_index']
                tx = int( oneLabel['tx'] )
                ty = int( oneLabel['ty'] )
                oneRow['detect_scene_column'] = np.mean( boxesNumpy[bId,[0,2]] ) + tx * (tileSize - tileOverlap)
                oneRow['detect_scene_row'] = np.mean( boxesNumpy[bId,[1,3]] ) + ty * (tileSize - tileOverlap)
                oneRow['score'] = scoresNumpy[bId]
                (is_vessel, is_fishing) = SARModel.convertVesselClassToValues( labelsNumpy[bId] )
                oneRow['is_vessel'] = is_vessel
                oneRow['is_fishing'] = is_fishing
                #
                allOut.append( oneRow )
        #
        cacheDF = pd.DataFrame( allOut )
        cacheDF.to_csv(cacheFN, index = False)

    #
    # Fill in SLC_product_identifier for evaluation
    xView3_SLC_GRD_DF = pd.read_csv(str(xView3_SLC_GRD_correspondences_path))
    xView3_SLC_GRD_DF = xView3_SLC_GRD_DF[['scene_id', 'SLC_product_identifier']]
    outDF = cacheDF.merge(xView3_SLC_GRD_DF)
    print(f'outDF: {outDF}')
    print(f'outDF.columns: {outDF.columns}')

    #
    # Fill in other fields
    outDF['vessel_length_m'] = 12.3
    outDF['partition'] = 'validation'
    outDF['product_type'] = 'SLC'
    outDF.to_csv('3_test_output.csv', index = False)

if __name__ == "__main__":
    main()
