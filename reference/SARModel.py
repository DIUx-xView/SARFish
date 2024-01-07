#!/usr/bin/env python3
#
"""SARFish reference model class definition. This file contains functions related 
to SARModel, to be used by 2_train.py, and 3_test.py
"""

from copy import deepcopy
import os
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import torch
import torchvision.models.detection
import torchvision.transforms
import torchvision.models.detection.image_list  

environment_path = Path("environment.yaml")
with open(str(environment_path), "r") as f:
    config = yaml.safe_load(f)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
os.environ['TORCH_USE_CUDA_DSA'] = "1" 
#os.environ['TORCH_HOME'] = config["TORCH_HOME"]

def obtainLabel( typeId ):
    """This function outputs:
        fileList - A list of file name of the tiles, and the associated index 
    of the detection
        allLabelDF - the details of the detection

    typeId=1 for training
    typeId=2 for validation
    typeId=3 for test

    NOTE: This function should also generate a list of file name, that has no 
        detection.

    NOTE: 
    df = pd.read_csv('SLC_train.csv')
    df[ ['is_vessel', 'is_fishing'] ].groupby(['is_vessel','is_fishing'], dropna=False).size()
    is_vessel  is_fishing
    True       False         23948
               True          12534
    NaN        NaN           11092
    False      NaN           16712
    """
    #
    # CONFIG
    SARFish_root_directory = Path(config["SARFish_root_directory"])
    product_type = Path(config["product_type"])
    foldCSV = Path(config["FOLD"]["FoldCSV"])
    whichFold = int(config["FOLD"]["WhichFold"])
    tileSize = int(config["CREATE_TILE"]["TileSize"])
    tileOverlap = int(config["CREATE_TILE"]["TileOverlap"])
    #
    df = pd.read_csv( str(foldCSV) )
    df = df[ df[f"Fold_{whichFold}"] == typeId ]
    sceneList = df['scene_id'].values
    #
    allLabelDF = pd.DataFrame()
    for partition in ['train','validation','public']:
        labelFN = Path(
            SARFish_root_directory, product_type, partition, 
            f"{product_type}_{partition}.csv"
        )
        df = pd.read_csv(str(labelFN))
        df = df[
            ['scene_id', f'{product_type}_product_identifier', 
            'swath_index', 'detect_scene_column', 'detect_scene_row', 
            'is_vessel', 'is_fishing']
        ]
        df = df.dropna( subset=['is_vessel'] )
        df = df[ df['scene_id'].isin( sceneList ) ]
        allLabelDF = pd.concat( [allLabelDF, df] )
    #
    # Derive the tile location (tx,ty) and the relative position in each tile (px,py)
    s = tileSize - tileOverlap 
    allLabelDF['tx'] = ( np.floor( allLabelDF['detect_scene_column'] / s ) ).astype(int)
    allLabelDF['ty'] = ( np.floor( allLabelDF['detect_scene_row'] / s ) ).astype(int)
    allLabelDF['px'] = ( allLabelDF['detect_scene_column'] - s * allLabelDF['tx'] ).astype(int)
    allLabelDF['py'] = ( allLabelDF['detect_scene_row'] - s * allLabelDF['ty'] ).astype(int)
    #
    # vessel_class = 0 (Unknown)
    # vessel_class = 1 (Fishing)
    # vessel_class = 2 (Vessel but not fishing)
    # vessel_class = 3 (Not even a vessel)
    allLabelDF['vessel_class'] = 0
    allLabelDF.loc[ allLabelDF['is_vessel'] & (allLabelDF['is_fishing'] == True), 'vessel_class' ] = 1
    allLabelDF.loc[ allLabelDF['is_vessel'] & (allLabelDF['is_fishing'] == False), 'vessel_class' ] = 2
    allLabelDF.loc[ allLabelDF['is_vessel']== False, 'vessel_class' ] = 3
    #
    # Create a file list, it has the file name, and the index of the labelDF (note: which is sorted)
    #
    allLabelDF = allLabelDF.sort_values( ['scene_id', 'swath_index', 'tx', 'ty'] )
    fileList = []
    prevFN = ''
    listStr = ''
    for k in range(len(allLabelDF)):
        product_identifier = allLabelDF.iloc[k][f'{product_type}_product_identifier']
        swath_index = allLabelDF.iloc[k]['swath_index']
        tx = allLabelDF.iloc[k]['tx']
        ty = allLabelDF.iloc[k]['ty']
        FN = Path(product_identifier, f"swath{swath_index}", f"{tx}_{ty}.npy")
        # if FN is the same as previous, append, else create
        if FN == prevFN:
            # append
            listStr = listStr + ',' + str(k)
        else:
            # save the previous record
            fileList.append( [prevFN, listStr] )
            # create new
            listStr = str(k)
            prevFN = FN
    #
    # remove the first record, which is known to be always ['', '']
    fileList.pop(0)
    # save the last record, which is missed from the loop
    fileList.append( [prevFN, listStr] )
    # 
    return fileList, allLabelDF


def convertVesselClassToValues( vessel_class ):
    """Given Vessel class:
        vessel_class = 0 (Unknown)
        vessel_class = 1 (Fishing)
        vessel_class = 2 (Vessel but not fishing)
        vessel_class = 3 (Not even a vessel)

    Return (is_vessel, is_fishing)
    """
    if vessel_class == 0:
        return (np.NaN, np.NaN)
    if vessel_class == 1:
        return (True, True)
    if vessel_class == 2:
        return (True, False)
    if vessel_class == 3:
        return (False, np.NaN)

#
# Dataset
#
class SarfishDataset(object):
    #
    def __init__( self, fileList, labelDF):
        self.fileList = fileList
        self.labelDF = labelDF
        self.tilePath = config["CREATE_TILE"]["TilePath"]
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")
        #
        # myTransform expect input of numpy (H x W x C), and outputs tensor (C x H x W)
        self.myTransform = torchvision.transforms.transforms.Compose(
            [torchvision.transforms.transforms.ToTensor(),
            torchvision.transforms.transforms.Normalize(
                mean = [0, 0, 0], std = [1, 1, 1] 
            )]
        )
        #
    #
    def __len__(self):
        return len(self.fileList)
    #
    def __getitem__(self, idx):
        halfBox = 32
        (imgFN, indexList) = self.fileList[idx]
        # get the image filename
        imgFN = Path(self.tilePath, imgFN)
        img = np.load(str(imgFN))        # 4 x H x W
        # TODO:  SOMEHOW only 3 channels are allowed
        img = img[0:3,:,:]
        imgH = img.shape[1]
        imgW = img.shape[2]
        #
        indexList = indexList.split(',')
        point_label = []
        class_label = []
        if len(indexList) > 0:
            # get the values from self.labelDF
            for indexStr in indexList:
                k = int(indexStr)
                px = self.labelDF.iloc[k]['px']
                py = self.labelDF.iloc[k]['py']
                x1 = px - halfBox
                y1 = py - halfBox
                x2 = px + halfBox
                y2 = py + halfBox
                # bound: TODO: This should be done in 1_create_tile.
                if (x1 < 0):
                    x1 = 0
                if (y1 < 0):
                    y1 = 0
                if (x2 >= imgW):
                    x2 = imgW-1
                if (y2 >= imgH):
                    y2 = imgH-1
                point_label.append( [x1, y1, x2, y2] )
                class_label.append( self.labelDF.iloc[k]['vessel_class'] )
        # convert the output to a particular format
        targets = {}
        targets['boxes'] = torch.tensor(point_label).to(self.device)
        targets['labels'] = torch.tensor(class_label).to(self.device)
        #
        # first convert from 3 x H x W to H x W x 3, which then convert back to 3 x H x W 
        print(f'img.shape: {img.shape}')
        img = self.myTransform( img.transpose([1,2,0]) )
        print(f'img.shape: {img.shape}')
        img = img.to(self.device)
        return img, targets


class SARFishModel(torch.nn.Module):
    #
    def __init__(self, num_classes):
        super(SARFishModel, self).__init__()
        #
        # Use pretrained model
        self.actualModel = torchvision.models.detection.fcos_resnet50_fpn( 
            weights = torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT, 
            weights_backbone = torchvision.models.resnet.ResNet50_Weights.DEFAULT,
        )
        oldWeight = self.actualModel.head.classification_head.cls_logits.weight
        self.actualModel.head.classification_head.cls_logits.out_channels = num_classes
        self.actualModel.head = torchvision.models.detection.fcos.FCOSHead(
            in_channels=256, num_anchors=1, num_classes=num_classes
        ) 
        newWeight = oldWeight[0:num_classes,:,:,:]
        self.actualModel.head.classification_head.cls_logits.weight = (
            torch.nn.Parameter( newWeight )
        )
    #
    #
    def forward(self, *input, **kwargs):
        out = self.actualModel.forward(*input, **kwargs)
        return out
