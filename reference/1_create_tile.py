#!/usr/bin/env python3

"""This code combines VH,VV complex SLC images into tiles images for the purpose of
   training/testing the detection/classification algorithm.

After running this script, it should create the following file structure:

[TilePath]/[scene_id]/swath1/0_0.npy    | This is the chip data (complex numpy array), 
                                        | the filename is tx_ty.npy, where tx is the tile index in x axis,
                                        | and ty is the tile index in y axis
                                        | Each .npy has 4 channels, vh_mag, vh_phase, vv_mag, vv_phase

[TilePath]/img_file_info.csv            | This file saved the file info of the scenes used
"""
from pathlib import Path
import yaml

import numpy as np
import pandas as pd

import GeoTiff

def combineVHVV(
        vhFN: Path, vvFN: Path, tileSize: int, tileOverlap: int
    ) -> np.ndarray:
    """INPUT: the file name of VH and VV

    1 - Pad an image to make it divisible by some block_size with along with overlap
        Pad on the right and bottom edges so annotations are still usable.
    2 - Calculate the absolute and the phase, of the VH and VV, and place them into 
        one 4 channel numpy array

    OUTPUT: numpy array of 4 x H x W
    """
    # Read VH image, NOTE: img is a masked array
    (vhImg, _, _, _) = GeoTiff.load_GeoTiff( str(vhFN) )
    imgH, imgW = vhImg.shape
    s = tileSize - tileOverlap
    newH = int( np.ceil( (imgH-tileOverlap) / s ) * s ) + tileOverlap
    newW = int( np.ceil( (imgW-tileOverlap) / s ) * s ) + tileOverlap
    padH = newH - imgH
    padW = newW - imgW
    #
    # This np.pad function also converts the masked array into a complex array.  A masked value becomes 0.
    padImg = np.pad(
        vhImg, pad_width=((0, padH), (0, padW)), mode="constant", 
        constant_values=0
    )
    vhAbs = np.abs( padImg )
    vhPhase = np.angle( padImg )
    #
    # do the same for VV image
    ( vvImg, _, _, _) = GeoTiff.load_GeoTiff( str(vvFN) )
    padImg = np.pad(
        vvImg, pad_width=((0, padH), (0, padW)), mode="constant", 
        constant_values=0
    )
    vvAbs = np.abs( padImg )
    vvPhase = np.angle( padImg )
    #
    # Create a 4 channel image of vhAbs, vhPhase, vvAbs, vvPhase
    out = np.stack( [vhAbs, vhPhase, vvAbs, vvPhase], axis=0 ) 
    return out

def chopAndSaveTiles(
        combined: np.ndarray, outDir: Path , tileSize: int, tileOverlap: int
    ):
    """Chop the combined image (4 x H x W) into tiles
    """
    s = tileSize - tileOverlap
    imgH = combined.shape[1]
    imgW = combined.shape[2]
    numTileX = int( imgW / s )
    numTileY = int( imgH / s )
    if not outDir.exists():
        outDir.mkdir(parents = True, exist_ok = True)

    for tx in range( numTileX ):
        for ty in range( numTileY ):
            x1 = tx * s
            y1 = ty * s
            x2 = x1 + tileSize
            y2 = y1 + tileSize
            FN = Path(outDir, f"{tx}_{ty}.npy")
            with open(str(FN), "wb") as f:
                np.save( f, combined[ :, y1:y2, x1:x2 ] )

def main():
    #=========================
    # Config
    #=========================

    environment_path = Path("environment.yaml")
    with open(str(environment_path), "r") as f:
        config = yaml.safe_load(f)

    foldCSV = Path(config["FOLD"]["FoldCSV"])
    foldDf = pd.read_csv( foldCSV )
    sceneList = foldDf['scene_id'].values
    swathList = [1, 2, 3]

    # the xView3_SLC_GRD_correspondences DataFrame is the mapping that 
    # allows you to iterate over the dataset and build paths
    xView3_SLC_GRD_correspondences_path = Path(
        config["xView3_SLC_GRD_correspondences_path"]
    )
    xView3_SLC_GRD_correspondences = pd.read_csv(
        str(xView3_SLC_GRD_correspondences_path)
    )
    xView3_SLC_GRD_correspondences = xView3_SLC_GRD_correspondences[
        xView3_SLC_GRD_correspondences["scene_id"].isin(sceneList)
    ]

    SARFish_root_directory = Path(config["SARFish_root_directory"])
    product_type = Path(config["product_type"])
    tileSize = int(config["CREATE_TILE"]["TileSize"])
    tileOverlap = int(config["CREATE_TILE"]["TileOverlap"])
    tilePath = Path(config["CREATE_TILE"]["TilePath"])
    if not tilePath.exists():
        tilePath.mkdir(parents = True, exist_ok = True)

    for index, row in xView3_SLC_GRD_correspondences.iterrows():
        for swath_index in [1, 2, 3]:
            print(f"tiling {row[f'{product_type}_product_identifier']}, swath {swath_index}")
            outDir = Path(tilePath, row[f"{product_type}_product_identifier"], f"swath{swath_index}")
            if outDir.exists():
                continue

            measurement_directory = Path(
                SARFish_root_directory, product_type, row["DATA_PARTITION"], 
                f"{row[f'{product_type}_product_identifier']}.SAFE", "measurement", 
            )
            vh_FN = Path(
                measurement_directory, row[f"SLC_swath_{swath_index}_vh"]
            )
            vv_FN = Path(
                measurement_directory, row[f"SLC_swath_{swath_index}_vv"]
            )
                
            combined = combineVHVV(vh_FN, vv_FN, tileSize, tileOverlap)
            chopAndSaveTiles(combined, outDir, tileSize, tileOverlap)

if __name__ == "__main__":
    main()
