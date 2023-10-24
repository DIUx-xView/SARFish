#!/usr/bin/env python3

"""A simple example script that processes all of the SARFish images in a
directory, saves the detections to a CSV file, and then evaluates the
results using the provided metric script.

! ./SARFish_reference.py 
"""

import argparse
from pathlib import Path
import os
import random
import sys
import yaml

import numpy as np
import pandas as pd
import torch

from GeoTiff import load_GeoTiff
from SARFish_metric import score
from scipy import signal as sig
from visualise_labels import scale_sentinel_1_image

random.seed(12345)

#---------------------------------------------------
# Examples of image processing scripts. Replace these with your own code.

def process_image(vh, vv, output_dets, num = 100):

    """A very simple detector that returns the num "brightest" blobs in
    the vh channel only.

    Calculate the background mean and variance in an annulus around
    the central pixel. Only calculate a fraction of the pixels in
    the annulus to improve speed.
    """

    A = 12   # Diameter of annulus will be 2*A - 1

    targ_mean = vh**2

    cnt = scnt = 0
    bkg_mean = torch.zeros_like(vh)
    bkg_var = torch.zeros_like(vh)

    for i in range(2*A-1):
        for j in range(2*A-1):
            if (i==0 or i==2*A-2) or (j==0 or j==2*A-2):
                if cnt % 3 == 0:  # Only evaluate every third pixel.
                    bkg_mean[i+1:,j+1:] += vh[:-i-1,:-j-1]
                    bkg_var[i+1:,j+1:] += targ_mean[:-i-1,:-j-1]
                    scnt += 1
                cnt += 1
    bkg_mean = bkg_mean / scnt
    bkg_var = bkg_var / scnt - bkg_mean ** 2

    # Calculate the possible target mean over a 3x3 pixel region.

    targ_mean[:] = 0
    for i in range(3):
        for j in range(3):
            targ_mean[i+1:,j+1:] += vh[:-i-1,:-j-1]
    targ_mean = targ_mean / 9

    # Final score is scaled by the background variance.
    
    score = torch.zeros_like(vh)
    score[:-A,:-A] = (targ_mean[2:-A+2,2:-A+2] - bkg_mean[A:,A:]) **2
    score[:-A,:-A] = score[:-A,:-A] / (bkg_var[A:,A:] + 1)
    score = score.numpy()

    del bkg_mean
    del bkg_var

    # Get the 20 * num set of pixels with the highest scores. Some of these
    # will be associated with the same object.

    coords = np.argpartition(score.ravel(), - 20*num)[-20*num:]
    coords = [(x % score.shape[1], x // score.shape[1]) for x in coords]
    coords = [(score[x[1],x[0]], x[0], x[1]) for x in coords]
    coords.sort(reverse = True)
    
    # Append all of the detections to the output data structure.

    cnt = 0
    while 1:
        if not(coords): break  # Run out of detections.
        det = coords.pop(0)

        # Check whether this is close to an existing detection.

        if score[det[2],det[1]] == 0: continue

        # Zero-out neighbouring scores so these are not output.

        score[max(det[2]-15,0):min(det[2]+15,score.shape[0]),
              max(det[1]-15,0):min(det[1]+15,score.shape[1])] = 0

        output_dets['detect_scene_column'].append(det[1])
        output_dets['detect_scene_row'].append(det[2])

        # Guess some arbitrary values for the length and ship class.

        output_dets['vessel_length_m'].append(100)
        output_dets['is_vessel'].append(random.random() < 0.6)
        output_dets['is_fishing'].append(random.random() < 0.3)

        cnt += 1
        if cnt == num: break


# Read in the complex valued images for each polarization, convert
# them to a real image, and then process them.

def process_image_SLC(im_name, vh_name, vv_name, output_dets):

    vh, _, _, _ = load_GeoTiff(str(Path(im_name, vh_name)))
    vv, _, _, _ = load_GeoTiff(str(Path(im_name, vv_name)))

    # Converts to real data. Change this to make use of
    # complex phase information.

    vh = np.abs(vh)
    vh = vh.astype(np.float16)
    vh[np.where(np.isinf(vh))] = 0

    vv = np.abs(vv)
    vv = vv.astype(np.float16)
    vh[np.where(np.isinf(vh))] = 0

    # Convert to torch for speed.

    vh = torch.from_numpy(vh)
    vv = torch.from_numpy(vv)
    process_image(vh, vv, output_dets, num = 200)


# Read in the real valued images for each polarization and
# process them.

def process_image_GRD(im_name, vh_name, vv_name, output_dets):

    vh, _, _, _ = load_GeoTiff(str(Path(im_name, vh_name)))
    vh = scale_sentinel_1_image(vh, product_type = "GRD")
    vh = vh.astype(np.float32)
    vh[np.where(np.isinf(vh))] = 0

    vv, _, _, _ = load_GeoTiff(str(Path(im_name, vv_name)))
    vv = scale_sentinel_1_image(vv, product_type = "GRD")
    vv = vv.astype(np.float32)
    vv[np.where(np.isinf(vh))] = 0

    # Torch is typically faster than numpy, so convert.

    vh = torch.from_numpy(vh)
    vv = torch.from_numpy(vv)
    process_image(vh, vv, output_dets, num = 300)


def SARFish_reference(
        SARFish_root_directory: Path, xv3c: pd.DataFrame, 
        product_type: {"GRD", "SLC"}, partition: {"train", "validation", "public"},
        predictions_output: Path
    ):
    img_path = Path(SARFish_root_directory, product_type, partition)

    # 2. SETUP THE DETECTION STRUCTURE.

    output_dets = {'partition': [], 'product_type': [], 'scene_id': [],
                   'detect_scene_column': [], 'detect_scene_row': [],
                   'vessel_length_m': [], 'is_vessel': [], 'is_fishing': []
                  }
    if product_type == 'GRD':
        output_dets['GRD_product_identifier'] = []
    elif product_type == 'SLC':
        output_dets['SLC_product_identifier'] = []
        output_dets['swath_index'] = []
    else:
        raise ValueError('Bad value for variable "product_type"')


    # 3. PROCESS ALL OF THE IMAGES

    xv3c = xv3c[xv3c['DATA_PARTITION'] == partition]

    print('\n')
    for i, xv3c_ in xv3c.iterrows():
        im = xv3c_[f"{product_type}_product_identifier"]
        im_name = Path(
            SARFish_root_directory, product_type, partition, 
            f'{im}.SAFE', 'measurement', 
        )

        nswaths = 1
        if product_type == 'GRD':
            im_info = xv3c_
            sceneID = im_info.at['scene_id']
        else:
            im_info = xv3c_
            sceneID = im_info.at['scene_id']
            nswaths = 3

        for k in range(1,nswaths+1):
            print(f'Processing {im}')
            if product_type == 'SLC':
                print(f'swath {k}')

            if product_type == 'GRD':
                vh_name = im_info.at['GRD_vh']
                vv_name = im_info.at['GRD_vv']
                process_image_GRD(im_name, vh_name, vv_name, output_dets)
            else:
                vh_name = im_info.at['SLC_swath_' + str(k) + '_vh']
                vv_name = im_info.at['SLC_swath_' + str(k) + '_vv']
                process_image_SLC(im_name, vh_name, vv_name, output_dets)

            # Fill in the missing detection fields.

            ndets = len(output_dets['is_vessel']) - len(output_dets['scene_id'])
            
            output_dets['partition'].extend([partition] * ndets)
            output_dets['product_type'].extend([product_type] * ndets)
            output_dets['scene_id'].extend([sceneID] * ndets)

            if product_type == 'GRD':
                output_dets['GRD_product_identifier'].extend([im] * ndets)
            else:
                output_dets['SLC_product_identifier'].extend([im] * ndets)
                output_dets['swath_index'].extend([k] * ndets)

        print('Stopped after processing only one image!')
        break  # Processing the images takes a long time. Just do one!

    import pickle
    with open('temp.pkl','wb') as fid:
        pickle.dump(output_dets, fid)

    # 4. SAVE THE RESULTS TO FILE

    print('Detections found:', len(output_dets['scene_id']))

    output_dets = pd.DataFrame(output_dets)
    output_dets.to_csv(str(predictions_output))


    # 5. EVALUATE THE RESULTS

    if partition == 'public':
        print('Do not have ground-truth for evaluation of "public" data')
        return

    # Load the ground-truth data to be used for evaluation.

    gtName = f'{product_type}_{partition}.csv'
    gt = pd.read_csv(str(Path(img_path, gtName)))

    # Call the metrics script to measure the performance.
    # score_all will be True for public evaluation.

    sc = score(
        predictions = output_dets, groundtruth = gt, 
        xView3_SLC_GRD_correspondences = xv3c, 
        SARFish_root_directory = SARFish_root_directory, 
        product_type = product_type, shoreline_type = 'xView3_shoreline', 
        score_all = False, drop_low_detect = True, costly_dist = True,  
        evaluation_mode = False #just for the single scene example
    )

def main():
    # 1. CONFIGURATION.
    environment_path = Path("environment.yaml")
    with open(str(environment_path), "r") as f:
        config = yaml.safe_load(f)

    SARFish_root_directory = Path(config["SARFish_root_directory"])
    xView3_SLC_GRD_correspondences_path = Path(
        config["xView3_SLC_GRD_correspondences_path"]
    )
    xView3_SLC_GRD_correspondences = pd.read_csv(
        str(xView3_SLC_GRD_correspondences_path)
    )
    SARFish_reference(
        SARFish_root_directory, xView3_SLC_GRD_correspondences, "GRD", 
        "validation", Path('reference_GRD_predictions.csv')
    )
    SARFish_reference(
        SARFish_root_directory, xView3_SLC_GRD_correspondences, "SLC", 
        "validation", Path('reference_SLC_predictions.csv')
    )

if __name__ == "__main__":
    main()
