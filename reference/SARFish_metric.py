#!/usr/bin/env python3

"""
! SARFISH_ROOT_DIRECTORY=; ./SARFish_metric.py -p ./labels/reference_GRD_predictions.csv -g "${SARFISH_ROOT_DIRECTORY}"/GRD/validation/GRD_validation.csv --sarfish_root_directory "${SARFISH_ROOT_DIRECTORY}" --product_type GRD --xview3_slc_grd_correspondences ./labels/xView3_SLC_GRD_correspondences.csv --shore_type xView3_shoreline --drop_low_detect --costly_dist --no-evaluation_mode
"""

import argparse
from functools import reduce
import itertools
import json
from pathlib import Path
from typing import Tuple, List, Union, Dict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance_matrix
from tqdm.auto import tqdm

from SAR import (get_spacing_and_timing_from_annotation,
    get_srgrConvParams_from_GRD_annotation,
    get_linear_interpolator_of_srgrConvParams,
    convert_SLC_image_coordinates_to_meters)

CBYELLOW = "\33[1;93m"
CBLUE = "\33[1;94m"
CRED = "\33[1;31m"
CPURPLE = "\33[1;35m"
CGREEN = "\33[1;32m"
CCYAN = "\33[1;36m"
CEND = "\33[0m"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def drop_low_confidence_preds(
        SARFish_root_directory: Path, predictions: pd.DataFrame, 
        groundtruth: pd.DataFrame, 
        xView3_SLC_GRD_correspondences: pd.DataFrame, product_type: {"GRD", "SLC"},
        assignment_tolerance_meters: float, costly_dist: bool = False
    ) -> pd.DataFrame:
    """Uses the hungarian matching algorithm to the find the lowest distance cost 
    assignment of predictions location labels to the "LOW" confidence 
    groundtruth location labels and remove them from further consideration in 
    the metrics. 
    """
    scene_ids_to_evaluate = reduce(
        np.intersect1d,
        [groundtruth['scene_id'].unique(), predictions['scene_id'].unique(), 
        xView3_SLC_GRD_correspondences['scene_id']]
    )
    if scene_ids_to_evaluate.size == 0:
        raise ValueError(
            "The intersection of scene_id(s) in the predictions and "
            "groundtruth pandas.DataFrames is empty"
        )
    else:
        low_confidence_indices = {
            'predictions': [], 'groundtruth': []
        }

    is_scene_ids_to_evaluate = (
        xView3_SLC_GRD_correspondences['scene_id'].isin(
            scene_ids_to_evaluate
        )
    )
    xView3_SLC_GRD_correspondences = xView3_SLC_GRD_correspondences[
        is_scene_ids_to_evaluate
    ]
    print(f"dropping predictions corresponding to low confidence groundtruth...")
    for index, xView3_SLC_GRD_correspondence in tqdm(xView3_SLC_GRD_correspondences.iterrows()):
        scene_predictions = predictions[
            predictions['scene_id'] == 
            xView3_SLC_GRD_correspondence['scene_id']
        ]
        scene_groundtruth = groundtruth[
            groundtruth['scene_id'] == 
            xView3_SLC_GRD_correspondence['scene_id']
        ]
        evaluation_columns = ['detect_scene_row', 'detect_scene_column']
        if product_type == "SLC":
           evaluation_columns.append('swath_index') 

        location_true_positive_indices_, _, _ = compute_loc_performance(
            SARFish_root_directory,
            scene_predictions[evaluation_columns],
            scene_groundtruth[evaluation_columns],
            xView3_SLC_GRD_correspondence, product_type,
            assignment_tolerance_meters,
            costly_dist
        )
        low_confidence_indices_ = mask_packaged_indices(
            location_true_positive_indices_,
            (scene_groundtruth.loc[location_true_positive_indices_['groundtruth']]['confidence'] == "LOW")
        )
        low_confidence_indices['predictions'].append(
            low_confidence_indices_['predictions']
        )
    
    print("")
    low_confidence_indices = {
        'predictions': np.concatenate(
            low_confidence_indices['predictions'], axis = 0
        )
    }
    predictions_high_medium_confidence = predictions.drop(
        index = low_confidence_indices['predictions']
    )
    return predictions_high_medium_confidence

def mask_close_to_shore_predictions(
        shoreline_meters: np.ndarray, predictions_meters: np.ndarray, 
        distance_from_shore_tolerance_meters: float, 
        far_from_shore_value_meters = 9999990.0
    ) -> np.ndarray:
    """Returns a mask indication which of the predictions is within 
    distance_from_shore_tolerance_meters of the shoreline vector.
    """
    tree1 = cKDTree(shoreline_meters)
    tree2 = cKDTree(predictions_meters)
    sdm = tree1.sparse_distance_matrix(
        tree2, distance_from_shore_tolerance_meters, p=2
    )
    pairwise_distances = sdm.toarray()
    pairwise_distances[pairwise_distances == 0] = far_from_shore_value_meters
    closest_pairwise_distances_meters = np.nanmin(pairwise_distances, axis=0)
    is_close_to_shore = (
        closest_pairwise_distances_meters != far_from_shore_value_meters
    )
    return is_close_to_shore

def get_shore_preds(
        SARFish_root_directory: str, 
        xView3_SLC_GRD_correspondence: pd.DataFrame, 
        predictions: pd.DataFrame, product_type: {"GRD", "SLC"},
        shoreline_type: {"xView3_shoreline", "global_shoreline_vector"} = "xView3_shoreline",
        distance_from_shore_tolerance_meters: float = 2000.0,
        far_from_shore_value_meters = 9999990.0
    ) -> pd.DataFrame:
    """Returns indices of the predictions within
    distance_from_shore_tolerance_meters of the shoreline_type. 
    """
    if len(predictions) == 0:
        return np.array([])

    partition = xView3_SLC_GRD_correspondence['DATA_PARTITION']
    product_title = xView3_SLC_GRD_correspondence['scene_id']
    GRD_vh_annotation_path = (
        lambda x: str(
            Path(
                SARFish_root_directory, "GRD", x['DATA_PARTITION'], 
                f"{x['GRD_product_identifier']}.SAFE",
                "annotation", x['GRD_vh_annotation']
            )
        )
    )(xView3_SLC_GRD_correspondence)
    if product_type == "GRD":
        (azimuthLineSpacingMeters, rangePixelSpacingMeters, _, _, _) = (
            get_spacing_and_timing_from_annotation(
                GRD_vh_annotation_path           
            )
        )
        predictions_meters = np.array(
            [predictions['detect_scene_row'] * azimuthLineSpacingMeters,
            predictions['detect_scene_column'] * rangePixelSpacingMeters]
        ).transpose()
        shoreline_path = (
            lambda x: str(
                Path(
                    SARFish_root_directory, "GRD", x['DATA_PARTITION'], 
                    f"{x['GRD_product_identifier']}.SAFE", "measurement",
                    f"{x['GRD_product_identifier']}_{shoreline_type}.npy",
                )
            )
        )(xView3_SLC_GRD_correspondence)
        with open(str(shoreline_path), "rb") as f:
            shoreline = np.load(f, allow_pickle = True)
    
        if shoreline.size == 0:
            return np.array([])

        shoreline_meters = np.array(
            [shoreline[:, 0] * azimuthLineSpacingMeters,
            shoreline[:, 1] * rangePixelSpacingMeters]
        ).transpose()
        close_to_shore_mask = (
            mask_close_to_shore_predictions(
                shoreline_meters, predictions_meters,
                distance_from_shore_tolerance_meters,
                far_from_shore_value_meters
            )
        )
        close_to_shore_indices = (
            predictions.index[close_to_shore_mask].to_numpy()
        )
        return close_to_shore_indices

    srgrConvParams = get_srgrConvParams_from_GRD_annotation(
        GRD_vh_annotation_path
    )
    _, _, _, _, GRD_nearEdgeSlantRangeMeters = (
        get_spacing_and_timing_from_annotation(GRD_vh_annotation_path)
    )
    srgrConvParams_interpolator = get_linear_interpolator_of_srgrConvParams(
        srgrConvParams
    )
    swaths_predictions_close_to_shore_indices = []
    for swath_index in [1, 2, 3]:
        swath_mask = predictions['swath_index'] == swath_index
        """change to indexing rather than appending an merging after 
        close_to_shore labels are generated for the SARFish dataset
        """
        swath_predictions = predictions[swath_mask].copy()
        if len(swath_predictions) == 0:
            continue

        SLC_vh_swath_annotation_path = (
            lambda x: str(
                Path(
                    SARFish_root_directory, "SLC", x['DATA_PARTITION'], 
                    f"{x['SLC_product_identifier']}.SAFE",
                    "annotation", x[f'SLC_swath_{swath_index}_vh_annotation']
                )
            )
        )(xView3_SLC_GRD_correspondence)
        (azimuthLineSpacingMeters, rangePixelSpacingMeters, 
        firstAzimuthLineUnixTimeSeconds,
        azimuthLineTimeIntervalSeconds, swath_nearEdgeSlantRangeMeters) = (
            get_spacing_and_timing_from_annotation(
                SLC_vh_swath_annotation_path
            )
        )
        swath_predictions_meters = convert_SLC_image_coordinates_to_meters(
            swath_predictions['detect_scene_row'].to_numpy(), 
            swath_predictions['detect_scene_column'].to_numpy(),
            swath_nearEdgeSlantRangeMeters, GRD_nearEdgeSlantRangeMeters,
            rangePixelSpacingMeters, azimuthLineSpacingMeters,
            firstAzimuthLineUnixTimeSeconds, azimuthLineTimeIntervalSeconds,
            srgrConvParams_interpolator
        )
        swath_shoreline_path = (
            lambda x: str(
                Path(
                    SARFish_root_directory, "SLC", x['DATA_PARTITION'], 
                    f"{x['SLC_product_identifier']}.SAFE", "measurement",
                    f"{x['SLC_product_identifier']}_{swath_index}_{shoreline_type}.npy",
                )
            )
        )(xView3_SLC_GRD_correspondence)
        with open(str(swath_shoreline_path), "rb") as f:
            shoreline = np.load(f, allow_pickle = True)
    
        if shoreline.size == 0:
            continue

        shoreline_meters = convert_SLC_image_coordinates_to_meters(
            shoreline[:, 0], shoreline[:, 1],
            swath_nearEdgeSlantRangeMeters, GRD_nearEdgeSlantRangeMeters,
            rangePixelSpacingMeters, azimuthLineSpacingMeters,
            firstAzimuthLineUnixTimeSeconds, azimuthLineTimeIntervalSeconds,
            srgrConvParams_interpolator
        )
        close_to_shore_mask = (
            mask_close_to_shore_predictions(
                shoreline_meters, swath_predictions_meters, 
                distance_from_shore_tolerance_meters,
                far_from_shore_value_meters
            )
        )
        swaths_predictions_close_to_shore_indices.append(
            swath_predictions.index[close_to_shore_mask].to_numpy()
        )
    if len(swaths_predictions_close_to_shore_indices) == 0:
        return np.array([])
    
    return np.concatenate(swaths_predictions_close_to_shore_indices)

def compute_loc_performance(
        SARFish_root_directory: Path, 
        predictions: pd.DataFrame, groundtruth: np.ndarray, 
        xView3_SLC_GRD_correspondence: pd.DataFrame, 
        product_type: {"GRD", "SLC"},
        assignment_tolerance_meters: float = 200.0,
        costly_dist: bool = False
    ) -> Tuple[Dict[str, int], List[int], List[int]]:
    """Returns a tuple containing numpy arrays of the indices of the:
    1.  true positives in both the predictions and groundtruth 
        pandas.DataFrames
    2.  false positives in the predictions
    3.  false negatives in the groundtruth
    For the location detection task.
    """
    partition = xView3_SLC_GRD_correspondence['DATA_PARTITION']
    product_title = xView3_SLC_GRD_correspondence['scene_id']
    GRD_vh_annotation_path = (
        lambda x: str(
            Path(
                SARFish_root_directory, "GRD", x['DATA_PARTITION'], 
                f"{x[f'GRD_product_identifier']}.SAFE",
                "annotation", x['GRD_vh_annotation']
            )
        )
    )(xView3_SLC_GRD_correspondence)
    if product_type == "GRD":
        (azimuthLineSpacingMeters, rangePixelSpacingMeters, _, _, _) = (
            get_spacing_and_timing_from_annotation(
                GRD_vh_annotation_path           
            )
        )
        predictions_meters = np.array(
            [predictions['detect_scene_row'] * azimuthLineSpacingMeters,
            predictions['detect_scene_column'] * rangePixelSpacingMeters]
        ).transpose()
        groundtruth_meters = np.array(
            [groundtruth['detect_scene_row'] * azimuthLineSpacingMeters,
            groundtruth['detect_scene_column'] * rangePixelSpacingMeters]
        ).transpose()
    elif product_type == "SLC":
        srgrConvParams = get_srgrConvParams_from_GRD_annotation(
            GRD_vh_annotation_path
        )
        _, _, _, _, GRD_nearEdgeSlantRangeMeters = (
            get_spacing_and_timing_from_annotation(GRD_vh_annotation_path)
        )
        srgrConvParams_interpolator = get_linear_interpolator_of_srgrConvParams(
            srgrConvParams
        )
        swaths_predictions_meters = []
        swaths_groundtruth_meters = []
        for swath_index in [1, 2, 3]:
            swath_predictions = predictions[
                predictions['swath_index'] == swath_index
            ].copy()
            swath_groundtruth = groundtruth[
                groundtruth['swath_index'] == swath_index
            ].copy()
            if len(swath_predictions) == 0:
                continue

            SLC_vh_swath_annotation_path = (
                lambda x: str(
                    Path(
                        SARFish_root_directory, "SLC", x['DATA_PARTITION'], 
                        f"{x[f'SLC_product_identifier']}.SAFE",
                        "annotation", x[f'SLC_swath_{swath_index}_vh_annotation']
                    )
                )
            )(xView3_SLC_GRD_correspondence)
            (azimuthLineSpacingMeters, rangePixelSpacingMeters, 
            firstAzimuthLineUnixTimeSeconds,
            azimuthLineTimeIntervalSeconds, swath_nearEdgeSlantRangeMeters) = (
                get_spacing_and_timing_from_annotation(
                    SLC_vh_swath_annotation_path
                )
            )
            swath_predictions_meters = convert_SLC_image_coordinates_to_meters(
                swath_predictions['detect_scene_row'].to_numpy(),
                swath_predictions['detect_scene_column'].to_numpy(),
                swath_nearEdgeSlantRangeMeters, GRD_nearEdgeSlantRangeMeters,
                rangePixelSpacingMeters, azimuthLineSpacingMeters,
                firstAzimuthLineUnixTimeSeconds, azimuthLineTimeIntervalSeconds,
                srgrConvParams_interpolator
            )
            swath_groundtruth_meters = convert_SLC_image_coordinates_to_meters(
                swath_groundtruth['detect_scene_row'].to_numpy(),
                swath_groundtruth['detect_scene_column'].to_numpy(),
                swath_nearEdgeSlantRangeMeters, GRD_nearEdgeSlantRangeMeters,
                rangePixelSpacingMeters, azimuthLineSpacingMeters,
                firstAzimuthLineUnixTimeSeconds, azimuthLineTimeIntervalSeconds,
                srgrConvParams_interpolator
            )
            swaths_predictions_meters.append(swath_predictions_meters)
            swaths_groundtruth_meters.append(swath_groundtruth_meters)
        predictions_meters = np.concatenate(swaths_predictions_meters)
        groundtruth_meters = np.concatenate(swaths_groundtruth_meters)
    else:
        raise ValueError(f"product_type not specified")

    pairwise_distances_meters = distance_matrix(
        predictions_meters, groundtruth_meters, p = 2
    )
    if costly_dist:
        pairwise_distances_meters[
            pairwise_distances_meters > assignment_tolerance_meters
        ] = 9999999
    assignment_row_indices, assignment_column_indices = (
        linear_sum_assignment(pairwise_distances_meters)
    )
    assignments_under_threshold = (
        pairwise_distances_meters[
            assignment_row_indices, assignment_column_indices
        ] < assignment_tolerance_meters
    )
    assignment_row_indices = (
        assignment_row_indices[assignments_under_threshold]
    )
    assignment_column_indices = (
        assignment_column_indices[assignments_under_threshold]
    )
    true_positive_predictions_indices = predictions.index[
        assignment_row_indices
    ].to_numpy()
    true_positive_groundtruth_indices = groundtruth.index[
        assignment_column_indices
    ].to_numpy()
    true_positive_indices = {
        'groundtruth': true_positive_groundtruth_indices,
        'predictions': true_positive_predictions_indices
    }
    groundtruth_indices = groundtruth.index
    false_negative_indices = np.setdiff1d(
        groundtruth.index.to_numpy(), true_positive_indices['groundtruth'], 
        assume_unique = True 
    )
    false_positive_indices = np.setdiff1d(
        predictions.index.to_numpy(), true_positive_indices['predictions'], 
        assume_unique = True 
    )
    if  (
        (len(predictions) != true_positive_indices['groundtruth'].size + false_positive_indices.size) or
        (len(groundtruth) != true_positive_indices['predictions'].size + false_negative_indices.size)
    ):
        raise ValueError(
            f"tp, fp, fn calculation error."
        )
    return  (true_positive_indices, false_positive_indices, 
            false_negative_indices)

def mask_packaged_indices(
        true_positive_indices: np.ndarray, mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
    masked_true_positive_indices = {
        'groundtruth': true_positive_indices['groundtruth'][mask],
        'predictions': true_positive_indices['predictions'][mask]
    }
    return masked_true_positive_indices

def compute_class_performance(
        predictions: pd.Series, groundtruth: pd.Series, 
        previous_task_true_positive_indices: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], 
        Dict[str, np.ndarray]
    ]:
    """Returns a tuple containing numpy arrays of the indices of the:
    1.  true positives in both the predictions and groundtruth 
        pandas.DataFrames
    2.  false positives in the predictions
    3.  false negatives in the groundtruth
    for classification tasks. predictions and groundtruth contain the values for 
    the classification task being evaluated. Depends on a set of true postitives 
    from a previous task.
    """
    groundtruth = groundtruth.loc[previous_task_true_positive_indices['groundtruth']]
    predictions = predictions.loc[previous_task_true_positive_indices['predictions']]
    is_true_positive = (
        (groundtruth == True).to_numpy() & (predictions == True).to_numpy() 
    )
    is_false_positive = (
        (groundtruth == False).to_numpy() & (predictions == True).to_numpy() 
    )
    is_false_negative = (
        (groundtruth == True).to_numpy() & (predictions == False).to_numpy() 
    )
    is_true_negative = (
        (groundtruth == False).to_numpy() & (predictions == False).to_numpy()
    )
    dependent_task_true_positive_indices = mask_packaged_indices(
        previous_task_true_positive_indices, is_true_positive
    )
    dependent_task_false_positive_indices = mask_packaged_indices(
        previous_task_true_positive_indices, is_false_positive
    )
    dependent_task_false_negative_indices = mask_packaged_indices(
        previous_task_true_positive_indices, is_false_negative
    )
    dependent_task_true_negative_indices = mask_packaged_indices(
        previous_task_true_positive_indices, is_true_negative
    )
    return  (dependent_task_true_positive_indices,
            dependent_task_false_positive_indices,
            dependent_task_false_negative_indices,
            dependent_task_true_negative_indices)

def compute_length_performance(
        predictions: pd.Series, groundtruth: pd.Series, 
        previous_task_true_positive_indices: Dict[str, np.ndarray],
        max_object_length_meters: float = 500.0
    ) -> float:
    """Returns the "Aggregate Percentage Error" of the predicted lengths.
    """

    if len(groundtruth) == 0:
        length_performance = 0.0
        return length_performance

    previous_task_true_positive_indices = mask_packaged_indices(
        previous_task_true_positive_indices, 
        ~groundtruth.loc[previous_task_true_positive_indices['groundtruth']].isna()
    )
    valid_groundtruth = groundtruth.loc[
        previous_task_true_positive_indices['groundtruth']
    ].to_numpy()
    valid_predictions = predictions.loc[
        previous_task_true_positive_indices['predictions']
    ].to_numpy()
    groundtruth_length = np.minimum(valid_groundtruth, max_object_length_meters)
    predictions_length = np.minimum(valid_predictions, max_object_length_meters)
    relative_error = (
        np.abs(predictions_length - groundtruth_length) / groundtruth_length
    ).sum()
    length_performance = (1.0 - min((relative_error / valid_groundtruth.size), 1.0))
    return length_performance

def calculate_p_r_f(
        true_positive_indices: np.ndarray, 
        false_positive_indices: np.ndarray, false_negative_indices: np.ndarray
    ) -> Tuple[float, float, float]:
    """Returns the precision, recall, and F1 score of a particular task from 
    the numbers of true postives, false positives and false negatives.
    """

    true_positive_count = true_positive_indices.size
    false_positive_count = false_positive_indices.size
    false_negative_count = false_negative_indices.size
    try:
        precision = (
            true_positive_count / (true_positive_count + false_positive_count)
        )
    except ZeroDivisionError:
        precision = 0
    try:
        recall = (
            true_positive_count / (true_positive_count + false_negative_count)
        )
    except ZeroDivisionError:
        recall = 0
    try:
        fscore = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        fscore = 0

    if precision == np.nan or recall == np.nan or fscore == np.nan:
        return 0, 0, 0
    else:
        return precision, recall, fscore

def aggregate_f(
        location_f1_score: float, location_close_to_shore_f1_score: float,
        is_vessel_f1_score: float, is_fishing_f1_score: float, 
        vessel_length_regression_track_error: float,
    ) -> float:
    """Returns the xView3-SAR/SARFish challenge aggregate score
    """
    aggregate = (
        location_f1_score * (
            1 + vessel_length_regression_track_error + is_vessel_f1_score + 
            is_fishing_f1_score + location_close_to_shore_f1_score 
        ) / 5
    )
    return aggregate

def pprint_confusion_matrix(
        true_positive_indices: Dict[str, np.ndarray], 
        false_positive_indices: np.ndarray, false_negative_indices: np.ndarray,
        true_negative_indices: Dict[str, np.ndarray] = None,
        task: str = ""
    ):
    tp = true_positive_indices.size
    fp = false_positive_indices.size
    fn = false_negative_indices.size
    if true_negative_indices is not None:
        tn = true_negative_indices.size
    else:
        tn = "N/A"

    max_value_length = reduce(
        max, [len(str(tp)), len(str(fp)), len(str(fn)), len(str(tn))]
    )
    print(CBYELLOW + f"\n{task} " + CEND + "task confusion matrix:")
    print(
        22*" " + 
        "\u2554" + (2*(max_value_length + 6) + 1)*"\u2550" + "\u2557"
    )
    print(
        22*" " + 
        "\u2551".ljust(max_value_length + 2) + f"groundtruth" + 
        (max_value_length + 1)*" " + "\u2551"
    )
    print(
        22*" " + 
        "\u2560" + (max_value_length + 6)*"\u2550" + 
        "\u2564" + (max_value_length + 6)*"\u2550" + "\u2563"
    )
    print(
        22*" " + 
        "\u2551" + f" True  " + (max_value_length - 1)*" " + 
        "\u2502" + f" False " + (max_value_length - 1)*" " + 
        "\u2551"
    )
    print("\u2554" + 13*"\u2550" + "\u2566" + 7*"\u2550" + 
        "\u256C" + (max_value_length + 6)*"\u2550" + "\u256A" + 
        (max_value_length + 6)*"\u2550" + "\u2563"
    )
    print(
        "\u2551" + " predictions " + 
        "\u2551" + " True  " + 
        "\u2551" + " tp:" + f" {tp} ".rjust(max_value_length + 2) + 
        "\u2502" + f" fp:" + f" {fp} ".rjust(max_value_length + 2) + "\u2551"
    )
    print(
        "\u2551" + 13*" " + 
        "\u255F" + 7*"\u2500" + 
        "\u256B" + (max_value_length + 6)*"\u2500" + 
        "\u253C" + (max_value_length + 6)*"\u2500" + "\u2562"
    )
    print(
        "\u2551" + 13*" " + 
        "\u2551" + " False " + 
        "\u2551" + " fn:" + f" {fn} ".rjust(max_value_length + 2) + 
        "\u2502" + f" tn:" + f" {tn} ".rjust(max_value_length + 2) + "\u2551"
    )
    print(
        "\u255A" + 13*"\u2550" + 
        "\u2569" + 7*"\u2550" + 
        "\u2569" + (max_value_length + 6)*"\u2550" + 
        "\u2567" + (max_value_length + 6)*"\u2550" + "\u255D"
    )

def pprint_p_r_f(
        precision: float, recall: float, f1_score: float, task: str
    ):
    colors = [CBLUE, CRED, CPURPLE]
    metrics = ["precision", "recall", "F1 score"]
    values = [precision, recall, f1_score]
    max_len = max([len(x) for x in metrics]) + 2
    print(CBYELLOW + f"\n{task} " + CEND + "task performance:")
    print("\u2554" + max_len * "\u2550" + "\u2564" + 20*"\u2550" + "\u2557")
    for color, metric, value in zip(colors, metrics, values):
        print(
            "\u2551" + color + f" {metric}".ljust(max_len) + CEND + "\u2502" + 
            f" {value:.16f}".ljust(20) + "\u2551"
        )
    print("\u255A" + max_len*"\u2550" + "\u2567" + 20*"\u2550" + "\u255D")

def pprint_metric(metric: str, value: float, task: str, color: str):
    max_len = len(str(metric)) + 2
    print(CBYELLOW + f"\n{task} " + CEND + "task performance:")
    print("\u2554" + max_len * "\u2550" + "\u2564" + 20*"\u2550" + "\u2557")
    print(
        "\u2551" + color + f" {metric}".ljust(max_len) + CEND + "\u2502" + 
        f" {value:.16f}".ljust(20) + "\u2551"
    )
    print("\u255A" + max_len*"\u2550" + "\u2567" + 20*"\u2550" + "\u255D")

def classification_score(
        predictions: pd.DataFrame, groundtruth: pd.DataFrame, 
        location_true_positive_indices: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
    """Returns the scores for the classification tasks.
    """
    (is_vessel_true_positive_indices, is_vessel_false_positive_indices, 
    is_vessel_false_negative_indices, is_vessel_true_negative_indices) = (
        compute_class_performance(
            predictions['is_vessel'], groundtruth['is_vessel'],
            location_true_positive_indices, 
        )
    )
    pprint_confusion_matrix(
        is_vessel_true_positive_indices['predictions'], 
        is_vessel_false_positive_indices['predictions'],
        is_vessel_false_negative_indices['predictions'],
        is_vessel_true_negative_indices['predictions'],
        task = "is_vessel"
    )
    is_vessel_precision, is_vessel_recall, is_vessel_f1_score = calculate_p_r_f(
        is_vessel_true_positive_indices['predictions'],
        is_vessel_false_positive_indices['predictions'],
        is_vessel_false_negative_indices['predictions']
    )
    pprint_p_r_f(
        is_vessel_precision, is_vessel_recall, is_vessel_f1_score, 
        task = "is_vessel"
    )

    location_true_positive_indices_where_is_vessel_true = mask_packaged_indices(
        location_true_positive_indices, 
        (groundtruth.loc[location_true_positive_indices['groundtruth'], 'is_vessel'] == True).to_numpy()
    )
    (is_fishing_true_positive_indices, is_fishing_false_positive_indices, 
    is_fishing_false_negative_indices, is_fishing_true_negative_indices) = (
        compute_class_performance(
            predictions['is_fishing'],
            groundtruth['is_fishing'],
            location_true_positive_indices_where_is_vessel_true, 
        )
    )
    pprint_confusion_matrix(
        is_fishing_true_positive_indices['predictions'], 
        is_fishing_false_positive_indices['predictions'],
        is_fishing_false_negative_indices['predictions'],
        is_fishing_true_negative_indices['predictions'],
        task = "is_fishing"
    )
    is_fishing_precision, is_fishing_recall, is_fishing_f1_score = (
        calculate_p_r_f(
            is_fishing_true_positive_indices['predictions'], 
            is_fishing_false_positive_indices['predictions'],
            is_fishing_false_negative_indices['predictions']
        )
    )
    pprint_p_r_f(
        is_fishing_precision, is_fishing_recall, is_fishing_f1_score, 
        task = "is_fishing"
    )
    classification_scores = {
        "vessel_fscore": is_vessel_f1_score,
        "fishing_fscore": is_fishing_f1_score,
    }
    return classification_scores

def compute_maritime_object_detection_track_performance(
        location_f1_score: float, location_close_to_shore_f1_score: float
    ) -> float:
    return np.mean([location_f1_score, location_close_to_shore_f1_score])

def compute_maritime_object_classification_track_performance(
        is_vessel_f1_score: float, is_fishing_f1_score: float
    ) -> float:
    return np.mean([is_vessel_f1_score, is_fishing_f1_score])

def score(
        predictions: pd.DataFrame, groundtruth: pd.DataFrame, 
        xView3_SLC_GRD_correspondences: pd.DataFrame,
        SARFish_root_directory: str, product_type: {"GRD", "SLC"}, 
        shoreline_type: {"xView3_shoreline", "global_shoreline_vector"} = None,
        distance_from_shore_tolerance_meters: float = 2000.0, 
        assignment_tolerance_meters: float = 200.0,
        score_all: bool = False, drop_low_detect: bool = True, 
        costly_dist: bool = False, evaluation_mode: bool = True,
        far_from_shore_value_meters: float = 9999990.0
    ) -> Dict[str, float]:
    """Returns the scores for a set of predictions against all the 
    xView3-SAR/SARFish tasks.
    """
    print(f"shoreline_type: ".ljust(20) + f"{shoreline_type}")
    print(f"score_all: ".ljust(20) + f"{score_all}")
    print(f"drop_low_detect: ".ljust(20) + f"{drop_low_detect}")
    print(f"costly_dist: ".ljust(20) + f"{costly_dist}")
    print(f"evaluation_mode: ".ljust(20) + f"{evaluation_mode}\n")
    groundtruth_scenes = groundtruth['scene_id'].unique()
    predictions_scenes = predictions['scene_id'].unique()
    scene_ids_to_evaluate = reduce(
        np.intersect1d,
        [groundtruth_scenes, predictions_scenes,  
        xView3_SLC_GRD_correspondences['scene_id']]
    )
    if scene_ids_to_evaluate.size == 0:
        raise ValueError(
            "The intersection of scene_id(s) in the predictions and "
            "groundtruth pandas.DataFrames is empty"
        )

    if (evaluation_mode == True) and (len(groundtruth_scenes) != len(scene_ids_to_evaluate)):
        raise ValueError(
            f"Evaluation mode == {evaluation_mode}: The predictions must "
            "contain results for each scene in the groundtruth."
        )

    if score_all != True:
        if drop_low_detect:
            predictions = drop_low_confidence_preds(
                SARFish_root_directory, predictions, groundtruth, 
                xView3_SLC_GRD_correspondences, product_type,
                assignment_tolerance_meters, costly_dist
            )
        groundtruth = groundtruth[
            groundtruth['confidence'].isin(["HIGH", "MEDIUM"])
        ]

    if shoreline_type != None:
        location_close_to_shore_true_positive_indices = {
            'predictions': [], 'groundtruth': []
        }
        location_close_to_shore_false_positive_indices = []
        location_close_to_shore_false_negative_indices = []

    location_true_positive_indices = {'predictions': [], 'groundtruth': []}
    location_false_positive_indices = []
    location_false_negative_indices = []
    is_scene_ids_to_evaluate = (
        xView3_SLC_GRD_correspondences['scene_id'].isin(
            scene_ids_to_evaluate
        )
    )
    xView3_SLC_GRD_correspondences = xView3_SLC_GRD_correspondences[
        is_scene_ids_to_evaluate
    ]
    for index, xView3_SLC_GRD_correspondence in xView3_SLC_GRD_correspondences.iterrows():
        print(
            f"evaluating " 
            f"{xView3_SLC_GRD_correspondence[f'{product_type}_product_identifier']}"
        )
        scene_predictions = predictions[
            predictions['scene_id'] == 
            xView3_SLC_GRD_correspondence['scene_id']
        ]
        scene_groundtruth = groundtruth[
            groundtruth['scene_id'] == 
            xView3_SLC_GRD_correspondence['scene_id']
        ]
        evaluation_columns = ['detect_scene_row', 'detect_scene_column']
        if product_type == "SLC":
           evaluation_columns.append('swath_index') 
 
        (location_true_positive_indices_, location_false_positive_indices_, 
        location_false_negative_indices_) = compute_loc_performance(
            SARFish_root_directory,
            scene_predictions[evaluation_columns],
            scene_groundtruth[evaluation_columns], 
            xView3_SLC_GRD_correspondence, product_type,
            assignment_tolerance_meters, costly_dist
        )
        location_true_positive_indices['predictions'].append(
            location_true_positive_indices_['predictions']
        )
        location_true_positive_indices['groundtruth'].append(
            location_true_positive_indices_['groundtruth']
        )
        location_false_positive_indices.append(
            location_false_positive_indices_
        )
        location_false_negative_indices.append(
            location_false_negative_indices_
        )
        if shoreline_type == None:
            continue

        close_to_shore_assignment_tolerance_meters = (
            distance_from_shore_tolerance_meters + assignment_tolerance_meters
        )
        scene_predictions_close_to_shore_indices = get_shore_preds(
            SARFish_root_directory, xView3_SLC_GRD_correspondence, 
            scene_predictions[evaluation_columns],
            product_type, shoreline_type,
            close_to_shore_assignment_tolerance_meters,
            far_from_shore_value_meters 
        )
        scene_groundtruth_close_to_shore = scene_groundtruth[
            scene_groundtruth[f'{shoreline_type}_distance_from_shore_km'] < 
            close_to_shore_assignment_tolerance_meters / 1000
        ]
        if (
            (scene_predictions_close_to_shore_indices.size == 0) and 
            (len(scene_groundtruth_close_to_shore) == 0)
        ):
            continue

        scene_predictions_close_to_shore = scene_predictions.loc[
            scene_predictions_close_to_shore_indices
        ]
        (location_close_to_shore_true_positive_indices_, 
        location_close_to_shore_false_positive_indices_,
        location_close_to_shore_false_negative_indices_) = (
            compute_loc_performance(
                SARFish_root_directory,
                scene_predictions_close_to_shore,
                scene_groundtruth_close_to_shore, 
                xView3_SLC_GRD_correspondence, product_type, 
                assignment_tolerance_meters, costly_dist
            )
        )
        location_close_to_shore_true_positive_indices['predictions'].append(
            location_close_to_shore_true_positive_indices_['predictions']
        )
        location_close_to_shore_true_positive_indices['groundtruth'].append(
            location_close_to_shore_true_positive_indices_['groundtruth']
        )
        location_close_to_shore_false_positive_indices.append(
            location_close_to_shore_false_positive_indices_
        )
        location_close_to_shore_false_negative_indices.append(
            location_close_to_shore_false_negative_indices_
        )

    location_true_positive_indices = {
        'predictions': np.concatenate(
            location_true_positive_indices['predictions'], axis = 0
        ),
        'groundtruth': np.concatenate(
            location_true_positive_indices['groundtruth'], axis = 0
        ),
    }
    location_false_positive_indices = np.concatenate(
        location_false_positive_indices, axis = 0
    )
    location_false_negative_indices = np.concatenate(
        location_false_negative_indices, axis = 0
    )
    pprint_confusion_matrix(
        location_true_positive_indices['predictions'],
        location_false_positive_indices,
        location_false_negative_indices,
        task = "location"
    )
    location_precision, location_recall, location_f1_score = calculate_p_r_f(
        location_true_positive_indices['predictions'], 
        location_false_positive_indices,
        location_false_negative_indices
    )
    pprint_p_r_f(
        location_precision, location_recall, location_f1_score, 
        task = "Maritime Object Detection Task"
    )

    if shoreline_type == None:
        location_close_to_shore_precision = 0
        location_close_to_shore_recall = 0
        location_close_to_shore_f1_score = 0
    elif (
        len(
            groundtruth[f'{shoreline_type}_distance_from_shore_km'] < 
            far_from_shore_value_meters / 1000
        ) == 0
    ):
        location_close_to_shore_precision = 0
        location_close_to_shore_recall = 0
        location_close_to_shore_f1_score = 0
    else:
        location_close_to_shore_true_positive_indices = {
            'predictions': np.concatenate(
                location_close_to_shore_true_positive_indices['predictions'],
                axis = 0
            ),
            'groundtruth': np.concatenate(
                location_close_to_shore_true_positive_indices['groundtruth'],
                axis = 0
            ),
        }
        location_close_to_shore_false_positive_indices = np.concatenate(
            location_close_to_shore_false_positive_indices, axis = 0
        )
        location_close_to_shore_false_negative_indices = np.concatenate(
            location_close_to_shore_false_negative_indices, axis = 0
        )
        pprint_confusion_matrix(
            location_close_to_shore_true_positive_indices['predictions'], 
            location_close_to_shore_false_positive_indices,
            location_close_to_shore_false_negative_indices,
            task = "location_close_to_shore"
        )
        (location_close_to_shore_precision, location_close_to_shore_recall, 
        location_close_to_shore_f1_score) = calculate_p_r_f(
            location_close_to_shore_true_positive_indices['predictions'], 
            location_close_to_shore_false_positive_indices,
            location_close_to_shore_false_negative_indices
        )
        pprint_p_r_f(
            location_close_to_shore_precision, 
            location_close_to_shore_recall, 
            location_close_to_shore_f1_score, 
            task = "Close-to-Shore Object Detection Task"
        )

    classification_scores = classification_score(
        predictions, groundtruth, location_true_positive_indices
    )
    is_vessel_f1_score = classification_scores['vessel_fscore']
    is_fishing_f1_score = classification_scores['fishing_fscore']

    vessel_length_regression_track_error = compute_length_performance(
        predictions['vessel_length_m'], groundtruth['vessel_length_m'], 
        location_true_positive_indices
    )

    aggregate = aggregate_f( 
        location_f1_score, location_close_to_shore_f1_score, 
        is_vessel_f1_score, is_fishing_f1_score, 
        vessel_length_regression_track_error
    )
    pprint_metric("score", aggregate, "aggregate", CCYAN)

    print('\n' + 80 * "\u2550")
    print(CPURPLE + f"SARFish Challenge Tracks" + CEND)
    maritime_object_detection_track_score = (
        compute_maritime_object_detection_track_performance(
            location_f1_score, location_close_to_shore_f1_score
        ) 
    )
    pprint_metric(
        "score", maritime_object_detection_track_score, 
        "Maritime Object Detection Track", CCYAN
    )
    maritime_object_classification_track_score = (
        compute_maritime_object_classification_track_performance(
            is_vessel_f1_score, is_fishing_f1_score
        )
    )
    pprint_metric(
        "score", maritime_object_classification_track_score, 
        "Maritime Object Classification Track", CCYAN
    )
    pprint_metric(
        "error", vessel_length_regression_track_error, 
        "Vessel Length Regression Track", CCYAN
    )
    scores = {
        "loc_fscore": location_f1_score,
        "loc_fscore_shore": location_close_to_shore_f1_score,
        "vessel_fscore": is_vessel_f1_score,
        "fishing_fscore": is_fishing_f1_score,
        #"length_acc": vessel_length_regression_track_error,
        "aggregate": aggregate,
        "maritime_object_detection_track_score": maritime_object_detection_track_score,
        "maritime_object_classification_track_score": maritime_object_classification_track_score,
        "vessel_length_regression_track_error": vessel_length_regression_track_error
    }
    return scores

def SARFish_metric(
        predictions_path: Path, groundtruth_path: Path,
        xView3_SLC_GRD_correspondences_path: Path, scene_id: str,
        SARFish_root_directory: Path, product_type: {"GRD", "SLC"}, 
        shoreline_type: {"xView3_shoreline", "global_shoreline_vector"},
        assignment_tolerance_meters: float, 
        distance_from_shore_tolerance_meters: float, 
        score_all: bool = False, drop_low_detect: bool = True, 
        costly_dist: bool = False, evaluation_mode: bool = True
    ) -> Dict[str, int]:
    """Calls the score function. Handles loading and selection of predictions 
    and groundtruth from command line arguments.
    """
    groundtruth = pd.read_csv(str(groundtruth_path))
    predictions = pd.read_csv(str(predictions_path))
    xView3_SLC_GRD_correspondences = pd.read_csv(
        str(xView3_SLC_GRD_correspondences_path)
    )
    if scene_id != None:
        groundtruth = groundtruth[groundtruth['scene_id'] == scene_id]
        predictions = predictions[predictions['scene_id'] == scene_id]

    scores = score(
        predictions, groundtruth, xView3_SLC_GRD_correspondences, 
        SARFish_root_directory, product_type, shoreline_type, 
        distance_from_shore_tolerance_meters, assignment_tolerance_meters, 
        score_all, drop_low_detect, costly_dist, evaluation_mode
    )
    print("")
    for key, value in scores.items():
        print(f"{key}: ".ljust(47) + f"{value}")

    return scores

def main():
    parser = argparse.ArgumentParser(
        description = "Score assignment for SARFish dataset GRD, SLC products."
    )
    parser.add_argument(
        "-p", "--predictions", help="/path/to/inference_file.csv", type = str,
        required = True
    )
    parser.add_argument(
        "-g", "--groundtruth", help="/path/to/labels_file.csv", type = str,
        required = True
    )
    parser.add_argument(
        "-s", "--scene_id", help="scene_id to evaluate", type = str,
        default=None
    )
    parser.add_argument(
        "-o", "--output", help="/path/to/output_file.json", type = str,
    )
    parser.add_argument(
        "--sarfish_root_directory", 
        help="/path/to/SARFish/root/directory", required = True
    )
    parser.add_argument(
        "--product_type", help="SARFish product type: {GRD, SLC}", 
        type = str, required = True
    )
    parser.add_argument(
        "-x", "--xview3_slc_grd_correspondences", 
        help = (
            "/path/to/file/encoding/correspondence/between/xView3/and/"
            "SARFish/SLC/GRD.csv"
        ), type = str, required = True
    )
    parser.add_argument(
        "--distance_tolerance", default = 200.0,
        help = (
            "Distance tolerance for assignment of true positive detections "
            " in (metres)"
        ), type=float
    )
    parser.add_argument(
        "--shore_tolerance", default = 2000.0,
        help = (
            "Distance tolerance for assigning close-to-shore detections "
            "(metres)"
        ), type = float
    )
    parser.add_argument(
        "--shore_type", 
        help = (
            "Type of shoreline to evaluate the close-to-shore performance on: "
            " 'xView3_shoreline': The orignal xView3 shoreline vector, "
            " 'global_shoreline_vector': GlobalIslands global shoreline vector"
        ), type = str
    )
    
    # Boolean parameters
    parser.add_argument(
        "--score_all", action = "store_true",
        help = (
            "Score against all ground truth labels (inclusive of low confidence "
            "labels). Default: False"
        ),
    )
    parser.add_argument(
        "--no-score_all", action = "store_false",
        help = (
            "Do not score against all ground truth labels (inclusive of low "
            "confidence labels)."
        ),
    )
    parser.add_argument(
        "--drop_low_detect", action = "store_true",
        help = (
            "Drop predictions that are matched to low confidence labels. "
            "Default: True."
        )
    )
    parser.add_argument(
        "--no-drop_low_detect", dest = "drop_low_detect", action = "store_false",
        help = (
            "Do not drop predictions that are matched to low confidence labels."
        )
    )
    parser.add_argument(
        "--costly_dist", action = "store_true",
        help = (
            "Assign a large number (9999999) to distances greater than the "
            "shore_tolerance threshold. Default: False"
        ),
    )
    parser.add_argument(
        "--no-costly_dist", dest = "costly_dist", action = "store_false",
        help = (
            "Do not to assign a large number (9999999) to distances "
            "greater than the shore_tolerance threshold."
        ),
    )
    parser.add_argument(
        "--evaluation_mode", action = "store_true", help = (
            "Sets the condition for challenge evalutation. The predictions will "
            "be evaluated against all the scenes present in the dataset partition, " 
            "regardless if they appear in the predictions or not. Default: True"
        ) 
    )
    parser.add_argument(
        "--no-evaluation_mode", dest = "evaluation_mode", action = "store_false", help = (
            "Do not set the condition for challenge evalutation. The predictions "
            "will be evaluated against ONLY the scenes present in the predictions." 
        ) 
    )
    # defaults
    parser.set_defaults(
        drop_low_detect = True, costly_dist = False, score_all = False, 
        evaluation_mode = True
    )
    args = parser.parse_args()
    computed_scores = SARFish_metric(
        Path(args.predictions), Path(args.groundtruth), 
        Path(args.xview3_slc_grd_correspondences), args.scene_id, 
        Path(args.sarfish_root_directory), args.product_type, args.shore_type, 
        args.distance_tolerance, args.shore_tolerance, args.score_all, 
        args.drop_low_detect, args.costly_dist, args.evaluation_mode
    )

if __name__ == "__main__":
    main()
