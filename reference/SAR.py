#!/usr/bin/env python3

from datetime import datetime, timezone
import dateutil
from pathlib import Path
import re
from typing import Tuple, List, Union, Dict
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from SARFish_constants import lightSpeedMetersPerSecond

def datetime_to_unix_time(
        dates: Union[datetime, pd.Series]
    ) -> Union[float, pd.Series]:
    unix_dates = dates - datetime(1970, 1, 1, tzinfo=timezone.utc)
    if not isinstance(dates, pd.Series):
        return unix_dates.total_seconds()
    return unix_dates.apply(lambda x: x.total_seconds())

def string_to_list(string_of_floats: str) -> List[float]:
    string_of_floats = re.sub(r'^[^0-9\-]*', '', string_of_floats)
    string_of_floats = re.sub(r'[^0-9]*$', '', string_of_floats) 
    list_of_strings = re.split(' ', string_of_floats)
    return [float(string) for string in list_of_strings]

def get_spacing_and_timing_from_annotation(
        annotation_path: Path
    ) -> Tuple[float, float, float, float, float]:
    with open(annotation_path, "r") as f:
        annotation_str = f.read()

    product = ET.fromstring(annotation_str)
    imageInformation = (
        product.find('imageAnnotation').find('imageInformation')
    )
    rangePixelSpacingMeters = float(
        imageInformation.find('rangePixelSpacing').text
    )
    azimuthLineSpacingMeters = float(
        imageInformation.find('azimuthPixelSpacing').text
    )
    firstAzimuthLineUTC = imageInformation.find('productFirstLineUtcTime').text 
    firstAzimuthLineUTC = dateutil.parser.parse(firstAzimuthLineUTC)
    firstAzimuthLineUTC = firstAzimuthLineUTC.replace(tzinfo = timezone.utc)
    firstAzimuthLineUnixTimeSeconds = datetime_to_unix_time(firstAzimuthLineUTC)
    azimuthLineTimeIntervalSeconds = float(
        imageInformation.find('azimuthTimeInterval').text
    )
    slantRangeOriginTimeSeconds = float(
        imageInformation.find('slantRangeTime').text
    )
    nearEdgeSlantRangeMeters = (
        slantRangeOriginTimeSeconds * ( lightSpeedMetersPerSecond / 2 )
    )
    return  (azimuthLineSpacingMeters, rangePixelSpacingMeters, 
            firstAzimuthLineUnixTimeSeconds, azimuthLineTimeIntervalSeconds, 
            nearEdgeSlantRangeMeters)

def get_srgrConvParams_from_GRD_annotation(
        annotation_path: Path
    ) -> pd.DataFrame:
    with open(annotation_path, "r") as f:
        annotation_str = f.read()

    product = ET.fromstring(annotation_str)
    coordinateConversionList = (
        product.find('coordinateConversion').find('coordinateConversionList')
    )
    if (len(coordinateConversionList) == 0):
        raise ValueError(
            f"No coordinate conversion parameters. {annotation_path} may not be "
            "a Sentinel-1 GRD annotation filepath."
        )

    data = {
        'azimuthLineTimeUTC':       [],
        'slantRangeTimeSeconds':    [],
        'sr0':                      [],
        'srgrCoefficients':         [],
        'gr0':                      [],
        'grsrCoefficients':         [],
    }
    for coordinateConversion in coordinateConversionList: 
        azimuthLineTimeUTC = dateutil.parser.parse(
            coordinateConversion.find('azimuthTime').text
        )
        azimuthLineTimeUTC = azimuthLineTimeUTC.replace(tzinfo = timezone.utc)
        data['azimuthLineTimeUTC'].append(azimuthLineTimeUTC)
        data['slantRangeTimeSeconds'].append(
            float(coordinateConversion.find('slantRangeTime').text)
        )
        data['sr0'].append(
            float(coordinateConversion.find('sr0').text)
        )
        data['srgrCoefficients'].append(
            string_to_list(coordinateConversion.find('srgrCoefficients').text)
        )
        data['gr0'].append(
            float(coordinateConversion.find('gr0').text)
        )
        data['grsrCoefficients'].append(
            string_to_list(coordinateConversion.find('grsrCoefficients').text)
        )
    srgrConvParams = pd.DataFrame(data)
    srgrConvParams['azimuthUnixTimeSeconds'] = datetime_to_unix_time(
        srgrConvParams['azimuthLineTimeUTC'] 
    )
    return srgrConvParams

def get_linear_interpolator_of_srgrConvParams(
        srgrConvParams: pd.DataFrame
    ) -> make_interp_spline:
    srgr_coefficients_lists = np.array(list(srgrConvParams['srgrCoefficients']))
    ground_range_origins_metres = srgrConvParams['gr0'].to_numpy()[:, None]
    linear_interpolator = make_interp_spline(
        srgrConvParams['azimuthUnixTimeSeconds'],
        np.concatenate(
            [srgr_coefficients_lists, ground_range_origins_metres], 
            axis=1
        )
    )
    return linear_interpolator

def convert_slant_range_to_ground_range(
        slant_ranges_from_near_edge_meters: Union[float, np.ndarray],
        srgrConvParams_interpolator: make_interp_spline, 
        azimuthLineUnixTimesSeconds: Union[float, np.ndarray]
    ) -> Union[float, np.array]:
    results = np.atleast_2d(
        srgrConvParams_interpolator(azimuthLineUnixTimesSeconds)
    )
    interpolated_srgr_coefficients = results[:, :-1]
    interpolated_ground_range_origins_metres = results[:, -1]
    powers = np.power(
        np.atleast_1d(slant_ranges_from_near_edge_meters)[:, None], 
        np.arange(interpolated_srgr_coefficients.shape[1])
    )
    ground_ranges_meters = (
        np.sum(interpolated_srgr_coefficients * powers, axis=1) - 
        interpolated_ground_range_origins_metres
    )
    return ground_ranges_meters

def convert_SLC_image_coordinates_to_meters(
        rows: np.ndarray, columns: np.ndarray,
        swath_nearEdgeSlantRangeMeters: float, 
        GRD_nearEdgeSlantRangeMeters: float,
        rangePixelSpacingMeters: float, 
        azimuthLineSpacingMeters: float, 
        firstAzimuthLineUnixTimeSeconds: float,
        azimuthLineTimeIntervalSeconds: float,
        srgrConvParams_interpolator: make_interp_spline
    ) -> np.ndarray:
    slant_ranges_meters = (
        swath_nearEdgeSlantRangeMeters - GRD_nearEdgeSlantRangeMeters + 
        columns * rangePixelSpacingMeters
    )
    azimuths_unix_times = (
        firstAzimuthLineUnixTimeSeconds + 
        rows * azimuthLineTimeIntervalSeconds
    )
    ground_ranges_meters = convert_slant_range_to_ground_range(
        slant_ranges_meters, srgrConvParams_interpolator, azimuths_unix_times
    )
    azimuths_meters = rows * azimuthLineSpacingMeters
    image_coordinates_meters = (
        np.array([azimuths_meters, ground_ranges_meters]).transpose()
    )
    return image_coordinates_meters
