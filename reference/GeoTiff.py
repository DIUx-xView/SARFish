#!/usr/bin/env python3

from pathlib import Path
from typing import Union, List, Tuple, Dict

from osgeo import gdal
import numpy as np
import torch

def get_band_shape(band :gdal.Band) -> Tuple[int, int]:
    number_of_band_columns = band.XSize
    number_of_band_rows = band.YSize
    return number_of_band_rows, number_of_band_columns

def convert_sentinel_1_image_to_tensor(
        data: np.ndarray, product_type: {"GRD", "SLC"}
    ) -> torch.Tensor:
    """
    Converts a numpy array containing SARFish Sentinel-1 GRD or SLC product data
    to torch.Tensor with a datatype that preserves precision.

    The following table relates the GRD and SLC data types with the available 
    numpy and torch dtypes that preserve the precison:

    | Sentinel-1 Product | GDALDataType name | numpy            | torch            |
    | ------------------ | ----------------- | ---------------- | ---------------- |
    | GRD                | GDT_UInt16        | numpy.uint16     | torch.int32      |
    | SLC                | GDT_CInt32        | numpy.complex64  | torch.complex64  |
    """
    if product_type == "SLC":
        data = torch.from_numpy(data) 
    elif product_type == "GRD":
        data = torch.from_numpy(data.astype(np.int32)) 
    
    return data

def load_GeoTiff(image_filepath: str) -> Tuple[
        np.ndarray, np.ndarray, Union[None, int, float], int
    ]:
    """
    Loads a Sentinel-1 SARFish GRD or SLC product preserving the datatype precision.
    Returns two numpy ndarrays, data representing values and a mask indicating 
    nodata values. Also returns the nodata value specified in the image file and 
    the GDALDataType. gdal.GetDataTypeName can be used to recover the name of 
    the GDALDataType.

    GDAL Supports data formats that are not supported by numpy and torch. 
    GDAL's python bindings of the ReadAsArray do the conversion to the 
    appropriate numpy dtype automatically. Conversion to the correct torch dtype 
    is shown in convert_sentinel_1_image_to_tensor. 

    The following table relates the GRD and SLC data types with the available 
    numpy and torch dtypes that preserve the precison:

    | Sentinel-1 Product | GDALDataType name | numpy            | torch            |
    | ------------------ | ----------------- | ---------------- | ---------------- |
    | GRD                | GDT_UInt16        | numpy.uint16     | torch.int32      |
    | SLC                | GDT_CInt32        | numpy.complex64  | torch.complex64  |

    GDALDataType: 
    https://gdal.org/doxygen/gdal_8h.html#a22e22ce0a55036a96f652765793fb7a4

    gdal.GetDataTypeByName:
    https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.GetDataTypeByName

    gdal.GetDataTypeName: 
    https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.GetDataTypeName

    GDALRasterBand::ReadAsArray:
    https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Band.ReadAsArray

    GDALRasterBand::GetMaskFlags gets the flags of the mask band associated 
    with the band it is called on
    https://gdal.org/doxygen/classGDALRasterBand.html#a181a931c6ecbdd8c84c5478e4aa48aaf

    Getting mask from nodata value or associated .msk file if they exist
    https://gdal.org/development/rfc/rfc15_nodatabitmask.html 
    """
    Dataset = gdal.Open(image_filepath)
    data_band = Dataset.GetRasterBand(1)
    number_of_data_band_rows, number_of_data_band_columns = get_band_shape(
        data_band
    )
    data_band_data_type_enum = data_band.DataType
    data = data_band.ReadAsArray(
        0, 0, number_of_data_band_columns, number_of_data_band_rows
    )
    if data_band.GetNoDataValue() != None:
        nodata_value = data_band.GetNoDataValue()
    elif Dataset.GetMetadata().get('NODATA_VALUES') != None:
        nodata_value = Dataset.GetMetadata().get('NODATA_VALUES')
        try:
            nodata_value = int(nodata_value)
        except:
            nodata_value = float(nodata_value)
    else:
        nodata_value = None

    nodata_mask_band = data_band.GetMaskBand()
    nodata_mask_band_flag = data_band.GetMaskFlags()
    del data_band
    nodata_mask = np.zeros_like(data, dtype = bool)
    if nodata_mask_band_flag != 1:
        number_of_nodata_mask_band_rows, number_of_nodata_mask_band_columns = (
            get_band_shape(nodata_mask_band)
        )
        nodata_mask = nodata_mask_band.ReadAsArray(
            0, 0, number_of_nodata_mask_band_columns, number_of_nodata_mask_band_rows
        )
        # consistent with the way np.ma.nodata_masked_array defines nodata_masks
        # where the nodata_mask == True where the data is invalid
        nodata_mask = ~nodata_mask.astype(bool)

    del nodata_mask_band
    del Dataset
    return data, nodata_mask, nodata_value, data_band_data_type_enum
