#!/usr/bin/env python3

"""
SARFish GRD example:
! SARFISH_ROOT_DIRECTORY=; ./visualise_labels.py --xview3_slc_grd_correspondences ./labels/xView3_SLC_GRD_correspondences.csv -i "${SARFISH_ROOT_DIRECTORY}"SARFish/GRD/validation/S1B_IW_GRDH_1SDV_20200803T075721_20200803T075746_022756_02B2FF_033A.SAFE/measurement/s1b-iw-grd-vh-20200803t075721-20200803t075746-022756-02b2ff-002_SARFish.tiff -l "${SARFISH_ROOT_DIRECTORY}"SARFish/GRD/validation/GRD_validation.csv -k GRD_product_identifier -v S1B_IW_GRDH_1SDV_20200803T075721_20200803T075746_022756_02B2FF_033A

SARFish SLC example:
! SARFISH_ROOT_DIRECTORY=; ./visualise_labels.py --xview3_slc_grd_correspondences ./labels/xView3_SLC_GRD_correspondences.csv -i "${SARFISH_ROOT_DIRECTORY}"SARFish/GRD/validation/S1B_IW_GRDH_1SDV_20200803T075721_20200803T075746_022756_02B2FF_033A.SAFE/measurement/s1b-iw-grd-vh-20200803t075721-20200803t075746-022756-02b2ff-002_SARFish.tiff -l "${SARFISH_ROOT_DIRECTORY}"SARFish/GRD/validation/GRD_validation.csv -k GRD_product_identifier -v S1B_IW_GRDH_1SDV_20200803T075721_20200803T075746_022756_02B2FF_033A
"""

import argparse
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
from osgeo import gdal
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from vispy import scene, app, visuals
import vispy

from GeoTiff import load_GeoTiff, convert_sentinel_1_image_to_tensor

def scale_sentinel_1_image(
        data: np.ndarray, product_type: {"GRD", "SLC"}
    ) -> np.ndarray:
    """
    Transforms SARFish sentinel-1 GRD and SLC products for viewing. Applies 
    log10 decibel scaling to GRD and SLC imagery. SLC image is also 
    "detected" [1] Page 103 - Section 7-2, step 12.

    Even on the cpu, operations on torch.Tensors are faster than numpy operations,
    hence the conversion to torch.Tensors and back

    1.
    @article{piantanida2016sentinel,
      url={https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/document-library/-/asset_publisher/1dO7RF5fJMbd/content/sentinel-1-level-1-detailed-algorithm-definition},
      title={Sentinel-1 level 1 detailed algorithm definition},
      author={Piantanida, Riccardo and Hajduch, G and Poullaouec, J},
      journal={ESA, techreport SEN-TN-52-7445},
      year={2016}
    }
    """
    data = convert_sentinel_1_image_to_tensor(data, product_type)    
    if product_type == "SLC":
        data += 0.001 #epsilon
        data = data.real**2 + data.imag**2
        data = 10 * torch.log10(data)
        return data.numpy()

    data = 10 * torch.log10(data)
    return data.numpy()

def is_SLC(image_filepath: str) -> Union[bool, None]:
    Dataset = gdal.Open(image_filepath)
    description = Dataset.GetMetadata().get('TIFFTAG_IMAGEDESCRIPTION')
    if description == None:
        raise ValueError(
            f"cannot determine the product_type of the Sentinel-1 "
            f"product: {image_filepath}."
        )
    elif '(SLC)' in description:
        return True
    else:
        return False

def is_GRD(image_filepath: str) -> Union[bool, None]:
    Dataset = gdal.Open(image_filepath)
    description = Dataset.GetMetadata().get('TIFFTAG_IMAGEDESCRIPTION')
    if description == None:
        raise ValueError(
            f"cannot determine the product_type of the Sentinel-1 "
            f"product: {image_filepath}."
        )
    elif '(GRD)' in description:
        return True
    else:
        return False

def load_and_scale_sentinel_1_image(image_filepath: str) -> Tuple[
        np.ndarray, np.ndarray]:
    data, mask, nodata_value, data_band_data_typ_enum = load_GeoTiff(
        image_filepath
    )
    if is_SLC(image_filepath):
        product_type = "SLC"
    elif is_GRD(image_filepath):
        product_type = "GRD"

    return scale_sentinel_1_image(data, product_type), mask

class SARFish_Plot(scene.SceneCanvas):
    def __init__(
            self, data: np.ndarray, mask: np.ndarray, 
            title: str = "SARFish product", show: bool = True, 
            keys: str = 'interactive', *args, **kwargs,
        ):
        scene.SceneCanvas.__init__(
            self, title = title, show = show, keys = keys, *args, **kwargs
        )
        self.unfreeze()
        self.size = (3840, 2160) # 4K vispy does coordinates in x, y order
        self.view = self.central_widget.add_view()
        self.data_shape = data.shape
        self.data_clim = (data.min(), data.max())
        self.colorbar_size = (self.data_shape[0], self.data_shape[1] // 20)
        self.marker_size =  self.size[1] // 75
        self.font_size = 20
        self.label_count = 0
        self.color_map = vispy.color.colormap.MatplotlibColormap('Blues_r')
        
        data = scene.visuals.Image(data, cmap = self.color_map)
        data.parent = self.view.scene

        mask = scene.visuals.Image(mask, cmap = 'grays')
        mask.set_gl_state('additive', depth_test=False)
        mask.parent = self.view.scene

        colorbar = scene.visuals.ColorBar(
            cmap = self.color_map,
            orientation = 'right', size = self.colorbar_size, 
            pos = (
                1.025 * self.data_shape[1] + self.colorbar_size[1] // 2, 
                self.data_shape[0] // 2
            ), 
            label = 'dB', label_color = 'white', clim = self.data_clim
        )
        colorbar.parent = self.view.scene
    
        legend_title = scene.visuals.Text(
            'Legend:', pos = (-0.025 * self.data_shape[1], 0), anchor_x = 'right',
            color = 'white', font_size = self.font_size, parent = self.view
        )
        legend_title.parent = self.view.scene

        self.label_objects = []

        self.view.camera = scene.PanZoomCamera(aspect = 1)
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()
        self.view.camera.zoom(1.2)
        self.transform = self.scene.node_transform(self.view.scene)
        self.freeze()

    def get_bbox_corners(self, bound: pd.DataFrame):
        x = np.meshgrid(
            bound[['left', 'right']].to_numpy(), 
            bound[['bottom', 'top']].to_numpy()
        )
        vertices = np.stack(x, axis = 2).reshape(-1, 2)
        return vertices

    def add_bboxes(
            self, bounds: pd.DataFrame, 
            color: Union[vispy.color.Color, vispy.color.ColorArray] = 'green' 
        ): 
        self.label_count += 1
        bounds = bounds.dropna()
        if len(bounds) == 0:
            return

        bbox_corners = bounds.apply(self.get_bbox_corners, axis = 1)
        bbox_corners = np.vstack(tuple(bbox_corners)) + 0.5 # pixel index -> centres
        bboxes = scene.visuals.Markers(
            pos = bbox_corners, edge_color = 'black', 
            face_color = color, size = 1, edge_width = 0.1, 
            symbol = 'cross', scaling = True, name = "bounding boxes", 
        )
        bboxes.set_gl_state(depth_test=False)
        bboxes.parent = self.view.scene

        bboxes_legend = scene.visuals.Text(
            "bounding boxes", pos = (
                -0.025 * self.data_shape[1], 
                0.05 * self.data_shape[0] * self.label_count
            ),
            anchor_x = 'right', color = color, font_size = 20
        )
        bboxes_legend.parent = self.view.scene

    def add_labels(
            self, columns: Union[pd.Series, np.ndarray], 
            rows: [pd.Series, np.ndarray], legend_label: str, 
            categories: Union[pd.Series, pd.DataFrame] = None, *args, **kwargs
        ): 
        kwargs.setdefault("color", "darkorange")
        kwargs.setdefault("marker_size", self.marker_size)
        kwargs.setdefault("font_size", self.font_size)
        kwargs.setdefault("symbol", "square")

        color = kwargs.get("color")
        marker_size = kwargs.get("marker_size")
        font_size = kwargs.get("font_size")
        symbol = kwargs.get("symbol")
        
        self.unfreeze()
        if not (isinstance(columns, pd.Series) or isinstance(columns, np.ndarray)):
            columns = np.array([columns])

        if not (isinstance(rows, pd.Series) or isinstance(rows, np.ndarray)):
            rows = np.array([rows])

        label_object = Labels(
            columns, rows, legend_label, categories, 
            marker_size = marker_size, color = color, font_size = font_size, 
            symbol = symbol
        )
        label_object.interactive = True
        label_object.parent = self.view.scene
        self.label_objects.append(label_object)

        self.label_count += 1
        if legend_label is None:
            legend_label = f"label_{label_count}"

        labels_legend = scene.visuals.Text(
            legend_label, pos = (
                -0.025 * self.data_shape[1], 
                0.05 * self.data_shape[0] * self.label_count
            ), anchor_x = 'right', color = color, font_size = font_size, 
        )
        labels_legend.parent = self.view.scene

        if categories is not None:
            self.categories = categories
        
        self.freeze()

    def scale_factor(self): 
        return self.transform.map([1, 1])[0] - self.transform.map([0, 0])[0] 

    def label_click_radius(self):
        return (5 * max(5 * self.scale_factor(), 1))

    def translate_label_categories(self):
        label_translation = 3 * self.label_click_radius()
        for label_object in self.label_objects:
            for child in label_object.children:
                child.transform = visuals.transforms.linear.STTransform(
                    translate = [label_translation, 0, 0]
                )

    def clear_label_categories(self):    
        for label_object in self.label_objects:
            for child in label_object.children:
                child.parent = None 

    def on_mouse_wheel(self, event: app.canvas.MouseEvent):
        if len(self.label_objects) == 0:
            return
        
        self.translate_label_categories()
    
    def on_mouse_press(self, event: app.canvas.MouseEvent):
        # https://github.com/vispy/vispy/blob/main/vispy/app/canvas.py#L453
        if len(self.label_objects) == 0:
            return

        if event.button != 1: # left click
            return

        visuals_under_cursor = self.visuals_at(event.pos, radius = 100)
        if len(visuals_under_cursor) == 1: #the viewbox will always be under cursor
            self.clear_label_categories()
            return
        
        for obj in visuals_under_cursor:
            if not isinstance(obj, Labels):
                visuals_under_cursor.remove(obj)

        selected_label = visuals_under_cursor[0]
        if selected_label.children is not None: 
            for child in selected_label.children:
                child.parent = None

        mouse_pos = self.transform.map(event.pos)[:2]
        closest_label_index = selected_label.get_closest_label_index_to_mousepress(
            mouse_pos, self.label_click_radius()
        )
        if closest_label_index is None:
            self.clear_label_categories()
            return
        
        selected_label.add_categories(closest_label_index)
        selected_label.category_info.transform = visuals.transforms.linear.STTransform(
            translate = [3 * self.label_click_radius(), 0, 0]
        )

    def run(self):
        app.run()

class Labels(scene.visuals.Compound):
    def __init__(
            self, columns: pd.Series, rows: pd.Series, label: str = "label", 
            categories: Union[pd.Series, pd.DataFrame] = None,
            *args, **kwargs,
        ): 
        self.unfreeze()
        self.color = kwargs.get("color")
        self.marker_size = kwargs.get("marker_size")
        self.font_size = kwargs.get("font_size")
        symbol = kwargs.get("symbol")
        self.type_face = 'DejaVu Sans Mono'
        self.columns = columns
        self.rows = rows
        self.categories = categories
        if categories is not None:
            self.max_category_name_length = max(
                len(column) for column in self.categories.columns.to_list()
            )

        column_centers = self.columns + 0.5 # pixel index -> centres:
        row_centers = self.rows + 0.5 

        self.labels = scene.visuals.Markers(
            pos = np.array([column_centers, row_centers]).T,
            size = self.marker_size, edge_width = 3, edge_color = self.color,
            face_color = (1,1,1,0.0), symbol = symbol, scaling = False,
        )
        self.labels.set_gl_state(depth_test=False)

        scene.visuals.Compound.__init__(self, [self.labels])
        self.freeze()

    def get_closest_label_index_to_mousepress(
            self, mouse_pos: np.ndarray, mouse_click_distance_tolerance: float
        ):
        self.unfreeze()
        self.label_pos = np.array([self.columns, self.rows]).T
        self.freeze()
        pairwise_distance_matrix = distance_matrix(
            self.label_pos, np.array(mouse_pos)[None, :], p = 2
        )
        assigned_label_index, _ = linear_sum_assignment(pairwise_distance_matrix) 
        assignments_under_threshold = (
            pairwise_distance_matrix[assigned_label_index] < 
            mouse_click_distance_tolerance
        ).squeeze()
        assigned_label_index = (
            assigned_label_index[assignments_under_threshold]
        ).squeeze()
        if assigned_label_index.size == 0:
            return
        
        return assigned_label_index

    def add_categories(self, closest_label_index: int):
        categories = self.categories.iloc[closest_label_index]
        category_information = ""
        for key, value in categories.items():
            category_information += (
                f"{key}:".ljust(self.max_category_name_length + 2) + f"{value}\n"
            )
        
        closest_label_pos = self.label_pos[closest_label_index]
        category_information += f"(x,y): {tuple(closest_label_pos)}\n"
        self.unfreeze()
        self.category_info = scene.visuals.Text(
            category_information, color = self.color, face = self.type_face, 
            font_size = self.font_size,
            pos = closest_label_pos, anchor_x = "left", 
            name = "category_info"
        )
        self.category_info.parent = self
        self.freeze()

def visualise_labels():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--xview3_slc_grd_correspondences", 
        help = (
            "/path/to/file/encoding/correspondence/between/xView3/and/"
            "SARFish/SLC/GRD.csv"
        ), type = str, required = True
    )
    parser.add_argument("-i", "--image_filepath", required=True, type=str, 
        help=f"path/to/input_data/"
    )
    parser.add_argument("-l", "--labels", required=True, type=str, 
        help=f"path/to/labels.csv"
    )
    parser.add_argument("-k", "--labels_column", required=True, type=str, 
        help=f"key for measurement/image name column in labels.csv"
    )
    parser.add_argument("-v", "--image_title", required=True, type=str, 
        help=f"value for measurement/image name in labels.csv"
    )
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    xView3_SLC_GRD_correspondences = pd.read_csv(
        args.xview3_slc_grd_correspondences
    )
    labels = labels[labels[args.labels_column] == args.image_title]
    xView3_SLC_GRD_correspondences = xView3_SLC_GRD_correspondences[
        xView3_SLC_GRD_correspondences[args.labels_column] == args.image_title
    ]
    if is_SLC(args.image_filepath): 
        swath_mask = labels['swath_index'].apply(
            lambda x: int(Path(args.image_filepath).stem[6]) == x
        )
        labels = labels[swath_mask]
    
    data, mask = load_and_scale_sentinel_1_image(str(args.image_filepath))
    SARFish = SARFish_Plot(data, mask, title = f"labels in {args.image_title}")
    SARFish.add_bboxes(
        bounds = labels[['left', 'right', 'bottom', 'top']]
    )
    SARFish.add_labels(
        columns = labels['detect_scene_column'], 
        rows = labels['detect_scene_row'], 
        categories = labels[
            ['is_vessel', 'is_fishing', 'vessel_length_m', 'confidence']
        ], legend_label = "groundtruth", color = "yellow"
    )
    app.run()

if __name__ == "__main__":
    visualise_labels()
