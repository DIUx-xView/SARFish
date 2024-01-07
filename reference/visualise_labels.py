#!/usr/bin/env python3

"""Visualisation tools for plotting SARFish imagery and associated labels and 
predictions.
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
        data: np.ndarray, nodata_mask: np.ndarray, product_type: {"GRD", "SLC"}
    ) -> np.ndarray:
    """
    Transforms SARFish sentinel-1 GRD and SLC products for viewing. Applies 
    log10 decibel scaling to GRD and SLC imagery, and sets nodata values to 0. 
    In addition, SLC image is also "detected" [1] Page 103 - Section 7-2, step 12.

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
    # nodata_mask is defined in line with np.ma.masked_array, 
    # nodata_mask == True where data is invalid
    output = np.zeros_like(data, dtype = np.float32)
    valid_data = convert_sentinel_1_image_to_tensor(
        data[~nodata_mask], product_type
    )
    if product_type == "SLC":
        valid_data += 0.1 #epsilon
        valid_data = np.abs(valid_data)**2
        scaled_valid_data = 10 * torch.log10(valid_data)
        output[~nodata_mask] = scaled_valid_data.numpy()
        return output

    scaled_valid_data = 10 * torch.log10(valid_data)
    output[~nodata_mask] = scaled_valid_data.numpy()
    return output 

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
    data, nodata_mask, nodata_value, data_band_data_typ_enum = load_GeoTiff(
        image_filepath
    )
    if is_SLC(image_filepath):
        product_type = "SLC"
    elif is_GRD(image_filepath):
        product_type = "GRD"

    return scale_sentinel_1_image(data, nodata_mask, product_type), nodata_mask

class SARFish_Plot(scene.SceneCanvas):
    def __init__(
            self, data: np.ndarray, nodata_mask: np.ndarray, 
            title: str = "SARFish product", show: bool = True, 
            keys: str = 'interactive', cmap = 'Greys_r', 
            clim: Tuple[int] = None, **kwargs,
        ):
        scene.SceneCanvas.__init__(
            self, title = title, show = show, keys = keys, **kwargs
        )
        self.unfreeze()
        self.size = (3840, 2160) # 4K vispy does coordinates in x, y order
        #self.size = tuple(elem/2 for elem in self.size)
        self.view = self.central_widget.add_view()
        self.data_shape = data.shape
        self.cmap = cmap
        self.clim = clim
        if self.clim is None:
            self.clim = (data.min(), data.max())

        self.colorbar_size = (self.data_shape[0], self.data_shape[1] // 20)
        self.font_size = 20
        self.label_count = 0
        
        data_image = scene.visuals.Image(
            data, cmap = self.cmap, clim = self.clim
        )
        data_image.parent = self.view.scene

        nodata_mask = nodata_mask.astype(np.uint8)
        nodata_mask_image = scene.visuals.Image(nodata_mask, cmap = 'grays')
        nodata_mask_image.set_gl_state('additive', depth_test=False)
        nodata_mask_image.parent = self.view.scene

        colorbar = scene.visuals.ColorBar(
            cmap = self.cmap,
            orientation = 'right', size = self.colorbar_size, 
            pos = (
                1.025 * self.data_shape[1] + self.colorbar_size[1] // 2, 
                self.data_shape[0] // 2
            ), 
            label = 'dB', label_color = 'white', clim = self.clim
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

        self.mouse_pos = None
        self.mouse_pixel_pos = None
        self.mouse_position_text = scene.visuals.Text(
            f'(x, y): ', pos = self.transform.map([0, self.font_size])[:2],
            anchor_x = 'left', color = 'orange', font_size = self.font_size
        )
        self.mouse_position_text.parent = self.view.scene

        self.freeze()

    def get_bbox_corners(self, bound: pd.DataFrame):
        x = np.meshgrid(
            bound[['left', 'right']].to_numpy(), 
            bound[['bottom', 'top']].to_numpy()
        )
        vertices = np.stack(x, axis = 2).reshape(-1, 2)
        return vertices

    def add_bboxes(self, bounds: pd.DataFrame, color: str = 'green'): 
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
            anchor_x = 'right', color = color, font_size = self.font_size
        )
        bboxes_legend.parent = self.view.scene

    def add_labels(
            self, columns: Union[pd.Series, np.ndarray], 
            rows: [pd.Series, np.ndarray], legend_label: str, 
            categories: Union[pd.Series, pd.DataFrame] = None, 
            color: str = "yellow", marker_size = None, symbol = "square", 
            font_size = None, type_face: str = 'DejaVu Sans Mono', 
            **kwargs,
        ): 
        if not (isinstance(columns, pd.Series) or isinstance(columns, np.ndarray)):
            columns = np.array([columns])

        if not (isinstance(rows, pd.Series) or isinstance(rows, np.ndarray)):
            rows = np.array([rows])
    
        if marker_size is None:
            marker_size =  self.size[1] // 75

        if font_size is None:
            font_size = self.font_size

        self.unfreeze()
        label_object = Labels(
            columns, rows, legend_label, categories, color, marker_size, symbol, 
            font_size, type_face, **kwargs
        )
        label_object.interactive = True
        label_object.parent = self.view.scene
        self.label_objects.append(label_object)

        self.label_count += 1
        if legend_label is None:
            legend_label = f"label_{label_count}"

        labels_legend = scene.visuals.Text(
            text = legend_label, color = color, pos = (
                -0.025 * self.data_shape[1], 
                0.05 * self.data_shape[0] * self.label_count
            ), anchor_x = 'right', face = type_face, font_size = 20, 
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
        self.mouse_position_text.pos = self.transform.map([0, self.font_size])[:2]
        if len(self.label_objects) == 0:
            return
        
        self.translate_label_categories()

    def on_mouse_move(self, event: app.canvas.MouseEvent):
        self.mouse_pos = self.transform.map(event.pos)[:2]
        previous_mouse_pixel_pos = self.mouse_pixel_pos
        self.mouse_pixel_pos = np.floor(self.mouse_pos)
        if previous_mouse_pixel_pos is None:
            return

        if (
            (previous_mouse_pixel_pos == self.mouse_pixel_pos).all() and
            (event.button is None)
        ):
            return

        self.mouse_position_text.pos = self.transform.map([0, self.font_size])[:2]
        if (
            not (
                (0 <= self.mouse_pixel_pos[0] <= self.data_shape[1]) and
                (0 <= self.mouse_pixel_pos[1] <= self.data_shape[0])
            ) and (self.mouse_position_text is not None)
        ):
            self.mouse_position_text.text = None
            return

        self.mouse_position_text.text = (
            f'(x, y): {self.mouse_pixel_pos[0], self.mouse_pixel_pos[1]}'
        )
    
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

        closest_label_index = selected_label.get_closest_label_index_to_mousepress(
            self.mouse_pos, self.label_click_radius()
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
            color: str = "yellow", marker_size = None, symbol = "square", 
            font_size = None, type_face: str = 'DejaVu Sans Mono', 
        ): 
        self.unfreeze()
        self.columns = columns
        column_centers = self.columns + 0.5 # pixel index -> centres:
        self.rows = rows
        row_centers = self.rows + 0.5 
        if categories is not None:
            self.max_category_name_length = max(
                len(column) for column in categories.columns.to_list()
            )

        self.categories = categories
        self.color = color
        self.marker_size = marker_size
        self.font_size = font_size
        self.type_face =  type_face
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
        self.unfreeze()
        self.category_info = scene.visuals.Text(
            category_information, color = self.color, face = self.type_face, 
            font_size = self.font_size, pos = closest_label_pos, 
            anchor_x = "left", name = "category_info"
        )
        self.category_info.parent = self
        self.freeze()
