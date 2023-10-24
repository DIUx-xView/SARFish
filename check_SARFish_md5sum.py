#!/usr/bin/env python3

"""Check the md5sum of the zipped SARFish dataset products.

! ./check_SARFish_md5sum.py
"""

from hashlib import md5
from pathlib import Path
from typing import Union, Tuple, List
import yaml

import pandas as pd

def get_md5sum(scene_id: str, file_path: Path) -> Tuple[str, str]:
    with open(file_path, "rb") as f:
        file_md5sum = md5(f.read()).hexdigest() 
        print(f"scene_id: {scene_id}".ljust(30) + f"file_md5sum: {file_md5sum}")
    return scene_id, file_md5sum

def map_SARFish_md5sum(
        df: pd.DataFrame, SARFish_root_directory: Path
    ) -> pd.DataFrame:
    print(f"checking md5sums:")
    for product_type in ["GRD", "SLC"]:
        print(f"\nproduct type: {product_type}")
        df.loc[:, f'{product_type}_file_path'] = df.apply(
            lambda x: Path(
                SARFish_root_directory, f"{product_type}", x['DATA_PARTITION'], 
                f"{x[f'{product_type}_product_identifier']}.SAFE.zip"
            ), axis = 1
        )
        df.loc[:, f'does_{product_type}_exist'] = (
            df[f'{product_type}_file_path'].apply(lambda x: x.is_file())
        )
        df.loc[:, f'computed_{product_type}_md5sum'] = None
        if df.loc[:, f'does_{product_type}_exist'].sum() == 0: 
            continue

        mapped_md5sum = dict(
            df.loc[df[f'does_{product_type}_exist']].apply(
                lambda x: get_md5sum(x['scene_id'], x[f'{product_type}_file_path']),
                axis = 1
            ).to_list()
        )
        df.loc[:, f'computed_{product_type}_md5sum'] = df['scene_id'].map(mapped_md5sum)

    return df

def check_SARFish_md5sum(
        df: pd.DataFrame, SARFish_root_directory: Path
    ):
    df = map_SARFish_md5sum(df, SARFish_root_directory)
    for product_type in ["GRD", "SLC"]:
        md5sum_matches = (
            df[f'computed_{product_type}_md5sum'] == 
            df[f'{product_type}_md5sum']
        )
        print(f"\n{product_type} products matching md5sum:")
        df.loc[df[f'does_{product_type}_exist'] & md5sum_matches].apply(
            lambda x: print(x[f'{product_type}_file_path']), 
            axis = 1
        )
        print(f"\n{product_type} products NOT matching md5sum:")
        df.loc[df[f'does_{product_type}_exist'] & ~md5sum_matches].apply(
            lambda x: print(x[f'{product_type}_file_path']), 
            axis = 1
        )
        print(f"\n{product_type} products NOT downloaded:\n")
        df.loc[~df[f'does_{product_type}_exist']].apply(
            lambda x: print(x[f'{product_type}_file_path']), 
            axis = 1
        )

def main():
    environment_path = Path("reference/environment.yaml")
    with open(str(environment_path), "r") as f:
        config = yaml.safe_load(f)

    SARFish_root_directory = Path(config["SARFish_root_directory"])
    xView3_SLC_GRD_correspondences_path = Path(
        "reference", config["xView3_SLC_GRD_correspondences_path"]
    )
    xView3_SLC_GRD_correspondences = pd.read_csv(
        str(xView3_SLC_GRD_correspondences_path)
    )
    check_SARFish_md5sum(
        xView3_SLC_GRD_correspondences.copy(deep = True), 
        SARFish_root_directory
    )
    
if __name__ == "__main__":
    main()
