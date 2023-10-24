License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

# SARFish challenge repository

SARFish [^1] is an imagery dataset for the purpose of training, validating and testing supervised machine learning models on the task of ship detection and classification. SARFish builds on the excellent work of the [xView3-SAR dataset](https://iuu.xview.us/dataset) by expanding the imagery data to include [Single Look Complex (SLC)](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/single-look-complex) as well as [Ground Range Detected (GRD)](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/ground-range-detected) imagery data taken directly from the European Space Agency 
(ESA) Copernicus Programme [Open Access Hub Website](https://scihub.copernicus.eu/).

## How to use this repo

### 1. SARFish\_Terms\_and\_Conditions.md

Read the terms and conditions for the:

- use of the SARFish dataset
- use of this repo
- participation in the SARFish challenge.

### 2. venv\_setup/venv\_create.sh

Create the virtual environment with necessary packages.

**Note: Requires >python3.8**

```bash
$ ./venv_setup/venv_create.sh -v venv -r ./venv_setup/venv_requirement.txt
$ source ./venv/bin/activate
```

Edit the file reference/environment.yaml to set path to the root directory of the SARFish dataset:

```yaml
SARFish_root_directory: /path/to/SARFish/root/ 
```

### 3. Download the data

The SARFish dataset is available for download at:

[full SARFish dataset](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish)
[sample SARFish dataset](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFishSample)

| dataset       | coincident GRD, SLC products | compressed (GB) | uncompressed (GB) |
| ------------- | ---------------------------- | --------------- | ----------------- |
| SARFishSample | 1                            | 4.3             | 8.2               |
| SARFish       | 753                          | 3293            | 6468              |

#### Full SARFish dataset

Make sure you have at least enough storage space for the uncompressed dataset.

```bash
cd /path/to/large/storage/location
```

[Create|login] to a [huggingface](https://huggingface.co) account. 

In your python3 virtual environment login to the huggingface command line interface.

```bash
huggingface-cli login
```

Copy the access token in settings/Access Tokens from your huggingface account. Clone the dataset

```bash
git lfs install
git clone https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish
```

#### SARFish sample dataset

Substitute the final command for the full dataset with the following:

```bash
git clone https://huggingface.co/datasets/ConnorLuckettDSTG/SARFishSample
```

### 4. check\_SARFish\_md5sum.py

Check the md5 sums of the downloaded SARFish products

```bash
./check_SARFish_md5sum.py
```

### 5. unzip\_batch.sh

Unzip SARFish data products in parallel.

```bash
cd /path/to/SARFish/directory/GRD
unzip\_batch.sh -p $(find './' -type f -name "*.SAFE.zip")

cd /path/to/SARFish/directory/SLC
unzip\_batch.sh -p $(find './' -type f -name "*.SAFE.zip")
```

### 6. Download the SARFish labels

Download the training and validation label files for both the GRD and SLC products from **RITWIK: INSERT LABEL DOWNLOAD LOCATION**

Add the label files to their respective partitions in the dataset file structure:

```bash
SARFish/
├── GRD
│   ├── public
│   ├── train
│   │   └── GRD_train.csv
│   └── validation
│       └── GRD_validation.csv
└── SLC
    ├── public
    ├── train
    │   └── SLC_train.csv
    └── validation
        └── SLC_validation.csv
```

### 7. reference/SARFish\_demo.ipynb

Run SARFish\_demo.ipynb:

```bash
python3 -m jupyter notebook reference/SARFish_demo.ipynb
```
The SARFish demo is jupyter notebook to help users get accustomed to:

- What the SARFish dataset is
- How to access the SARFish dataset
- Dataset structure
- How to load and visualise the SARFish imagery data
- How to load and visualise the SARFish groundtruth labels
- SARFish challenge prediction submission format
- How to evaluate model performance using the SARFish metric
- How to participate in the SARFish challenge

### 8. reference/SARFish\_reference.py

Run a baseline detector and classifier on the SARFish SLC data.

```bash
./reference/SARFish_reference.py
```

### 9. reference/SARFish\_metric.py

Evaluate a model's performance using the metrics for the SARFish dataset.

```bash
./reference/SARFish_metric.py \
    -p ./reference/labels/reference_GRD_predictions.csv \
    -g /path/to/SARFish/root/GRD/validation/GRD_validation.csv \
    --sarfish_root_directory /path/to/SARFish/root/ \
    --product_type GRD \
    --xview3_slc_grd_correspondences ./reference/labels/xView3_SLC_GRD_correspondences.csv \
    --shore_type xView3_shoreline \
    --drop_low_detect \
    --costly_dist \
    --evaluation_mode
```

[^1] T.-T. Cao et al., “SARFish: Space-Based Maritime Surveillance Using Complex Synthetic Aperture Radar Imagery,” in 2022 International Conference on Digital Image Computing: Techniques and Applications (DICTA), 2022, pp. 1–8. 

