License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

# SARFish challenge repository

SARFish [1] is an imagery dataset for the purpose of training, validating and testing supervised machine learning models on the task of ship detection and classification. SARFish builds on the excellent work of the [xView3-SAR dataset](https://iuu.xview.us/dataset) by expanding the imagery data to include [Single Look Complex (SLC)](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/single-look-complex) as well as [Ground Range Detected (GRD)](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/ground-range-detected) imagery data taken directly from the European Space Agency 
(ESA) Copernicus Programme [Open Access Hub Website](https://scihub.copernicus.eu/).

Links:
- Data:
    - [SARFish](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish)
    - [SARFishSample](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFishSample)
- [Labels](https://iuu.xview.us/download-links)
- Challenge:
    - [Maritime Object Detection Track](https://www.kaggle.com/competitions/sarfish-maritime-object-detection) 
    - [Maritime Object Prediction Track](https://www.kaggle.com/competitions/sarfish-maritime-object-classification)
    - [Vessel Length Regression Track](https://www.kaggle.com/competitions/sarfish-vessel-length-regression)
- [GitHub repo](https://github.com/RitwikGupta/SARFish)
- [Mailbox](SARFish.Dataset@defence.gov.au)
- [DAIRNet](https://www.dairnet.com.au/events/workshop-on-complex-valued-deep-learning-and-sarfish-challenge/)

## How to use this repo

### 1. SARFish\_Terms\_and\_Conditions.md

Read the terms and conditions for the:

- use of the SARFish dataset
- use of this repo
- participation in the SARFish challenge.

### 2. Install gdal

```bash
$ <package-manager> install g++ python-devel python3-devel gdal gdal-devel 
```

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

Install git lfs

```bash
<package-manager> install git-lfs
git lfs install
```

Copy the access token in settings/Access Tokens from your huggingface account. Clone the dataset


```bash
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

Unzip SARFish data products.

```bash
cd /path/to/SARFish/directory/GRD
unzip\_batch.sh -p $(find './' -type f -name "*.SAFE.zip")

cd /path/to/SARFish/directory/SLC
unzip\_batch.sh -p $(find './' -type f -name "*.SAFE.zip")
```

### 6. Download the SARFish labels

Download the training and validation label files for both the GRD and SLC products from the [xView3 website](https://iuu.xview.us/download-links)

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

### 7. Get started with the SARFish dataset and challenge: Run the SARFish_demo.ipynb notebook

```bash
python3 -m jupyter notebook reference/SARFish_demo.ipynb
```
The SARFish demo is jupyter notebook to help users understand:

- What is the SARFish Challenge?
- What is the SARFish dataset?
- How to access the SARFish dataset
- Dataset structure
- How to load and visualise the SARFish imagery data
- How to load and visualise the SARFish groundtruth labels
- How to train, validate and test the reference/baseline model
- SARFish challenge prediction submission format
- How to evaluate model performance using the SARFish metric

### 8. Train and evaluate the baseline reference model

A baseline reference implementation of a real-valued deep learning model is provided for the purpose of introducing new users to training and validating, testing models on the SARFish SLC data in addition to illustrating the use of the SARFish metrics. The reference model demonstrates how to use the SARFish metrics during training, testing and evaluation to help inform the development of better performing models.

The baseline uses the predefined PyTorch implementation of FCOS; chosen because it uses the concept of “centre-ness”, which we believe is applicable to the maritime objects in this dataset.

```bash
SARModel.py
```

The baseline can be trained and evaluated by sequentially running the following scripts:

1_create_tile.py generates the tiles used for training the baseline. Approximately 300GB is required for storage.

```bash
./1_create_tile.py
```

The following trains, validates and tests the baseline model n a small subset of the SARFish dataset detailed in fold.csv.

```bash
./2_train.py
./3_test.py
```

4_evaluate.py calls the SARFish_metric.py script on the testing scenes to determine model peformance on the SARFish challenge 
tasks.

```bash
./4_evaluate.py
```

The following scripts call the model over the entire public partition of the SARFish dataset to generate the submission/predictions uploaded to the Kaggle competition as the benchmark.

```bash
./5_inference.py
./6_concatenate_scene_predictions.py
```

### 9. reference/SARFish\_metric.py

Evaluate the baseline model's performance on a scene from the validation partition using the metrics for the SARFish dataset.

```bash
./SARFish_metric.py \
    -p labels/reference_model/reference_predictions_SLC_validation_S1B_IW_SLC__1SDV_20200803T075720_20200803T075748_022756_02B2FF_E5D2.csv \
    -g /path/to/SARFish/root/SLC/validation/SLC_validation.csv \
    --sarfish_root_directory /path/to/SARFish/root/ \
    --product_type SLC \
    --xview3_slc_grd_correspondences labels/xView3_SLC_GRD_correspondences.csv \
    --shore_type xView3_shoreline \
    --no-evaluation-mode
```

[1] T.-T. Cao et al., “SARFish: Space-Based Maritime Surveillance Using Complex Synthetic Aperture Radar Imagery,” in 2022 International Conference on Digital Image Computing: Techniques and Applications (DICTA), 2022, pp. 1–8. 
