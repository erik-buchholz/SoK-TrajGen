# Dataset Description

This directory contains the data used for our evaluation.
The data is described below.

## Table Of Contents

<!-- TOC -->
* [Dataset Description](#dataset-description)
  * [Table Of Contents](#table-of-contents)
  * [Foursquare New York City (FS NYC)](#foursquare-new-york-city-fs-nyc)
    * [Verification](#verification)
  * [GeoLife](#geolife)
    * [Pre-Processing](#pre-processing)
    * [Verification](#verification-1)
<!-- TOC -->

## Foursquare New York City (FS NYC)

The files `test_latlon.csv` and `train_latlon.csv` are taken from the LSTM-TrajGAN repository
available at [GitHub](https://github.com/GeoDS/LSTM-TrajGAN).
The file `all_latlon.csv` is the concatenation of the two files above and contains all trajectories.
All three files are already contained in this repository.

| File               | Number of Trajectories | Number of Points | Source                                                                                  |
|--------------------|------------------------|------------------|-----------------------------------------------------------------------------------------|
| `test_latlon.csv`  | 1,027                  | 22,153           | [LSTM-TrajGAN](https://github.com/GeoDS/LSTM-TrajGAN/blob/master/data/test_latlon.csv)  |
| `train_latlon.csv` | 2,052                  | 44,809           | [LSTM-TrajGAN](https://github.com/GeoDS/LSTM-TrajGAN/blob/master/data/train_latlon.csv) |
| `all_latlon.csv`   | 3,079                  | 66,962           | /                                                                                       |

The FourSquare NYC dataset does not require further pre-processing and can be used as is.

### Verification

To verify the integrity of the data, you can run the following command (from the repository's root directory):

```shell
python -m unittest test/test_fs_nyc.py
```

## GeoLife

**If you simply want to reproduce the paper's results, you can use the pre-processed version of the dataset we included in `data/geolife_FIFTH-RING_5_60_200_TRUNCATE`.
In this case, you don't have to download the raw dataset and can skip the pre-processing step.**

### Download of Raw Dataset

The GeoLife dataset is available
at [Microsoft Research](https://www.microsoft.com/en-us/download/details.aspx?id=52367).
Due to its size, we do not include into this repository.
Please download it manually from the provided link and extract it into the directory `data/geolife`.

```shell
cd data
wget https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip
unzip Geolife\ Trajectories\ 1.3.zip
mv Geolife\ Trajectories\ 1.3 geolife
```

The resulting directory structure should be as follows:

```
data/
├──geolife/
│  ├──Data/
│  │  ├──000/
│  │  │  ├──Trajectory/
│  │  │  │  ├──20081023025304.plt
│  │  │  │  ├──...
│  │  ├──...
│  ├──User Guide-1.3.pdf
```

### Pre-Processing

The GeoLife dataset requires some pre-processing before it can be used.
For pre-processing, you can use the script `stg.datasets.geolife.py`.

To display all available options, run (from the repository's root directory):

```shell
python -m stg.datasets.geolife --help
```

The script performs the pre-processing used for the evaluation in the paper by default:

```shell
python -m stg.datasets.geolife
```

If the default pre-processing is used, the resulting dataset will be stored
in `data/geolife_FIFTH-RING_5_60_200_TRUNCATE`.

### Verification

To verify the correctness of the pre-processing, you can run the following command:

```shell
FORCE_TESTS=true RUN_SLOW_TESTS=true python -m unittest test/test_geolife.py
```
