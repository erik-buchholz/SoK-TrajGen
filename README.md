[![Unittests](../../actions/workflows/unittests.yml/badge.svg)](../../actions/workflows/unittests.yml)
[![GeoLife Pre-Processing](../../actions/workflows/geolife_preprocessing.yml/badge.svg)](../../actions/workflows/geolife_preprocessing.yml)

# SoK: Can Trajectory Generation Combine Privacy and Utility?

Artifacts for PETS'24 paper "SoK: Can Trajectory Generation Combine Privacy and Utility?".

## Table of Contents

<!-- TOC -->
* [SoK: Can Trajectory Generation Combine Privacy and Utility?](#sok-can-trajectory-generation-combine-privacy-and-utility)
  * [Table of Contents](#table-of-contents)
  * [Citation](#citation)
  * [Abstract](#abstract)
  * [Basic Requirements](#basic-requirements)
    * [Hardware Requirements](#hardware-requirements)
    * [Software Requirements](#software-requirements)
    * [Estimated Time and Storage Consumption](#estimated-time-and-storage-consumption)
    * [Storage Consumption](#storage-consumption)
    * [Runtime](#runtime)
  * [Setup](#setup)
    * [Environment](#environment)
      * [Pip Setup](#pip-setup)
      * [Conda Setup](#conda-setup)
    * [[Optional] Datasets](#optional-datasets)
      * [[Optional] Pre-Processing GeoLife](#optional-pre-processing-geolife)
      * [Verification](#verification)
  * [Tests](#tests)
  * [Usage](#usage)
    * [Figures](#figures)
      * [Port Forwarding / Remote access](#port-forwarding--remote-access)
      * [Notebooks](#notebooks)
    * [Model Training](#model-training)
      * [Runtimes and Examples](#runtimes-and-examples)
    * [LSTM-TrajGAN Experiments](#lstm-trajgan-experiments)
      * [1. LSTM-TrajGAN convergence.](#1-lstm-trajgan-convergence)
      * [2. LSTM-TrajGAN Loss Functions](#2-lstm-trajgan-loss-functions)
      * [3. LSTM-TrajGAN [5] vs RAoPT [7]](#3-lstm-trajgan-5-vs-raopt-7)
  * [Contact](#contact)
  * [Acknowledgements](#acknowledgements)
  * [References](#references)
  * [Licence](#licence)
<!-- TOC -->

## Citation

If you use the code in this repository, please cite the following paper:

```bibtex
@article{BSW+24,
  title={{SoK: Can Trajectory Generation Combine Privacy and Utility?}},
  author={Buchholz, Erik and Abuadbba, Alsharif and Wang, Shuo and Nepal, Surya and Kanhere, Salil S.},
  journal={Proceedings on Privacy Enhancing Technologies},
  month={July},
  year={2024},
  volume={2024},
  number={3},
  address={Bristol, UK}
}
```

## Abstract

This repository contains the artifacts for the paper "SoK: Can Trajectory Generation Combine Privacy and Utility?"
with the following abstract:

> While location trajectories represent a valuable data source for analyses and location-based services, they can reveal
> sensitive information, such as political and religious preferences.
> Differentially private publication mechanisms have been proposed to allow for analyses under rigorous privacy
> guarantees.
> However, the traditional protection schemes suffer from a limiting privacy-utility trade-off and are vulnerable to
> correlation and reconstruction attacks.
> Synthetic trajectory data generation and release represent a promising alternative to protection algorithms.
> While initial proposals achieve remarkable utility, they fail to provide rigorous privacy guarantees.
> This paper proposes a framework for designing a privacy-preserving trajectory publication approach by defining five
> design goals, particularly stressing the importance of choosing an appropriate Unit of Privacy.
> Based on this framework, we briefly discuss the existing trajectory protection approaches, emphasising their
> shortcomings.
> This work focuses on the systematisation of the state-of-the-art generative models for trajectories in the context of
> the proposed framework.
> We find that no existing solution satisfies all requirements.
> Thus, we perform an experimental study evaluating the applicability of six sequential generative models to the
> trajectory domain.
> Finally, we conclude that a generative trajectory model providing semantic guarantees remains an open research
> question and propose concrete next steps for future research.

## Basic Requirements

### Hardware Requirements

This repository does not require special hardware. However, **a GPU is highly recommended**,
and training on the CPU will lead to increased runtimes. When running the notebooks in **Google Colab**,
a runtime type with GPU **is recommended**. For example, verify the notebooks with the free **T4 GPU**.
This runtime will work for all notebooks.

### Software Requirements

The code can be executed on any UNIX-based OS, i.e., Linux or macOS. The following software packages have to be installed:

```shell
sudo apt install python3.10 python3.10-venv
```

### Estimated Time and Storage Consumption

### Storage Consumption

The initial git repository has a size of **2GB**.
If a virtual environment (python venv) is created, the combined space of the repository and venv is approximately
**8GB**.
To execute the provided code, no additional space is required, such that **10GB** storage should suffice.
If the models are trained and additional parameters are stored, more storage is required, depending on the frequency of storage.
Overall, **20GB** should be sufficient.

**TL;DR: 20GB of storage should suffice.**

### Runtime

The runtime of the scripts is heavily influenced by used the hardware. We provide details on the runtimes below for each script on the PETS Artifact Standard VM (4 cores, 8GB memory, 40GB disk, Ubuntu 22.4). Please note that this VM does not have a GPU. Training on a system with GPU will be significantly faster. Additionally, we provide the runtimes for notebooks on Google Colab with the
*T4 GPU* runtime.
We did not re-run the [LSTM-TrajGAN Measurement](#lstm-trajgan-experiments) in the VM due to their long run time, but state the runtimes measured on our private evaluation machine with the following specifications:
(2x Intel Xeon Silver 4208, 128GB RAM, Ubuntu 20.04.01 LTS, 4 NVIDIA Tesla T4 GPUs with 16GB RAM each).
Below, we use the following notation:

- **Standard VM:** (4 cores, 8GB memory, 40GB disk, Ubuntu 22.4)
- **Google Colab:** T4 GPU runtime
- **GPU Server:** (2x Intel Xeon Silver 4208, 128GB RAM, Ubuntu 20.04.01 LTS, 4 NVIDIA Tesla T4 GPUs with 16GB RAM each)

## Setup

If you want to get started quickly, you can also use the Google Colab notebooks linked in Section
[Figures](#figures).

### Environment
The environment can be set up using either Conda or pip.
We use unittests to verify that the environment has been set up correctly.

#### Pip Setup

[![Unittests](../../actions/workflows/unittests.yml/badge.svg)](../../actions/workflows/unittests.yml)

```shell
./setup.sh
```

This will create a pip venv in the directory `venv` and install the required packages.
To modify the environment name, edit the line `VENV_NAME="venv"` in `setup.sh`.
To activate the created environment in your current shell, run:

```shell
source venv/bin/activate  # Replace venv with your environment name
# source YOUR_ENV_NAME/bin/activate
```

#### Conda Setup

Install conda environment and activate in current shell:

```shell
./setup_conda.sh
# The following lines are necessary to activate the environment in the current shell
source ~/miniconda/bin/activate
conda init
# Activate the environment
conda activate stg
```

This will create a conda environment with the name `stg` and install the required packages. 
To modify the environment name, edit the line `VENV_NAME="stg"` in `setup_conda.sh`.
To activate the environment run (required after each new shell):

```shell
conda activate stg  # Replace stg with your environment name defined in setup_conda.sh
# conda activate YOUR_ENV_NAME
```

### [Optional] Datasets

The datasets used in this project and their setup are described in the [data/README.md](data/README.md).
We include both datasets in the pre-processed form used in the paper.

#### [Optional] Pre-Processing GeoLife

The GeoLife dataset requires some pre-processing before it can be used.
We included the pre-processed dataset in the repository, but the pre-processing can be repeated if desired.
For instance, it is possible to adjust parameters.
Details are provided in the [data/README.md](data/README.md).

The following is **optional**!
To download the dataset and perform the same pre-processing as used in the paper, run from the repository's root:
```shell
cd data
echo "Downloading GeoLife dataset"
wget https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip
unzip Geolife\ Trajectories\ 1.3.zip
mv Geolife\ Trajectories\ 1.3 geolife
echo "Pre-processing GeoLife dataset"
cd ..
python3 -m stg.datasets.geolife
echo "Verifying GeoLife dataset"
FORCE_TESTS=true RUN_SLOW_TESTS=true python3 -m unittest test.test_geolife
echo "Done"
```

If the default pre-processing is used, the resulting dataset will be stored
in `data/geolife_FIFTH-RING_5_60_200_TRUNCATE`.
In case of a custom pre-processing, the path will change accordingly.

#### Verification

For both the Foursquare NYC and GeoLife dataset, we provide unittests in `test/test_fs_nyc.py`
and `test/test_geolife.py`, respectively, to verify that the installation has been successful.
Because loading the GeoLife dataset is time-intensive, the unittests for the GeoLife dataset are skipped by default.
To include them, set the environment variable `RUN_SLOW_TESTS` to `true`.
If you performed the pre-processing yourself, you can also set `FORCE_TESTS` to `true` to force the tests to run.
However, if you only use the provided version, this will not be possible as some tests require the original dataset.

```shell
FORCE_TESTS=false RUN_SLOW_TESTS=true python3 -m unittest discover -s test
```

## Tests

The project contains some tests in the `test/` directory.
These are by no means exhaustive but can be used to test whether the setup was successful.
Run

```shell  
python3 -m unittest discover -s test  
```   

from the repository's root directory to run all tests. [**Runtime Standard VM:** ~9s]
By default, some slow tests testing the correct pre-processing of the GeoLife dataset are skipped. If you want to include these, run:

```shell  
RUN_SLOW_TESTS=true python3 -m unittest discover -s test  
``````

**Runtime Standard VM:** ~160s

## Usage

In the following, we describe how to run the code.
We start with the reproduction of the results shown in the paper and describe general usage afterward.
For the following to work, we assume that the Python environment has been set up as described in Section [Setup](#setup).
**Don't forget to activate your pip or conda environment before running the scripts! (See: [Setup](#setup))**

### Figures

The figures in the paper can be generated through the provided Jupyter notebooks.
The results will be written into `img/`.
All models will use the parameters derived from training on our machine, which are included in this repository in the `parameters/` directory.
If you want to train the models yourself, please refer to Section [Model Training](#model-training).
In that case, you have to change the paths in the notebooks to point to the newly trained model parameters.
You can use Google Colab to run the notebooks without setting up the environment on your machine.
If you want to run the notebooks on your machine, run the following command from the repository's root directory:

```shell
jupyter notebook --port 9999
```

Then, you can open the notebooks in your browser by navigating to `http://localhost:9999`.
See the [Official Jupyter Notebook Guide](https://docs.jupyter.org/en/latest/running.html) for more details.

#### Port Forwarding / Remote access

If you want to run Jupyter notebooks on a remote machine but access them in your local browser,
you can use SSH port forwarding as follows (`9999` can be replaced by an arbitrary port):

1. Start Jupyter Notebook on the remote machine:

  ```shell
  jupyter notebook --port 9999 --no-browser 
  ```

2. Forward the port to your local machine (to be executed on your *local* machine):

  ```shell
  ssh -NfL 9999:localhost:9999 USER@REMOTE
  ```

3. Open your browser and navigate to `http://localhost:9999`.

#### Notebooks

For the notebooks using GPU, we recommend using the "**T4 GPU**" runtime in Google Colab.
The notebooks can also be run on a CPU, but the training will be significantly slower.
Runtimes for GPU notebooks are provided based on the T4 GPU in Google Colab, and others on the default CPU.
The following notebooks are available:

**Paper Figures:**

- Figure 1: `notebooks/paper_figures.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/paper_figures.ipynb)
  - **Runtime Colab:** 01:00 min
- Figure 2: _LaTeX Figure - no code provided_
- Figure 3: `notebooks/geopointgan.ipynb`  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/geopointgan.ipynb) -
  **GPU Recommended**
  - **Runtime Colab:** 07:38 min
- Figure 4: `notebooks/overview_figures.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/overview_figures.ipynb) -
  **GPU Recommended**
  - **Runtime Colab:** 01:37 min
- Figure 5: `notebooks/overview_figures.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/overview_figures.ipynb) -
  **GPU Recommended**
  - **Runtime Colab:** 01:37 min

**Dataset Notebooks:**

- FourSquare NYC Dataset: `notebooks/foursquare_data.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/foursquare_data.ipynb)
  - **Runtime Colab:** 00:43 min
- GeoLife Dataset _explaining pre-processing choices in
  detail_: `notebooks/geolife.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/geolife.ipynb) -
  - Very resource-intensive. Will takes >10 hours in Colab. We recommend using the notebook only for the explanations, but to run the script `python3 -m stg.datasets.geolife` for the actual preprocessing.

**Model Notebooks:**

- LSTM-TrajGAN: `notebooks/lstm_trajgan.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/lstm_trajgan.ipynb) -
  **GPU Recommended**
    - Please note that the LSTM-TrajGAN experiments do not all work in the notebook, but separate scripts explained in Section [LSTM-TrajGAN experiments](#lstm-trajgan-experiments) have to be used.
    - **Runtime Colab:** 01:35 min
- Noise-TrajGAN: `notebooks/noise_trajGAN.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/noise_trajGAN.ipynb) -
  **GPU Recommended**
  - **Runtime Colab:** 01:35 min
- AR-RNN: `notebooks/ar_rnn.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/ar_rnn.ipynb) -
  **GPU Recommended**
  - **Runtime Colab:** 03:59 min
- START-RNN: `notebooks/start_rnn.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/start_rnn.ipynb) -
  **GPU Recommended**
  - **Runtime Colab:** 03:53 min
- RGAN: `notebooks/rgan.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/rgan.ipynb) -
  **GPU Recommended**
  - **Runtime Colab:** 01:27 min
- CNN-GAN: `notebooks/cnn_gan.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/cnn_gan.ipynb) -
  **GPU Recommended**
  - **Runtime Colab:** 01:10 min
- GeoTrajGAN: `notebooks/geotrajgan.ipynb` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erik-buchholz/SoK-TrajGen/blob/main/notebooks/geotrajgan.ipynb) -
  **GPU Recommended**
  - **Runtime Colab:** 01:50 min

### Model Training

We provide the parameters of the trained models in `parameters/` such that code can be executed without time-intensive training.

The training of all models can be repeated to reproduce our results.
The model training is implemented in `stg/run/`.
The training can be run with the following commands (Use `--help`/`-h` to see the available options):

1. RNN-based models (AR-RNN, START-RNN):
  - `python3 -m stg.run.train_rnn -g GPU -d DATASET --model MODELNAME`
  - MODELNAME can be `ar_rnn` or `start_rnn`
2. RGAN:
  - `python3 -m stg.run.train_rgan -g GPU -d DATASET`
3. CNN-GAN
  - `python3 -m stg.run.train_cnn_gan -g GPU -d DATASET`
4. Noise-TrajGAN
  - `python3 -m stg.run.train_noise_trajgan -g GPU -d DATASET`
5. GeoTrajGAN
  - `python3 -m stg.run.train_gtg -g GPU -d DATASET`

If no GPU is available, use `-g -1`. Then, the training will be performed on the CPU.
To avoid lengthy command line inputs, it is possible to create and use configuration files.
These are stored in `config/` and can be used as follows:

```shell
python3 -m stg.run.rain_MODEL -g GPU -d DATASET -c CONFIG_NAME
```

Example:

```shell
python3 -m stg.run.train_rnn -g 0 -d fs --config ar_rnn_fs
```

#### Runtimes and Examples

In the following, we provide commands and approximated runtimes on the **PETS Standard VM (4 cores, 8GB memory, 40GB
disk, Ubuntu 22.4, No GPU)** and our **GPU Server (2x Intel Xeon Silver 4208, 128GB RAM, Ubuntu 20.04.01 LTS, 4 NVIDIA
Tesla T4 GPUs with 16GB RAM each)** for all configurations used in the paper (compare Table 7).
GPU usage will allow for a significant speed-up. Estimated runtimes are shown during execution thorugh a progress bar.

**Remember to replace `-g 0` by `-g -1` on machines without GPU!**

| Model   | Dataset   | Command                                                                                       | Standard VM ([HH:]MM:SS) | GPU Server ([HH:]MM:SS) |
|---------|-----------|-----------------------------------------------------------------------------------------------|--------------------------|-------------------------|
| AR      | MNIST-Seq | `python3 -m stg.run.train_rnn -g 0 -d mnist_sequential --model  ar_rnn -c ar_rnn_mnist`       | 10:00                    | 05:00                   |
| AR      | FS        | `python3 -m stg.run.train_rnn -g 0 -d fs --model  ar_rnn -c ar_rnn_fs`                        | 10:00                    | 02:00                   |
| AR      | Geolife   | `python3 -m stg.run.train_rnn -g 0 -d geolife --model  ar_rnn -c ar_rnn_geolife`              | 03:00:00                 | 30:00                   |
| START   | MNIST-Seq | `python3 -m stg.run.train_rnn -g 0 -d mnist_sequential --model  start_rnn -c start_rnn_mnist` | 10:00                    | 05:00                   |
| START   | FS        | `python3 -m stg.run.train_rnn -g 0 -d fs --model  start_rnn -c start_rnn_fs`                  | 07:00                    | 02:00                   |
| START   | Geolife   | `python3 -m stg.run.train_rnn -g 0 -d geolife --model  start_rnn -c start_rnn_geolife`        | 03:00:00                 | 30:00                   |
| RGAN    | MNIST-Seq | `python3 -m stg.run.train_rgan -g 0 -d mnist_sequential -c rgan_mnist_iwgan`                  | 15:00:00                 | 10:00:00                |
| RGAN    | FS        | `python3 -m stg.run.train_rgan -g 0 -d fs -c rgan_fs_iwgan`                                   | 02:00:00                 | 02:00:00                |
| RGAN    | Geolife   | `python3 -m stg.run.train_rgan -g 0 -d geolife -c rgan_geolife_iwgan`                         | (not tested)             | 35:00:00                |
| CNN-GAN | MNIST-Seq | `python3 -m stg.run.train_cnn_gan -g 0 -d mnist_sequential -c cnn_gan_mnist_iwgan`            | >300h                    | 17:30:00                |
| CNN-GAN | FS        | `python3 -m stg.run.train_cnn_gan -g 0 -d fs -c cnn_gan_fs_iwgan`                             | 03:00:00                 | 01:00:00                |
| CNN-GAN | Geolife   | `python3 -m stg.run.train_cnn_gan -g 0 -d geolife -c cnn_gan_geolife_iwgan`                   | (not tested)             | 06:00:00                |
| NTG     | MNIST-Seq | `python3 -m stg.run.train_noise_trajgan -g 0 -d mnist_sequential -c noise_tg_mnist`           | 10:00:00                 | 10:00:00                |
| NTG     | FS        | `python3 -m stg.run.train_noise_trajgan -g 0 -d fs -c noise_tg_fs`                            | 03:00:00                 | 02:30:00                |
| NTG     | Geolife   | `python3 -m stg.run.train_noise_trajgan -g 0 -d geolife -c noise_tg_geolife`                  | (not tested)             | 50:00:00                |
| GTG     | FS        | `python3 -m stg.run.train_gtg -g 0 -d fs -c gtg_gan_fs`                                       | 40:00:00                 | 02:00:00                |
| GTG     | Geolife   | `python3 -m stg.run.train_gtg -g 0 -d geolife -c gtg_gan_geolife`                             | (not tested)             | 90:00:00                |


### LSTM-TrajGAN Experiments

The three LSTM-TrajGAN [6] experiments are located in `stg/eval/`.
All experiments have an argument parser that provides details on the available options.
The results of the experiments can be viewed through the notebook `notebooks/lstm_trajgan.ipynb`.

However, due to the high computational cost, the training has to be run via the command line.
The experiments can be run with the following commands:

#### 1. LSTM-TrajGAN convergence.

_This measurement corresponds to **Table 4** in the paper._
The results are discussed in Appendix C.1.

```shell
python3 -m stg.eval.lstm_convergence -g GPU --dataset DATASET -b NUM_BATCHES
```

Examples (Total runtimes for *N_FOLD=5* on GPU Server in Comments):

```shell
# Case 1: FS NYC with 2,000 batches [793s]
python3 -m stg.eval.lstm_convergence -g 0 --dataset fs -b 2000  
# Case 2: FS NYC with 20,000 batches [8,037s]
python3 -m stg.eval.lstm_convergence -g 0 --dataset fs -b 20000  
# Case 3: Geolife with 2,000 batches [1,853s]
python3 -m stg.eval.lstm_convergence -g 0 --dataset geolife -b 2000  
# Case 4: Geolife with 20,000 batches [12,503s]
python3 -m stg.eval.lstm_convergence -g 0 --dataset geolife -b 20000  
# Case 5: Geolife Spatial with 2,000 batches [1,419s]
python3 -m stg.eval.lstm_convergence -g 0 --dataset geolife -b 2000 --spatial  
# Case 6: Geolife Spatial with 20,000 batches [9,631s] 
python3 -m stg.eval.lstm_convergence -g 0 --dataset geolife -b 20000 --spatial  
```  

#### 2. LSTM-TrajGAN Loss Functions

_This measurement corresponds to **Table 5** in the paper._
The results are discussed in Appendix C.2.

```shell
python3 -m stg.eval.lstm_gan -g GPU -d DATASET -b NUM_BATCHES
```

For instance (Total runtimes for *N_FOLD=5* on GPU Server in Comments):

```shell  
python3 -m stg.eval.lstm_gan -g 1 -d fs -b 2000
# Runtime GPU Server: 2,321s
python3 -m stg.eval.lstm_gan -g 1 -d geolife -b 2000
# Runtime GPU Server: 4,500s
python3 -m stg.eval.lstm_gan -g 1 -d geolife -b 2000 --spatial
# Runtime GPU Server: 3,490s
```

#### 3. LSTM-TrajGAN [5] vs RAoPT [7]

_This measurement corresponds to **Table 6** in the paper._
The results are discussed in Appendix C.3.

If the models should be trained with GPU, this evaluation requires **two
** separate GPUs, one for PyTorch and one for TensorFlow.
Unfortunately, the two frameworks cannot share a GPU within the same script.
The `-g` option specified the GPU used by TensorFlow, while the PyTorch GPU is set via `-p`.
Alternatively, it is possible to train the models with CPU by setting `-g -1` and `-p -1`, however, this will be significantly slower.
The option `--latlon_only` can be used to only use the latitude and longitude coordinates of the trajectories,
i.e., using _Geolife Spatial_ instead of _Geolife_.

Examples (Total runtimes for *N_FOLD=5* on GPU Server in Comments):

```shell  
python3 -m stg.eval.raopt_vs_lstm --runs 5 --dataset fs -g 2 -p 3 --batch_size 128 --early_stop 250  --epochs 1500  
# Runtime GPU Server: 2,615.26s
python3 -m stg.eval.raopt_vs_lstm --runs 5 --dataset geolife -g 2 -p 3 --batch_size 512 --epochs 1500 --latlon_only
# Runtime GPU Server: 14,837.45s
python3 -m stg.eval.raopt_vs_lstm --runs 5 --dataset geolife -g 2 -p 3 --batch_size 512 --epochs 1500
# Runtime GPU Server: 14,653.63s
```

## Contact

**Author:** [Erik Buchholz](https://www.erikbuchholz.de) ([e.buchholz@unsw.edu.au](mailto:e.buchholz@unsw.edu.au))

**Supervision:**

- [Prof. Salil Kanhere](https://salilkanhere.net/)
- [Dr. Surya Nepal](https://people.csiro.au/N/S/Surya-Nepal)

**Involved Researchers:**

- [Dr. Sharif Abuadbba](https://people.csiro.au/A/S/sharif-abuadbba)
- [Dr. Shuo Wang](https://people.csiro.au/w/s/shuo-wang)

**Maintainer E-mail:** [e.buchholz@unsw.edu.au](mailto:e.buchholz@unsw.edu.au)

## Acknowledgements

The authors would like to thank the University of New South Wales,
the Commonwealth of Australia, and the Cybersecurity Cooperative Research Centre Limited, whose activities are partially
funded by the Australian Government’s Cooperative Research Centres Programme, for their support.

## References

This work is based on the following publications:

[1] C. Esteban, S. L. Hyland, and G. Rätsch, “Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs.” arXiv, Dec. 03, 2017. doi: 10.48550/arXiv.1706.02633.

[2] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville, “Improved Training of Wasserstein GANs,” in Advances in Neural Information Processing Systems, in NIPS’17, vol. 30. Long Beach, California, USA: Curran Associates, Inc., 2017, pp. 5769–5779. doi: 10.5555/3295222.3295327.

[3] H. Petzka, A. Fischer, and D. Lukovnicov, “On the regularization of Wasserstein GANs.” arXiv, Mar. 05, 2018. doi: 10.48550/arXiv.1709.08894.

[4] C. Donahue, J. McAuley, and M. Puckette, “Adversarial Audio Synthesis.” arXiv, Feb. 08, 2019. Accessed: May 18, 2023. [Online]. Available: http://arxiv.org/abs/1802.04208

[5] J. Rao, S. Gao, Y. Kang, and Q. Huang, “LSTM-TrajGAN: A Deep Learning Approach to Trajectory Privacy Protection,” Leibniz International Proceedings in Informatics, vol. 177, no. GIScience, pp. 1–16, 2020, doi: 10.4230/LIPIcs.GIScience.2021.I.12.

_Their code is available at:_

[6] J. Rao, S. Gao, Y. Kang, and Q. Huang, “LSTM-TrajGAN.” GeoDS Lab @UW-Madison, 2020. Accessed: Sep. 25, 2023. [Online]. Available: https://github.com/GeoDS/LSTM-TrajGAN

[7] E. Buchholz, A. Abuadbba, S. Wang, S. Nepal, and S. S. Kanhere, “Reconstruction Attack on Differential Private Trajectory Protection Mechanisms,” in Proceedings of the 38th Annual Computer Security Applications Conference, in ACSAC ’22. New York, NY, USA: Association for Computing Machinery, December 2022, pp. 279–292. doi: 10.1145/3564625.3564628.

_Their code is available at:_

[8] E. Buchholz, S. Abuadbba, S. Wang, S. Nepal, and S. S. Kanhere, “Reconstruction Attack on Protected Trajectories (RAoPT).” [Online]. Available: https://github.com/erik-buchholz/RAoPT

[9] T. Cunningham, K. Klemmer, H. Wen, and H. Ferhatosmanoglu, “GeoPointGAN: Synthetic Spatial Data with Local Label Differential Privacy.” arXiv, May 18, 2022. doi: 10.48550/arXiv.2205.08886.

_Their code is available at:_

[10] K. Klemmer, “PyTorch implementation of GeoPointGAN.” Feb. 27, 2023. Accessed: Mar. 03, 2023. [Online]. Available: https://github.com/konstantinklemmer/geopointgan/

[11] Erik Linder-Norén, “Pytorch-GAN.” 2021. [Online]. Available: https://github.com/eriklindernoren/PyTorch-GAN

## Licence

MIT License

Copyright © Cyber Security Research Centre Limited 2022.
This work has been supported by the Cyber Security Research Centre (CSCRC) Limited
whose activities are partially funded by the Australian Government’s Cooperative Research Centres Programme.
We are currently tracking the impact CSCRC funded research. If you have used this code/data in your project,
please contact us at contact@cybersecuritycrc.org.au to let us know.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
