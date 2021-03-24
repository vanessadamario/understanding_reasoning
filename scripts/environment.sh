#!/bin/bash

# config_runtime.sh COPYRIGHT Fujitsu Limited 2021 and FUJITSU LABORATORIES LTD. 2021
# Authors: Atsushi Kajita (kajita@fixstars.com), G R Ramdas Pillai (ramdas@fixstars.com)

conda create -n vqa python=3.8
conda activate vqa
pip install PTable --progress-bar off
pip install opencv-python --progress-bar off
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -U ray==1.1.0 --progress-bar off
pip install nvgpu --progress-bar off
pip install reprint --progress-bar off
pip install plotly dash dash-core-components dash_html_components dash_table --progress-bar off
conda install -y -q scikit-image
conda install -y -q -c anaconda cython
conda install -y -q jupyter
conda install -y -q numba
conda install -y -q tqdm
conda install -y -q h5py



#Satori Environment:
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access/linux-ppc64le/
conda create -n vqa python=3.6
conda activate vqa
conda install pytorch
conda install -c plotly plotly
conda install -c conda-forge ptable
conda install -c anaconda psutil
conda install -c conda-forge dash dash-core-components
conda install -y -q scikit-image
conda install -y -q -c anaconda cython
conda install -y -q jupyter
conda install -y -q numba
conda install -y -q tqdm
conda install -y -q h5py
conda install -c conda-forge aiohttp
conda install -c conda-forge setproctitle
conda install -c anaconda grpcio
# Currently (Mar 23, 2021), pytorch throwing bus error, suspect lower shared memory storage on Satori nodes.
