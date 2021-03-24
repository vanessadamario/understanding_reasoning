#!/bin/bash

# config_runtime.sh COPYRIGHT Fujitsu Limited 2021 and FUJITSU LABORATORIES LTD. 2021
# Authors: Atsushi Kajita (kajita@fixstars.com), G R Ramdas Pillai (ramdas@fixstars.com)

# run1.sh
sudo mkdir /mnt/nvme
sudo mkfs -q -t ext4 /dev/nvme1n1
sudo mount /dev/nvme1n1 /mnt/nvme/
sudo chmod -R 777 /mnt/nvme/
cd /mnt/nvme
mkdir Downloads; cd Downloads/
wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_460.32.03_linux.run
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
sudo apt update
sudo apt install -y build-essential
sudo bash cuda_11.2.1_460.32.03_linux.run --silent --driver --toolkit --toolkitpath=/mnt/nvme/CUDA11 --samples --samplespath=/mnt/nvme/CUDA11/Samples --librarypath=/mnt/nvme/CUDA11 --installpath=/mnt/nvme/CUDA11 --no-man-page
sudo apt install -y git
sudo apt install -y nvtop
sudo apt install -y htop
sudo apt install -y net-tools
sudo apt install -y openssh-server
sudo timedatectl set-timezone America/New_York
sudo apt install -y svtools moreutils
bash Anaconda3-2020.11-Linux-x86_64.sh -bf -p /mnt/nvme/anaconda3
echo "export LD_LIBRARY_PATH=/mnt/nvme/CUDA11/lib64:${LD_LIBRARY_PATH}" >> ~/.bashrc
echo "export PATH=/mnt/nvme/CUDA11/bin:${PATH}" >> ~/.bashrc
echo "export PATH=/mnt/nvme/anaconda3/bin:${PATH}" >> ~/.bashrc
cd ~


source ~/.bashrc

#run 2.sh
conda init bash
conda update -y -q conda
conda create -n vqa -y -q python=3.8
echo "conda activate vqa" >> ~/.bashrc
echo "cd /mnt/nvme" >> ~/.bashrc

source ~/.bashrc


