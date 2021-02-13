sudo mkdir /mnt/instance_store
sudo mkfs -t ext4 /dev/nvme1n1
sudo mount /dev/nvme1n1 /mnt/instance_store/
mkdir Downloads
cd Downloads/
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
sudo apt update
sudo apt install build-essential
sudo bash cuda_11.2.0_460.27.04_linux.run 
sudo apt install git
sudo apt install nvtop
sudo apt install htop
sudo apt install net-tools
sudo timedatectl set-timezone America/New_York
sudo apt install svtools moreutils
bash Anaconda3-2020.11-Linux-x86_64.sh