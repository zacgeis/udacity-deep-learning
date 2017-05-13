#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda -y
fi
apt-get install libcupti-dev -y
apt-get install python3-pip -y
pip3 install --upgrade pip
pip3 install tensorflow-gpu
pip3 install jupyter
echo "you also have to install https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170427/Ubuntu16_04_x64/libcudnn6_6.0.21-1+cuda8.0_amd64-deb"
