#!/bin/bash

echo "Currently, build only works for ubuntu 20.04 & 22.04 and cuda 121."

echo "Download ScratchLLM system dependencies"
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update -y
sudo apt-get update -y
# This is different from README in scratchLLM. The reason is gcc 13 does not
# support cuda 121.
sudo apt install g++-11 -y
sudo apt install numactl -y
sudo apt install gdb -y
sudo apt install make -y
sudo apt install emacs -y
sudo apt-get install libgflags-dev -y
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev -y
sudo apt-get install nlohmann-json3-dev -y
sudo apt-get install libre2-dev -y