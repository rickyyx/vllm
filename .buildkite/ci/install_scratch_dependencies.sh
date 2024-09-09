#!/bin/bash

echo "Currently, build only works for ubuntu 20.04 & 22.04 and cuda 121."

echo "Download ScratchLLM system dependencies"
sudo apt-get update
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update -y
# This is different from README in scratchLLM. The reason is gcc 13 does not
# support cuda 121.
sudo apt-get install g++-11 numactl gdb make emacs cmake build-essential pkg-config libgoogle-perftools-dev nlohmann-json3-dev libre2-dev -y
