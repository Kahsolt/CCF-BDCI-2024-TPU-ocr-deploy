#!/usr/bin/env bash

# 编译 sophon-sail 及其一切基础依赖！！！
# WTF: 赛方还是一直装死不给定论，我们只能自力更生

# ↓↓↓ run on host Windows
SET MOUNT_PATH=/workspace
docker run --name build-sophon-sail --volume="%CD%":%MOUNT_PATH% --workdir %MOUNT_PATH% --memory="0" -it -d sophgo/tpuc_dev:latest


# ↓↓↓ run in Docker container
export BASE_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
echo $BASE_PATH

# 系统环境基础准备
alias cls=clear
alias py=python
alias rmr='rm -rf'

pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

truncate /etc/apt/sources.list --size 0
echo deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse           >> /etc/apt/sources.list
echo deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse   >> /etc/apt/sources.list
echo deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse >> /etc/apt/sources.list
echo deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse            >> /etc/apt/sources.list
cat /etc/apt/sources.list
apt update
apt install -y nano apt-file
apt-file update


# Step 0: get cross-build toolchain
[ ! -d duo-sdk ] && git clone https://github.com/milkv-duo/duo-sdk
export PATH=$PATH:$BASE_PATH/duo-sdk/riscv64-linux-musl-x86_64/bin/
# get system image builder
[ ! -d duo-buildroot-sdk ] && git clone https://github.com/milkv-duo/duo-buildroot-sdk


# Step 1: compile libsophon (for riscv)
[ ! -d libsophon ] && git clone https://github.com/sophgo/libsophon
apt install -y \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  libncurses5-dev \
  libgflags-dev \
  libgtest-dev \
  dkms

pushd $BASE_PATH/libsophon
mkdir build && cd build
cmake \
  -DPLATFORM=soc \
  -DSOC_LINUX_DIR=$BASE_PATH/duo-buildroot-sdk/linux_5.10/ \
  -DLIB_DIR=$BASE_PATH/libsophon/3rdparty/soc/ \
  -DCROSS_COMPILE_PATH=$BASE_PATH/duo-sdk/riscv64-linux-musl-x86_64 \
  -DCMAKE_TOOLCHAIN_FILE=$BASE_PATH/toolchain-riscv64-linux-musl-x86_64.cmake \
  -DCMAKE_INSTALL_PREFIX=$PWD/../install ..
make
make driver
make vpu_driver
make jpu_driver
make package
popd

# FIXME: 不是戈门，libsophon真的能驱动 cv180x 的 TPU 吗，它的 libsophon\driver 下只有 bm1682/bm1684 阿，ntm玩我？？！
