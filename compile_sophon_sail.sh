#!/usr/bin/env bash

# 编译板上 sophon.sail 运行时 (不行！！！)
# libsophon 是由 riscv64-linux-gnu-gcc 编译的，而 MilkV-Duo 是 riscv64-unknown-linux-musl-gcc 编译的，运行时不匹配！！
# libsophon 未发行源码只有二进制，那么只可能在 MilkV-Duo 上迁移一套 riscv64-linux-gnu-gcc 的运行时 (can we do this?)

# 下载算能SDK整合包 `SDK-24.04.01.zip` (14G)
# https://developer.sophgo.com/site/index/material/88/all.html


# 安装 libsophon (RISCV 版)
# BUG: 得用 riscv64-linux-musl-x86_64 重新编译才能在 Duo 上跑，但 libsophon 未发行源码不可能重编
# https://doc.sophgo.com/sdk-docs/v24.04.01/docs_latest_release/docs/libsophon/guide/html/1_install.html#linux
tar xvf downloads/SDK-24.04.01/libsophon_20240624_160933/libsophon_0.5.1_riscv64.tar.gz -C /opt
mv /opt/libsophon_0.5.1_riscv64/opt/sophon /opt/
rmdir /opt/libsophon_0.5.1_riscv64


# 编译 & 安装 sophon-sail (RISCV MODE)
# https://github.com/sophgo/sophon-sail?tab=readme-ov-file#%E7%BC%96%E8%AF%91%E5%8F%AF%E8%A2%ABpython3%E6%8E%A5%E5%8F%A3%E8%B0%83%E7%94%A8%E7%9A%84wheel%E6%96%87%E4%BB%B6
# https://github.com/sophgo/sophon-sail/blob/master/GETPYTHON.md

# 拉取 sophon-sail 项目
git clone https://github.com/sophgo/sophon-sail

# 下载交叉编译工具链 (依照 libsophon 使用 riscv64-linux-gnu-gcc 工具链)
#sudo apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
# 下载交叉编译工具链 (依照 MilkV-Duo 使用 riscv64-linux-musl-gcc 工具链)
git clone https://github.com/milkv-duo/duo-sdk
export PATH=$PATH:/workspace/duo-sdk/riscv64-linux-musl-x86_64/bin/

# 修改编译器前缀
# sophon-sail\cmake\BM168x_RISCV\ToolChain_riscv64_linux.cmake: line 6
# set(CROSS_COMPILE riscv64-linux-gnu-) -> set(CROSS_COMPILE riscv64-unknown-linux-musl-)

# 交叉编译目标 python 版本
# https://www.cnblogs.com/tiange-137/p/17459787.html

# 交叉编译目标 python 版本，直接从板子上搞过来!!
mkdir -p runtime/python3.9.5/bin
mkdir -p runtime/python3.9.5/lib
scp    root@192.168.42.1:/usr/bin/python3.9                    runtime/python3.9.5/bin
scp    root@192.168.42.1:/usr/lib/libpython3.so                runtime/python3.9.5/lib
scp    root@192.168.42.1:/usr/lib/libpython3.9.so              runtime/python3.9.5/lib
scp    root@192.168.42.1:/usr/lib/libpython3.9.so.1.0          runtime/python3.9.5/lib
scp -r root@192.168.42.1:/usr/lib/python3.9                    runtime/python3.9.5/lib
scp    root@192.168.42.1:/lib/ld-musl-riscv64v0p7_xthead.so.1  /lib

# 编译动态链接库 (注意这里都是 riscv 的二进制)
pushd sophon-sail
mkdir -p build ; cd build
cmake -DBUILD_TYPE=riscv  \
  -DONLY_RUNTIME=ON \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/BM168x_RISCV/ToolChain_riscv64_linux.cmake \
  -DPYTHON_EXECUTABLE=../../downloads/python_3.9.0/bin/python3.9 \
  -DCUSTOM_PY_LIBDIR=../../downloads/python_3.9.0/lib \
  -DLIBSOPHON_BASIC_PATH=/opt/sophon/libsophon-0.5.1 ..
make pysail
popd

# 编译 python 轮子 (为啥编出来是 3.8.0 ??!)
pushd sophon-sail/python/riscv
bash ./sophon_riscv_whl.sh

# 安装 python 轮子
# - 不要用这个，会提示磁盘空间不足
#cp dist/sophon_riscv64-3.8.0-py3-none-any.whl ../../../runtime
#pip3 install sophon_riscv64-3.8.0-py3-none-any.whl --force-reinstall
# - 直接拷贝文件夹算了 :(
objdump -x sophon/sail.so | grep NEEDED
scp -r sophon root@192.168.42.1:/root
scp /opt/sophon/libsophon-0.5.1/lib/libbmrt.so.1.0 root@192.168.42.1:/lib/libbmlib.so.0
scp /opt/sophon/libsophon-0.5.1/lib/libbmrt.so.1.0 root@192.168.42.1:/lib/libbmrt.so.1.0
scp /usr/riscv64-linux-gnu/lib/ld-linux-riscv64-lp64d.so.1 root@192.168.42.1:/lib/ld-linux-riscv64-lp64d.so.1
popd

# 板上尝试 import 仍然报错
