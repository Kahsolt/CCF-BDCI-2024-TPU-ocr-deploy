#!/usr/bin/env bash

# 编译板上 cviruntime 运行时
# WTF: 不用编译这个，直接用 https://github.com/milkv-duo/tpu-sdk-cv180x 或者我修改过的 https://github.com/Kahsolt/tpu-sdk-cv180x-ocr

# 下载交叉编译工具链
git clone https://github.com/milkv-duo/duo-sdk
export PATH=$PATH:/workspace/duo-sdk/riscv64-linux-musl-x86_64/bin/


# 编译 cviruntime 项目群
git clone https://github.com/sophgo/cviruntime
export BASE_PATH=/workspace/cviruntime
pushd cviruntime

# 1. 子项目 flatbuffers (非交叉编译)
git clone https://github.com/google/flatbuffers
pushd flatbuffers
cmake -G "Unix Makefiles"
make -j
whereis flatc
popd

# 2. 子项目 cvibuilder (非交叉编译)
git clone https://github.com/sophgo/cvibuilder
pushd cvibuilder
mkdir -p build ; cd build
cmake -DFLATBUFFERS_PATH=../flatbuffers -DCMAKE_INSTALL_PREFIX=../install_cvibuilder ..
make
make install
tree ../install_cvibuilder
popd

# 3. 子项目 cvikernel (交叉编译?)
git clone https://github.com/sophgo/cvikernel
pushd cvikernel
mkdir -p build ; cd build
sudo apt-get install ninja-build
cmake -G Ninja -DCHIP=cv180x -DCMAKE_INSTALL_PREFIX=../install_cvikernel ..
cmake --build .
cmake --build . --target install
tree ../install_cvikernel
popd

# 4. 子项目 cnpy (交叉编译)
# FUCK IT: 这个依赖暂时不管，需要进一步交叉编译 zlib
git clone https://github.com/wwwuxy/cnpy-for-tpu_mlir cnpy
pushd cnpy
mkdir -p build ; cd build
cmake -DZLIB_INCLUDE_DIR=/usr/include -DZLIB_LIBRARY=/usr/lib -DCMAKE_INSTALL_PREFIX=../install_cnpy ..
make
make install
tree ../install_cnpy
popd

# 主项目 cviruntime (交叉编译)
# 修改 CMakeLists.txt:
#  - line 28，删除 -Werror, 加上 -Wunused-parameter
#  - 把 install_cnpy 加入 include/lib (暂时不管)
# 修改 tool\CMakeLists.txt
#  - 注释掉其他目标，仅保留 cvimodel_tool 这一个 add_executable
mkdir -p build ; cd build
cmake -G Ninja -DCHIP=cv180x -DRUNTIME=SOC -DFLATBUFFERS_PATH=$BASE_PATH/flatbuffers -DCVIBUILDER_PATH=$BASE_PATH/cvibuilder/install_cvibuilder -DCVIKERNEL_PATH=$BASE_PATH/cvikernel/install_cvikernel -DCMAKE_INSTALL_PREFIX=../install ..
ninja
ninja install
cmake --build . --target test -- -v
tree ../install

popd

echo built at /workspace/cviruntime/install
echo 这个项目就是 tpu-sdk-cv180x 项目的母版，可以在这个项目上做 c++ 开发
echo 卧槽，那不就是说可以直接在 tpu-sdk-cv180x 上面做开发吗？？！
