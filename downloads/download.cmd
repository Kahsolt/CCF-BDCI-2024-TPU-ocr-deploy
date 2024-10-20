@ECHO OFF

PUSHD %~dp0

REM 比赛给定出的样例工程，即本仓库原型结构
REM wget -nc https://competition-main.oss-cn-beijing.aliyuncs.com/dfadminwebsite-production/uploads/images/competitions/1044/ocr-595521.zip -O ocr.zip

REM 模型训练集 + 量化校准数据集
wget -nc https://competition-main.oss-cn-beijing.aliyuncs.com/dfadminwebsite-production/uploads/images/competitions/1044/datasets-101982.zip -O datasets.zip
unzip datasets.zip -d %~dp0..

POPD
