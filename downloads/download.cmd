@ECHO OFF

PUSHD %~dp0

REM 比赛给定出的样例工程，即本仓库原型结构
REM wget -nc https://competition-main.oss-cn-beijing.aliyuncs.com/dfadminwebsite-production/uploads/images/competitions/1044/ocr-595521.zip -O ocr.zip

REM 模型训练集 + 量化校准数据集
wget -nc https://competition-main.oss-cn-beijing.aliyuncs.com/dfadminwebsite-production/uploads/images/competitions/1044/datasets-101982.zip -O datasets.zip
unzip datasets.zip -d %~dp0..

REM 复赛数据集
wget -nc https://competition-main.oss-cn-beijing.aliyuncs.com/dfadminwebsite-production/uploads/images/competitions/1044/MSRA_Photo-398013.zip -O MSRA_Photo.zip
unzip MSRA_Photo.zip -d %~dp0..

REM 复赛数据集的源 (仅用于检测任务，无识别文本标注)
wget -nc http://pages.ucsd.edu/%7Eztu/publication/MSRA-TD500.zip
unzip MSRA-TD500.zip -d %~dp0..

POPD
