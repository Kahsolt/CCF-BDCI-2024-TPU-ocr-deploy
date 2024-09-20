@ECHO OFF

PUSHD %~dp0

REM contest demo project
wget -nc https://competition-main.oss-cn-beijing.aliyuncs.com/dfadminwebsite-production/uploads/images/competitions/1044/ocr-595521.zip
wget -nc https://competition-main.oss-cn-beijing.aliyuncs.com/dfadminwebsite-production/uploads/images/competitions/1044/datasets-101982.zip

REM chineseocr_lite
git clone https://github.com/DayBreak-u/chineseocr_lite

POPD

ECHO Done!
ECHO.

PAUSE
