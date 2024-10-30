@ECHO OFF

SET MOUNT_PATH=/workspace

CD /D %~dp0
docker run --name tpu-mlir --volume="%CD%":%MOUNT_PATH% --workdir %MOUNT_PATH% --memory="0" -it -d -p 10000:10000 sophgo/tpuc_dev:latest
