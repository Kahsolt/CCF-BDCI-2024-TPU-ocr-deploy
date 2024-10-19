#!/usr/bin/env bash

# prebuild all model via 'convert_cvimodel.sh'

bash ./compile_cvimodel.sh det v4
bash ./compile_cvimodel.sh rec v4
bash ./compile_cvimodel.sh det v3
bash ./compile_cvimodel.sh rec v3
bash ./compile_cvimodel.sh det v2
bash ./compile_cvimodel.sh rec v2

bash ./compile_cvimodel.sh det mb
bash ./compile_cvimodel.sh rec mb
bash ./compile_cvimodel.sh cls mb

rm -rf ./tmp
