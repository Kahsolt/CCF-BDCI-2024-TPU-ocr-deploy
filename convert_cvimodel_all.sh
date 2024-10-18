#!/usr/bin/env bash

# prebuild all model via 'convert_cvimodel.sh'

bash ./convert_cvimodel.sh det v4
bash ./convert_cvimodel.sh rec v4
bash ./convert_cvimodel.sh det v3
bash ./convert_cvimodel.sh rec v3
bash ./convert_cvimodel.sh det v2
bash ./convert_cvimodel.sh rec v2

bash ./convert_cvimodel.sh det mb
bash ./convert_cvimodel.sh rec mb
bash ./convert_cvimodel.sh cls mb
