#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh
module load python/3.11/3.11.2
source .venv/bin/activate
python3 test_monet.py