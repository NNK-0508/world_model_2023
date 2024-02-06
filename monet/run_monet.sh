#!/bin/bash

current_date=$(date +"%Y-%m-%d")
file_path="logs/$current_date/$file_count"
mkdir -p "$file_path"

file_count=$(ls "$file_path" | wc -l)
mkdir -p "$file_path"/"$file_count"

qsub -g gcc50435 -m e -j y -o "$file_path"/"$file_count"/stderror.log ./job_scripts/run_monet.sh