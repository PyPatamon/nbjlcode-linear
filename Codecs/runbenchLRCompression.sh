#########################################################################
# File: runbenchDetlaCompression.sh
# Author: Naiyong Ao
# Email: aonaiyong@gmail.com
# Time: Mon 09 Feb 2015 08:15:54 PM CST
#########################################################################
#!/bin/bash

index_dir="/home/naiyong/dataset/"
minlens=(1 3 4 5)

rm -rf stats/*
for i in ${minlens[@]}
do
    for dataset in `ls $index_dir`
    do
        ./benchCompression $index_dir$dataset"/" $dataset LR $i >> "stats/"$dataset"_LR_CPUCodecsStats.txt"
    done
done
