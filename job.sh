#!/bin/bash

job_i=$1
job_j=$2
device=$3

data_type=$4
trials=$5
dims=$6
samples=$7
mixtures=$8

for (( i=job_i; i<=job_j; i++ ))
do
  tmux new -d -s $data_type-$samples-$dims-$i
  tmux send -t $data_type-$samples-$dims-$i.0 "CUDA_VISIBLE_DEVICES=$device python3 main.py $data_type $trials $dims 25 $samples $mixtures" ENTER
  sleep 1.05
done
