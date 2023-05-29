#!/bin/sh
#PBS -V
#PBS -q v100
#PBS -N N4
#PBS -l nodes=1:ppn=16
#export CUDA_VISIBLE_DEVICES="4,5"
source /public/home/group_zyl/.bashrc
# conda environment
conda_env=pt200
export OMP_NUM_THREADS=16
#path to save the code
path="/public/home/group_zyl/zyl/program/Equi-MPNN/"

conda activate $conda_env 
cd $PBS_O_WORKDIR 
python3 $path >out
