#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=sAXPYp
#SBATCH -D .
#SBATCH --output=facil
#SBATCH --error=facil
#SBATCH -A cuda
#SBATCH -p cuda

## SOLO 1 DE LAS TRES OPCIONES PUEDE ESTAR ACTIVA

## OPCION A: Usamos 1 RTX 3080
#SBATCH --qos=cuda3080  
#SBATCH --gres=gpu:rtx3080:1

## OPCION B: Usamos 2 RTX 3080
##SBATCH --qos=cuda3080  
##SBATCH --gres=gpu:rtx3080:2

## OPCION C: Usamos 4 RTX 3080
##SBATCH --qos=cuda3080  
#3SBATCH --gres=gpu:rtx3080:4

export PATH=/Soft/cuda/12.2.2/bin:$PATH

./Hist-CPU.exe 


#echo "PRINT GPU SUMMARY nsys nvprof --print-gpu-summary"
#echo "===================================================================="
#nsys nvprof --print-gpu-summary ./SaxpyP.exe
#echo "===================================================================="

#echo "PRINT GPU SUMMARY nsys nvprof --print-gpu-trace"
#echo "===================================================================="
#nsys nvprof --print-gpu-trace ./SaxpyP.exe
#echo "===================================================================="









