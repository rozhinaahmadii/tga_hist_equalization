#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=sAXPYp
#SBATCH -D .
#SBATCH --output=facil
#SBATCH --error=facil
#SBATCH -A cuda
#SBATCH -p cuda

## SOLO 1 DE LAS TRES OPCIONES PUEDE ESTAR ACTIVA
## OPCION A: Usamos la RTX 4090
##SBATCH --qos=cuda4090  
##SBATCH --gres=gpu:rtx4090:1

## OPCION B: Usamos las 4 RTX 3080
##SBATCH --qos=cuda3080  
##SBATCH --gres=gpu:rtx3080:4

## OPCION C: Usamos 1 RTX 3080
#SBATCH --qos=cuda3080  
#SBATCH --gres=gpu:rtx3080:1

export PATH=/Soft/cuda/12.2.2/bin:$PATH

./Hist-CPU.exe 
##./SaxpyP.exe 
##./SaxpyP.exe 
##./SaxpyP.exe 

#echo "PRINT GPU SUMMARY nsys nvprof --print-gpu-summary"
#echo "===================================================================="
#nsys nvprof --print-gpu-summary ./SaxpyP.exe
#echo "===================================================================="

#echo "PRINT GPU SUMMARY nsys nvprof --print-gpu-trace"
#echo "===================================================================="
#nsys nvprof --print-gpu-trace ./SaxpyP.exe
#echo "===================================================================="

#./SaxpyP.exe 16777216 32 
#./SaxpyP.exe 16777216 64
#./SaxpyP.exe 16777216 128
#./SaxpyP.exe 16777216 256
#./SaxpyP.exe 16777216 512
#./SaxpyP.exe 16777216 1024

## 1024+32
##./SaxpyP.exe 16777216 1056

#./SaxpyP.exe 16777216 50 
#./SaxpyP.exe 16777216 100
#./SaxpyP.exe 16777216 200
#./SaxpyP.exe 16777216 500
#./SaxpyP.exe 16777216 1000








