#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=histEq
#SBATCH -D .
#SBATCH --output=submit-histEq.o%j
#SBATCH --error=submit-histEq.e%j
#SBATCH -A cuda
#SBATCH -p cuda

## SOLO 1 DE LAS TRES OPCIONES PUEDE ESTAR ACTIVA
## OPCION A: Usamos la RTX 4090
#SBATCH --qos=cuda4090  
#SBATCH --gres=gpu:rtx4090:1

## OPCION B: Usamos las 4 RTX 3080
##SBATCH --qos=cuda3080  
##SBATCH --gres=gpu:rtx3080:4

## OPCION C: Usamos 1 RTX 3080
##SBATCH --qos=cuda3080  
##SBATCH --gres=gpu:rtx3080:1

# Load CUDA environment (if required)
export PATH=/Soft/cuda/12.2.2/bin:$PATH

# Run your CUDA program (with image input/output)
OUTPUT_NAME="result_$(date +%s).jpg"
# ./main.exe img/sample.jpg out/result.jpg
./main.exe img/sample.jpg out/$OUTPUT_NAME
