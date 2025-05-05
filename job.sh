#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=PythonCUDA
#SBATCH -D .
#SBATCH --output=submit-PythonCUDA.o%j
#SBATCH --error=submit-PythonCUDA.e%j
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


export CUDA_HOME=/Soft/cuda/12.2.2
export PATH=/Soft/python_modules/bin:/Soft/cuda/12.2.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/Soft/cuda/12.2.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


python3 TestPythonCuda00.py > Res00.txt
python3 TestPythonCuda01.py > Res01.txt
python3 TestPythonCuda02.py > Res02.txt
python3 TestPythonCuda03.py > Res03.txt
python3 TestPythonCuda04.py > Res04.txt
python3 TestPythonCuda05.py > Res05.txt
python3 TestPythonCuda06.py > Res06.txt


