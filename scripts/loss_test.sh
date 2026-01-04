#!/bin/bash
#SBATCH --job-name=FloorPLanGenLossTest   # Nazwa zadania
#SBATCH --ntasks=1                        # Uruchomienie na jednym procesorze
#SBATCH --mem=32gb                        # Wymagana pamięć RAM
#SBATCH --time=1:00:00                   # maksymalny limit czasu DD-HH:MM:SS
#SBATCH --output=loss_test.log             # Log ze stdout i stderr
#SBATCH --account=cadcam
#SBATCH --gpus=1
#SBATCH --partition=experimental
#SBATCH --parsable                          # to return only the job id upon submission


source venv/bin/activate
python3 train.py configs/loss_test.yaml
