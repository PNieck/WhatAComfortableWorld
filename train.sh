#!/bin/bash
#SBATCH --job-name=FloorPLanGenTraining   # Nazwa zadania
#SBATCH --ntasks=1                        # Uruchomienie na jednym procesorze
#SBATCH --mem=16gb                        # Wymagana pamięć RAM
#SBATCH --time=01:00:00                   # maksymalny limit czasu DD-HH:MM:SS
#SBATCH --output=training.log             # Log ze stdout i stderr
#SBATCH --account=cadcam
#SBATCH --gpus=1


venv\Scripts\activate
python3 train.py configs/main.yaml
