#!/bin/bash
#SBATCH --job-name=FloorPLanGenInference    # Nazwa zadania
#SBATCH --ntasks=1                              # Uruchomienie na jednym procesorze
#SBATCH --mem=16gb                              # Wymagana pamięć RAM
#SBATCH --time=05:00:00                         # Maksymalny limit czasu DD-HH:MM:SS
#SBATCH --output=inference.log              # Log ze stdout i stderr
#SBATCH --account=cadcam
#SBATCH --gpus=1
#SBATCH --partition=experimental,short
#SBATCH --parsable


source venv/bin/activate
python3 inference.py configs/sota.yaml
