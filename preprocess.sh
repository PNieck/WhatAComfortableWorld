#!/bin/bash
#SBATCH --job-name=FloorPLanGenPreprocessing    # Nazwa zadania
#SBATCH --ntasks=1                              # Uruchomienie na jednym procesorze
#SBATCH --mem=16gb                              # Wymagana pamięć RAM
#SBATCH --time=00:30:00                         # Maksymalny limit czasu DD-HH:MM:SS
#SBATCH --output=preprocessing.log              # Log ze stdout i stderr
#SBATCH --account=cadcam


venv\Scripts\activate
python3 preprocess.py configs/main.yaml data/data.mat
