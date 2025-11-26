#!/bin/bash
#SBATCH --job-name=FloorPLanGenTraining   # Nazwa zadania
#SBATCH --ntasks=1                        # Uruchomienie na jednym procesorze
#SBATCH --mem=32gb                        # Wymagana pamięć RAM
#SBATCH --time=24:00:00                   # maksymalny limit czasu DD-HH:MM:SS
#SBATCH --output=training_gpt2_bigger.log             # Log ze stdout i stderr
#SBATCH --mail-type=END,FAIL              # Powiadomienia mailowe. Opcje: NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=piotr.nieciecki01@gmail.com         # adres e-mail
#SBATCH --account=cadcam
#SBATCH --gpus=1
#SBATCH --partition=experimental,short
#SBATCH --parsable                          # to return only the job id upon submission


source venv/bin/activate
python3 train.py configs/gpt2_bigger.yaml
