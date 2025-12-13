#!/bin/bash
#SBATCH --job-name=FloorPLanGenTraining         # Nazwa zadania
#SBATCH --ntasks=1                              # Uruchomienie na jednym procesorze
#SBATCH --mem=32gb                              # Wymagana pamięć RAM
#SBATCH --time=00:30:00                         # maksymalny limit czasu DD-HH:MM:SS
#SBATCH --output=training_gpt2_fast_test.log    # Log ze stdout i stderr
#SBATCH --mail-type=END,FAIL                    # Powiadomienia mailowe. Opcje: NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=piotr.nieciecki01@gmail.com # adres e-mail
#SBATCH --account=cadcam                        # Grupa badawcza
#SBATCH --gpus=1                                # Liczba kart graficznych
#SBATCH --partition=experimental,short          # Kolejki do których wrzucam zadnie (slurm sam wybiera tą na której zadnie uruchomi się jako pierwsze)
#SBATCH --parsable                              # Komenda sbatch zwraca tylko job id


source venv/bin/activate
python3 train.py configs/gpt2_fast_test.yaml
