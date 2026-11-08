#!/bin/bash

#SBATCH --gres=shard:10    # Reserve 10 GB of VRAM for your job.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmagic
python3 tools/train.py configs/pix2pix/pix2pix_sonar_disc-variants.py
