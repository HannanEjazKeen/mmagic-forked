#!/bin/bash

#SBATCH --gres=shard:12    # Reserve 10 GB of VRAM for your job.

python3 tools/train.py configs/pix2pix/pix2pix_sonar_disc-variants.py