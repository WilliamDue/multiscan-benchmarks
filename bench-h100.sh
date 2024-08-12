#!/bin/bash
#SBATCH -p gpu --gres=gpu:h100:1 --mem=64G
#SBATCH --time=1:00:00

make
