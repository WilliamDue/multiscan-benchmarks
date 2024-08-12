#!/bin/bash
#SBATCH -p gpu --gres=gpu:a100:1 --mem=64G
#SBATCH --time=10:00

cd lexer && make
