#!/bin/bash
export PYTHONPATH=/work/xz464/composition_antispoofing_dkucc/Unet:$PYTHONPATH
source ~/.bashrc
conda activate Antispoofing
(
while true; do
    nvidia-smi >> /work/xz464/composition_antispoofing_dkucc/Unet/nvidia-smi.out
    sleep 60
done
) &
cd /work/xz464/composition_antispoofing_dkucc
python -m Unet.train