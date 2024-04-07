name=HiLo_experiment
source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh;
conda activate icon3090v1; cd /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/best_so_far/codes; 

#command for train on thuman2
CUDA_VISIBLE_DEVICES=7 python -m apps.train --gpus 0 --num_gpus 1 -cfg configs.yaml --mlpSe --filter --sse --pamir_icon --PE_sdf 14 --pamir_vol_dim 6 --train_on_thuman --datasettype thuman2 --se_end_channel 3 --se_reduction 4 --se_start_channel 1 --name $name --adaptive_pe_sdf --smpl_attention --barf_c2f 0 0.1 --seed 1 --num_epoch 10

#command for testing on cape
CUDA_VISIBLE_DEVICES=7  python -m apps.train --gpus 0 --num_gpus 1 -cfg configs.yaml --mlpSe --filter --sse --pamir_icon --PE_sdf 14 --pamir_vol_dim 6 --datasettype cape --se_end_channel 3 --se_reduction 4 --se_start_channel 1 --name $name --adaptive_pe_sdf --smpl_attention --seed 1  -test --logger wandb


