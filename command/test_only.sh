name=HiLo_experiment1
source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh;
conda activate icon3090v1; 

#command for testing on cape
CUDA_VISIBLE_DEVICES=3  python -m apps.train --gpus 0 --num_gpus 1 -cfg configs.yaml --mlpSe --filter --sse --pamir_icon --PE_sdf 14 --pamir_vol_dim 6 --datasettype cape --se_end_channel 3 --se_reduction 4 --se_start_channel 1 --name $name --adaptive_pe_sdf --smpl_attention --seed 1  -test --logger wandb --ckpt_full_path ckpt/last.ckpt


