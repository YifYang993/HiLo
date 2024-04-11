name=HiLo_experiment1
source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh;
conda activate icon3090v1; cd /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/best_so_far/codes; 

CUDA_VISIBLE_DEVICES=3 python -m apps.inferv1 -cfg ./configs/configs_trainonthuman2.yaml --mlpSe --filter --sse --pamir_icon --PE_sdf 14 --pamir_vol_dim 6 --se_end_channel 3 --se_reduction 4 --se_start_channel 1  --smpl_attention  -gpu 0 -loop_smpl 100 -loop_cloth 200 -hps_type pymaf --ckpt_path ./ckpt/last.ckpt --adaptive_pe_sdf --expname infer_in_the_wild -in_dir ./test_images -out_dir ./results