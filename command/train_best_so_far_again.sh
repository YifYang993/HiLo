# for seed in {1993,3407,1,10,100,1000}
# do
# source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh;
# conda activate icon3090v1; cd /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/best_so_far/codes; CUDA_VISIBLE_DEVICES=2  python -m apps.train --gpus 0 --num_gpus 1 -cfg /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/best_so_far/codes/configs.yaml --mlpSe --filter --sse --pamir_icon --PE_sdf 14 --pamir_vol_dim 6 --datasettype cape --se_end_channel 3 --se_reduction 4 --se_start_channel 1 --name trainonthuman2/best_so_far_seed1v5 --adaptive_pe_sdf --smpl_attention --barf_c2f 0 0.1 --seed $seed --num_epoch 10
# done

#_best_so_far_train_on_thuman2_seed_{$seed}
for seed in {3407,1993,1,10,100,1000,2000,3000,4000,5000}
do
name=runagain1_best_so_far_train_on_thuman2_seed_{$seed}
command="source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh;
conda activate icon3090v1; cd /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/best_so_far/codes; CUDA_VISIBLE_DEVICES=7  python -m apps.train --gpus 0 --num_gpus 1 -cfg /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/best_so_far/codes/configs.yaml --mlpSe --filter --sse --pamir_icon --PE_sdf 14 --pamir_vol_dim 6 --train_on_thuman --datasettype thuman2 --se_end_channel 3 --se_reduction 4 --se_start_channel 1 --name $name --adaptive_pe_sdf --smpl_attention --barf_c2f 0 0.1 --seed $seed --num_epoch 10"

command_test="source /mnt/cephfs/home/qiuzhen/anaconda3/etc/profile.d/conda.sh;
conda activate icon3090v1; cd /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/best_so_far/codes; CUDA_VISIBLE_DEVICES=7  python -m apps.train --gpus 0 --num_gpus 1 -cfg /mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/best_so_far/codes/configs.yaml --mlpSe --filter --sse --pamir_icon --PE_sdf 14 --pamir_vol_dim 6 --datasettype cape --se_end_channel 3 --se_reduction 4 --se_start_channel 1 --name $name --adaptive_pe_sdf --smpl_attention --seed $seed --num_epoch 10 -test --logger wandb"


rootpath=/mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/$name
codepath=$rootpath/
target_directory=$codepath

# Create a timestamp for the filename
timestamp=$(date +"%Y%m%d%H%M%S")

# Filename
mkdir -p $target_directory
train_filename="${timestamp}_train_command.txt"
log_filepath_train="$target_directory/$train_filename"
test_filename="${timestamp}_test_command.txt"
log_filepath_test="$target_directory/$test_filename"
# Save the command to the target directory
echo "$command" > $log_filepath_train
echo "$command_test" > $log_filepath_test

# Execute the command
eval "$command"
eval "$command_test"
done