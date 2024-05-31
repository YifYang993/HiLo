# ignore all the warnings
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger("wandb").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from apps.ICON import ICON
from lib.dataset.PIFuDataModule import PIFuDataModule
from lib.common.config import get_cfg_defaults
from lib.common.train_util import SubTrainer, load_networks
from pytorch_lightning import Trainer
import os
import os.path as osp
import argparse
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from termcolor import colored
# print(colored("For debug setting cuda visible diveices here!","red")
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["WANDB__SERVICE_WAIT"]="300"
# print(colored(f"!!!!Note set cuda visible devices here","red"))
from pytorch_lightning.utilities.distributed import rank_zero_only
@rank_zero_only
def save_code(cfg,args):
    import datetime
    import shutil
    from distutils.dir_util import copy_tree
    # timestr=str(now.month)+str(now.day)+str(now.hour)+str(now.minute)+str(now.second)
    experiment_dir = os.path.join(cfg.results_path,cfg.name,"codes")
    print("saving code to path:",experiment_dir)
    copy_tree('apps/', experiment_dir+"/apps")
    copy_tree('lib/', experiment_dir+"/lib")
    shutil.copy(args.config_file, experiment_dir+"/configs.yaml")
    logstr=str(args)
    
    with open(experiment_dir+"/args.txt",'w') as f:
        f.writelines(logstr)

def gettime():
    from datetime import datetime
    # create a datetime object with the current time
    now = datetime.now()

    # format the time string with year, month, day, hour, minute and second
    time_string = now.strftime("-%Y-%m-%d-%H-%M")

    print("Time string:", time_string)
    return time_string

@rank_zero_only
def checkname(args,cfg):
    if args.name!="baseline/icon-filter_batch2_newresumev1"  and not args.test_mode: ###conflict with ddp
        exp_name=args.name
        if os.path.exists(os.path.join(cfg.results_path,args.name,"codes")) and not args.test_mode and not args.resume:
            print("Experiment name exists, modify the experiment name!")
            exp_name=exp_name+gettime()
        name_dict=["name",exp_name]
        cfg.merge_from_list(name_dict)
    print("experimentname",cfg.name)
    return cfg

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config_file", type=str, default='configs/train/icon/icon-filter_test.yaml',help="path of the yaml config file")
    parser.add_argument("--proj_name", type=str, default='Human_3d_Reconstruction')
    parser.add_argument("--savepath", type=str, default='/mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/results/')
    parser.add_argument("-test", "--test_mode", default=False, action="store_true")
    parser.add_argument("-val", "--val_mode", default=False, action="store_true")
    parser.add_argument("--test_code", default=False, action="store_true")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--offline",default=False, action="store_true")
    parser.add_argument("--logger",type=str, default="tb")
    parser.add_argument("--name",type=str, default='baseline/icon-filter_batch2_newresumev1')
    parser.add_argument("--gpus", type=str, default='0') 
    parser.add_argument("--num_gpus", type=int, default=1) 
    parser.add_argument("--mlp_first_dim", type=int, default=0) 
    parser.add_argument("--PE_sdf", type=int, default=0) 
    parser.add_argument("--batch_size", type=int, default=2) 
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--datasettype", nargs='+',  type=str, default=["cape"]) ##1 2 3  
    ####model
    parser.add_argument("--mlpSe", default=False, action="store_true") ##spatial se
    parser.add_argument("--mlpSev1", default=False, action="store_true") ##channel se
    parser.add_argument("--mlpSemax", default=False, action="store_true")
    parser.add_argument("--mlp3d", default=False, action="store_true")
    parser.add_argument("--conv3d_start", type=int, default=2)
    parser.add_argument("--conv3d_kernelsize", type=int, default=1) ###3 5 7 is not applicable, using them results in collapsed solution
    parser.add_argument("--pad_mode", type=str, default='zeros')
    parser.add_argument("--se_start_channel", type=int, default=2)
    parser.add_argument("--se_end_channel", type=int, default=4)
    parser.add_argument("--se_reduction", type=int, default=16)
    parser.add_argument("--cse", default=False, action="store_true")
    parser.add_argument("--sse", default=False, action="store_true")
    ####uncertainty
    parser.add_argument("--uncertainty", default=False, action="store_true")
    parser.add_argument("--beta_min", type=float, default=0)
    parser.add_argument("--beta_plus", type=float, default=0)
    parser.add_argument("--kl_div", default=False, action="store_true")

    ######
    #####useclip
    parser.add_argument("--use_clip", default=False, action="store_true")
    parser.add_argument("--clip_fuse_layer", type=str, default="23") ##1 2 3
    
    #####
    ###mlp unet
    parser.add_argument("--use_unet", default=False, action="store_true")
    parser.add_argument('--mlp_dim', nargs='+', type=int, default=[13, 512, 256, 128, 1]) #res_layers 13,128,256,512,256,128,1
    parser.add_argument('--res_layers', nargs='+', type=int, default=[2,3,4]) #2,3,4,5,6
    ######
    ###triplane
    parser.add_argument("--norm_mlp", type=str, default='batch')  # 'batch' instance group
    parser.add_argument('--triplane', action='store_true',default=False)
    parser.add_argument("--hourglass_dim", type=int, default=6) ##note should be 
    ###dropout
    parser.add_argument('--dropout', type=float, default=0) #2,3,4,5,6
    parser.add_argument('--perturb_sdf', type=float, default=0) #2,3,4,5,6

    ##global and local  
    
    parser.add_argument('--train_on_thuman', default=False, action="store_true") 
    parser.add_argument("--pamir_icon", default=False, action="store_true")
    parser.add_argument('--noise_scale', nargs='+', type=float, default=[0,0]) #2,3,4,5,6
    parser.add_argument('--smplx2smpl', default=False, action="store_true") #2,3,4,5,6
    parser.add_argument('--train_on_cape', default=False, action="store_true") 
    parser.add_argument("--pamir_vol_dim", type=int, default=3)
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--no-filter', dest='filter', action='store_false')
    parser.set_defaults(filter=True)
    args = parser.parse_args()
    if args.datasettype==['cape']:
        rotation_num=3
    elif args.datasettype==["thuman2"]:
        rotation_num=36
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(["net.mlp_dim",args.mlp_dim, "net.res_layers",args.res_layers, "net.voxel_dim", args.pamir_vol_dim, "net.use_filter", args.filter, "dataset.noise_scale", args.noise_scale, "net.norm_mlp", args.norm_mlp, "batch_size",args.batch_size, "net.hourglass_dim", args.hourglass_dim, "dataset.types", args.datasettype,"num_threads",  args.num_worker, "dataset.rotation_num", rotation_num])
    # cfg=checkname(args,cfg)
    exp_name=args.name
    if args.name!="baseline/icon-filter_batch2_newresumev1"  and not args.test_mode and not args.val_mode and not args.resume: ###conflict with ddp
        if os.path.exists(os.path.join(cfg.results_path,args.name,"codes")):
            print("Experiment name exists, modify the experiment name!")
            assert 1==0
        else:
            name_dict=["name",exp_name]
            cfg.merge_from_list(name_dict)
    print("noise scale",args.noise_scale)
    cfg.freeze()
    print("note cfg is freeze pamir voxel dim", cfg.net.voxel_dim, "use filter", cfg.net.use_filter, "noise scale", cfg.dataset.noise_scale, "norm_mlp",cfg.net.norm_mlp,"batch_size",args.batch_size)
    os.makedirs(osp.join(cfg.results_path, cfg.name), exist_ok=True)
    os.makedirs(osp.join(cfg.ckpt_dir, cfg.name), exist_ok=True)
    if args.logger=="wandb":
            if not args.offline: 
                wandb_logger = WandbLogger(name=cfg.name, project=args.proj_name, save_dir=args.savepath)
            if args.offline or args.test_code:
                wandb_logger = WandbLogger(name=cfg.name, project=args.proj_name, save_dir=args.savepath,offline=True)
    elif args.logger=="tb":
        wandb_logger = pl_loggers.TensorBoardLogger(
            save_dir=args.savepath, name=cfg.name, default_hp_metric=False
        )
    else:
        print("logger not implemented")
        assert 1==0
    if cfg.overfit:
        cfg_overfit_list = ["batch_size", 1]
        cfg.merge_from_list(cfg_overfit_list)
        save_k = 0

    checkpoint = ModelCheckpoint(
        dirpath=osp.join(cfg.ckpt_dir, cfg.name),
        save_top_k=1,
        verbose=False,
        save_last=True,
        save_weights_only=True, ##here for resuming model we save optimizer lr scheduler,etc.
        monitor="val/avgloss",
        mode="min",
        filename="{epoch:02d}",
    )



    freq_eval = cfg.freq_eval
    if cfg.fast_dev > 0:
        freq_eval = cfg.fast_dev

    trainer_kwargs = {
        # "gpus": cfg.gpus,
        # "auto_select_gpus": True,
        "reload_dataloaders_every_epoch": True,
        "sync_batchnorm": True,
        "benchmark": True,
        "logger": wandb_logger,
        "track_grad_norm": -1,
        "num_sanity_val_steps": cfg.num_sanity_val_steps,
        "checkpoint_callback": checkpoint,
        "limit_train_batches": cfg.dataset.train_bsize,
        "limit_val_batches": cfg.dataset.val_bsize if not cfg.overfit else 0.001,
        "limit_test_batches": cfg.dataset.test_bsize if not cfg.overfit else 0.0,
        "profiler": None,
        "fast_dev_run": cfg.fast_dev,
        "max_epochs": cfg.num_epoch,
        "callbacks": [LearningRateMonitor(logging_interval="step")],
        # "profiler":True,
        "gpus": args.num_gpus,
        
    }

    datamodule = PIFuDataModule(cfg, args)

    datamodule.setup(stage="fit")
    train_len = datamodule.data_size["train"]
    val_len = datamodule.data_size["val"]
    trainer_kwargs.update(
        {
            "log_every_n_steps":
                int(cfg.freq_plot * train_len // cfg.batch_size),
            "val_check_interval":
                int(freq_eval * train_len // cfg.batch_size) if freq_eval > 10 else freq_eval,
        }
    )



    if cfg.overfit:
        cfg_show_list = ["freq_show_train", 100.0, "freq_show_val", 10.0]
    else:
        cfg_show_list = [
            "freq_show_train",
            max(cfg.freq_show_train * train_len // cfg.batch_size,1.0),
            "freq_show_val",
            max(cfg.freq_show_val * val_len, 1.0),
        ]

    cfg.merge_from_list(cfg_show_list)

    if args.test_mode:
        cfg_test_mode = [
            "test_mode",
            True,
            "dataset.types",
            ["cape"],
            "dataset.scales",
            [100.0],
            "dataset.rotation_num",
            3,
            "mcube_res",
            256,
            "clean_mesh",
            True,
        ]
        cfg.merge_from_list(cfg_test_mode)

    if not args.test_mode and not args.resume:
        save_code(cfg, args)


    model = ICON(cfg, args)


    trainer = SubTrainer(accelerator='ddp' if args.num_gpus>1 else None,**trainer_kwargs) ##delete normal filter, voxilization, and reconengine while saving checkpoint
    # trainer = Trainer(**trainer_kwargs)
    # load checkpoints
    if not cfg.test_mode and not args.resume: 
            print("loading filter from cfg")
            resume_path=cfg.resume_path
    elif cfg.test_mode or args.resume:
        #/mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/ckpt/baseline/icon-filter_batch2_withnormal_wosdf/epoch=09.ckpt
        # resume_path="/mnt/cephfs/dataset/NVS/experimental_results/avatar/icon/data/ckpt/baseline/icon_unet_5layer_v1/epoch=00.ckpt"
        resume_path=os.path.join(cfg.ckpt_dir,cfg.name,'last.ckpt')
        print("loading prtrained filter model from ",resume_path)    
        if not os.path.exists(resume_path):
            NotADirectoryError("checkpoint {} not exists".format(resume_path))
    currentepoch=load_networks(cfg, model, mlp_path=resume_path, normal_path=cfg.normal_path)
    if args.resume: 
        print("resuming from epoch",currentepoch)
        trainer.current_epoch=currentepoch
    if args.test_code: 
        trainer.max_epochs=2
        trainer.log_every_n_steps=1
        trainer.val_check_interval=1


    trainer.fit(model=model, datamodule=datamodule)
    # trainer.test(model=model, datamodule=datamodule)
    trainer_kwargs.update({"gpus": 1})
    trainer_val = SubTrainer(**trainer_kwargs)
    trainer_val.test(model=model, datamodule=datamodule)


##########################################################
####test_mode_in_cape#####################################
##########################################################
    args.num_gpus=1
    print("setting perturb sdf from", args.perturb_sdf)
    args.perturb_sdf=0
    print("to ", args.perturb_sdf)
    cfg_test_mode = [
        "test_mode",
        True,
        "dataset.types",
        ["cape"],
        "dataset.scales",
        [100.0],
        "dataset.rotation_num",
        3,
        "mcube_res",
        256,
        "clean_mesh",
        True,
    ]
    cfg.merge_from_list(cfg_test_mode)
    checkpoint = ModelCheckpoint(
        dirpath=osp.join(cfg.ckpt_dir, cfg.name),
        save_top_k=1,
        verbose=False,
        save_last=True,
        save_weights_only=True, ##here for resuming model we save optimizer lr scheduler,etc.
        monitor="val/avgloss",
        mode="min",
        filename="{epoch:02d}",
    )
    datamodule_val = PIFuDataModule(cfg,args)
    model_val = ICON(cfg, args)
    
    trainer_kwargs_val = {
        # "gpus": cfg.gpus,
        # "auto_select_gpus": True,
        "reload_dataloaders_every_epoch": True,
        "sync_batchnorm": True,
        "benchmark": True,
        "logger": wandb_logger,
        "track_grad_norm": -1,
        "num_sanity_val_steps": cfg.num_sanity_val_steps,
        "checkpoint_callback": checkpoint,
        "limit_train_batches": cfg.dataset.train_bsize,
        "limit_val_batches": cfg.dataset.val_bsize if not cfg.overfit else 0.001,
        "limit_test_batches": cfg.dataset.test_bsize if not cfg.overfit else 0.0,
        "profiler": None,
        "fast_dev_run": cfg.fast_dev,
        "max_epochs": cfg.num_epoch,
        "callbacks": [LearningRateMonitor(logging_interval="step")],
        # "profiler":True,
        "gpus": 1,
        
    }
    trainer_val = SubTrainer(**trainer_kwargs_val) ##delete normal filter, voxilization, and reconengine while saving checkpoint
    # trainer = Trainer(**trainer_kwargs)
    # load checkpoints

    # resume_path="data/ckpt/icon-filter.ckpt"
    resume_path=os.path.join(cfg.ckpt_dir,cfg.name,'last.ckpt')   
    if not os.path.exists(resume_path):
        NotADirectoryError("checkpoint {} not exists".format(resume_path))
    load_networks(cfg, model_val, mlp_path=resume_path, normal_path=cfg.normal_path)
    if args.test_code: 
        trainer_val.log_every_n_steps=1
        trainer_val.val_check_interval=1

    # np.random.seed(1993)
    trainer_val.test(model=model_val, datamodule=datamodule_val)

