import os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from x_secretary import *
from x_secretary.pipeline import Image_KD_training,Image_classification_val,Default_DataHook,Record_Loss,Image_training
from functools import partial

from spikingjelly.clock_driven.functional import reset_net
from accelerate.utils import set_seed
from pathlib import Path

CFG=Configuration().add_args([
    ('--tIMG_TRAIN_PATH',Path,'/root/autodl-tmp/tiny-imagenet-200/train','dataset root'),
    ('--tIMG_VAL_PATH',Path,'/root/autodl-tmp/tiny-imagenet-200/val','dataset root'),
    ('-w','--WEIGHT_DIR',Path,'y_tmp/','output root dir'),
    ('-s','--SEED',int,0,'random seed'),

    ('-f','--FRAME_COUNT',int,4,'time step'),
    ('-g','--gpu',str,'0','GPU index'),
    ('--neuron',str,'PLIF','neuron type'),
    ('--arch',str,'SEW2s18','network architecture'),
]).update_from_files(['./dataset/tinyImageNet.yaml'])

set_seed(CFG.SEED)
CFG.LOCAL_RANK,CFG.WORLD_SIZE=init_cuda(CFG.gpu)
SCT=Training_Secretary(saved_dir=CFG.WEIGHT_DIR)

########################################################################
# ------------------ dataset ---------------------------
from dataset.tinyImageNet import get_tinyImageNet200_train_gpu, get_tinyImageNet200_val_gpu
CFG.train_dataset,CFG.train_trans=get_tinyImageNet200_train_gpu(CFG.tIMG_TRAIN_PATH)
CFG.val_dataset,CFG.val_trans=get_tinyImageNet200_val_gpu(CFG.tIMG_VAL_PATH)

# --------------------piplines-------------------------
from x_secretary.data_recorder import Serial,Avg
from pytorch_warmup import LinearWarmup
def Run():
    name=SCT.stage_env
    eval_pipline=Image_classification_val(CFG.VAL_BATCH_SIZE,CFG.net,CFG.val_dataset,
            data_hooks=[partial(Default_DataHook.unpack_classification,device='cuda'),CFG.val_trans],
            on_turn_begin=lambda : reset_net(CFG.net))

    def eval(training_status):
        ep=training_status['ep']
        SCT.cuda_VRAM_usage('GB')
        with Image_classification_val.switch_eval_train(CFG.net):
            acc,r,val_loss=eval_pipline(loss=torch.nn.CrossEntropyLoss())
        avg_training_loss=SCT.data[Avg(f"{name}_training_loss_{ep}")]
        SCT.data[Serial(f'{name}_val_acc',ep)]=r
        SCT.data[Serial(f'{name}_val_loss',ep)]=val_loss
        SCT.info(f'epoch: {ep}, val acc: {r*100:3f}% ({acc}),val loss: {val_loss:.5f} avg training loss:{avg_training_loss:.5f}')
        SCT.save(CFG.net,best_mode=True,best_value=r)

    Image_training(CFG,default_device='cuda',
        on_epoch_end=eval,
        data_hooks=[partial(Default_DataHook.unpack_classification,device='cuda'),CFG.train_trans],
        on_turn_end=[Record_Loss(SCT,name_prefix=name)],
        on_turn_begin= lambda _ : reset_net(CFG.net),
        )()


########################################################################
# ------------------------ construct network ---------------------------
from xs_snn.norm.RateBatchNorm import RateBatchNorm
from network.l_sign import CLASSIFICATION
from model import PLIF,VNext2,Original_MLF
from xs_snn.neuron import ReLU6
NEURON_MAP=get_name_dict([PLIF,ReLU6,Original_MLF,VNext2])

from network.snn.snn_resnet2 import SEW2s18,SR2s18,SEW2s34,SR2s34
from network.snn.snn_vgg import VGGs11,VGGs16

NETWORK_MAP=get_name_dict([SR2s18,SEW2s18,VGGs11,SEW2s34,SR2s34,VGGs16])
CFG.net=NETWORK_MAP[CFG.arch](
    neuron_model=NEURON_MAP[CFG.neuron],
    frame_count=CFG.FRAME_COUNT,
    mode=CLASSIFICATION.tinyImageNet,
    num_classes=CFG.N_CLASSES,
    norm_layer=RateBatchNorm)

CFG.net.train()
CFG.net.cuda()
if CFG.DDP:
    CFG.net=torch.nn.SyncBatchNorm.convert_sync_batchnorm(CFG.net)
    CFG.net=torch.nn.parallel.distributed.DistributedDataParallel(CFG.net,find_unused_parameters=True)

p_decay,p_norm=split_decay_parameters(CFG.net)
CFG.opt=torch.optim.SGD(
    [{'params':p_decay,'initial_lr':CFG.initial_lr,'weight_decay':CFG.weight_decay},
     {'params':p_norm, 'initial_lr':CFG.initial_lr,'weight_decay':0}],
    lr=CFG.initial_lr,momentum=CFG.momentum,)
CFG.warmup_scheduler=LinearWarmup(CFG.opt,CFG.WARMUP)
CFG.lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(CFG.opt,CFG.T_MAX,CFG.MIN_LR)

CFG.loss=torch.nn.CrossEntropyLoss()

CFG.NAME=f'tIMG_{CFG.neuron}_{CFG.arch}_f{CFG.FRAME_COUNT}s{CFG.SEED}'

SCT.log_to_cfg(f'parameters: {count_parameters(CFG.net)}\n')
SCT.log_cfg_changes(CFG)

########################################################################
SCT.set_working_dir_name(CFG.NAME).timing('total')
Run()
SCT.timing('total')

# python main_train_timg.py
# torchrun --nproc_per_node=4 main_train_timg.py