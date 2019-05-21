import sys
sys.path.append(".")
from train_net import *

cfg=Config('volleyball')

cfg.device_list="0,1,2,3"
cfg.training_stage=1
cfg.stage1_model_path=''
cfg.train_backbone=True

cfg.batch_size=8
cfg.test_batch_size=4
cfg.num_frames=1
cfg.train_learning_rate=1e-5
cfg.lr_plan={}
cfg.max_epoch=200
cfg.actions_weights=[[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]  

cfg.exp_note='Volleyball_stage1'
train_net(cfg)