import sys
sys.path.append(".")
from train_net import *

cfg=Config('volleyball')

cfg.device_list="0,1,2,3"
cfg.training_stage=2
cfg.stage1_model_path='result/STAGE1_MODEL.pth'  #PATH OF THE BASE MODEL
cfg.train_backbone=False

cfg.batch_size=32 #32
cfg.test_batch_size=8 
cfg.num_frames=3
cfg.train_learning_rate=2e-4 
cfg.lr_plan={41:1e-4, 81:5e-5, 121:1e-5}
cfg.max_epoch=150
cfg.actions_weights=[[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]  

cfg.exp_note='Volleyball_stage2'
train_net(cfg)