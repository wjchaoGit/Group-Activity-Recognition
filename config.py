import time
import os


class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # Global
        self.image_size = 720, 1280  #input image size
        self.batch_size =  32  #train batch size 
        self.test_batch_size = 8  #test batch size
        self.num_boxes = 12  #max number of bounding boxes in each frame
        
        # Gpu
        self.use_gpu=True
        self.use_multi_gpu=True   
        self.device_list="0,1,2,3"  #id list of gpus used for training 
        
        # Dataset
        assert(dataset_name in ['volleyball', 'collective'])
        self.dataset_name=dataset_name 
        
        if dataset_name=='volleyball':
            self.data_path='data/volleyball'  #data path for the volleyball dataset
            self.train_seqs = [ 1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54,
                                0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]  #video id list of train set 
            self.test_seqs = [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]  #video id list of test set
            
        else:
            self.data_path='data/collective'  #data path for the collective dataset
            self.test_seqs=[5,6,7,8,9,10,11,15,16,25,28,29]
            self.train_seqs=[s for s in range(1,45) if s not in self.test_seqs]
        
        # Backbone 
        self.backbone='inv3' 
        self.crop_size = 5, 5  #crop size of roi align
        self.train_backbone = False  #if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  #output feature map size of backbone 
        self.emb_features=1056   #output feature map channel of backbone

        
        # Activity Action
        self.num_actions = 9  #number of action categories
        self.num_activities = 8  #number of activity categories
        self.actions_loss_weight = 1.0  #weight used to balance action loss and activity loss
        self.actions_weights = None

        # Sample
        self.num_frames = 3 
        self.num_before = 5
        self.num_after = 4

        # GCN
        self.num_features_boxes = 1024
        self.num_features_relation=256
        self.num_graph=16  #number of graphs
        self.num_features_gcn=self.num_features_boxes
        self.gcn_layers=1  #number of GCN layers
        self.tau_sqrt=False
        self.pos_threshold=0.2  #distance mask threshold in position relation

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 2e-4  #initial learning rate 
        self.lr_plan = {41:1e-4, 81:5e-5, 121:1e-5}  #change learning rate in these epochs 
        self.train_dropout_prob = 0.3  #dropout probability
        self.weight_decay = 0  #l2 weight decay
    
        self.max_epoch=150  #max training epoch
        self.test_interval_epoch=2
        
        # Exp
        self.training_stage=1  #specify stage1 or stage2
        self.stage1_model_path=''  #path of the base model, need to be set in stage2
        self.test_before_train=False
        self.exp_note='Group-Activity-Recognition'
        self.exp_name=None
        
        
    def init_config(self, need_new_folder=True):
        if self.exp_name is None:
            time_str=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name='[%s_stage%d]<%s>'%(self.exp_note,self.training_stage,time_str)
            
        self.result_path='result/%s'%self.exp_name
        self.log_path='result/%s/log.txt'%self.exp_name
            
        if need_new_folder:
            os.mkdir(self.result_path)