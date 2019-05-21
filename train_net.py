import torch
import torch.optim as optim

import time
import random
import os
import sys

from config import *
from volleyball import *
from collective import *
from dataset import *
from gcn_model import *
from base_model import *
from utils import *


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    
    # Reading dataset
    training_set,validation_set=return_dataset(cfg)
    
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 4
    }
    training_loader=data.DataLoader(training_set,**params)
    
    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)
    
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    basenet_list={'volleyball':Basenet_volleyball, 'collective':Basenet_collective}
    gcnnet_list={'volleyball':GCNnet_volleyball, 'collective':GCNnet_collective}
    
    if cfg.training_stage==1:
        Basenet=basenet_list[cfg.dataset_name]
        model=Basenet(cfg)
    elif cfg.training_stage==2:
        GCNnet=gcnnet_list[cfg.dataset_name]
        model=GCNnet(cfg)
        # Load backbone
        model.loadmodel(cfg.stage1_model_path)
    else:
        assert(False)
    
    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)

    model=model.to(device=device)
    
    model.train()
    model.apply(set_bn_eval)
    
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.train_learning_rate,weight_decay=cfg.weight_decay)

    train_list={'volleyball':train_volleyball, 'collective':train_collective}
    test_list={'volleyball':test_volleyball, 'collective':test_collective}
    train=train_list[cfg.dataset_name]
    test=test_list[cfg.dataset_name]
    
    if cfg.test_before_train:
        test_info=test(validation_loader, model, device, 0, cfg)
        print(test_info)

    # Training iteration
    best_result={'epoch':0, 'activities_acc':0}
    start_epoch=1
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
            
        # One epoch of forward and backward
        train_info=train(training_loader, model, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info=test(validation_loader, model, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info)
            
            if test_info['activities_acc']>best_result['activities_acc']:
                best_result=test_info
            print_log(cfg.log_path, 
                      'Best group activity accuracy: %.2f%% at epoch #%d.'%(best_result['activities_acc'], best_result['epoch']))
            
            # Save model
            if cfg.training_stage==2:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
                torch.save(state, filepath)
                print('model saved to:',filepath)   
            elif cfg.training_stage==1:
                for m in model.modules():
                    if isinstance(m, Basenet):
                        filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
                        m.savemodel(filepath)
#                         print('model saved to:',filepath)
            else:
                assert False
    
   
def train_volleyball(data_loader, model, device, optimizer, epoch, cfg):
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data=[b.to(device=device) for b in batch_data]
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
        activities_in=batch_data[3].reshape((batch_size,num_frames))

        actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
        activities_in=activities_in[:,0].reshape((batch_size,))

        # forward
        actions_scores,activities_scores=model((batch_data[0],batch_data[1]))

        # Predict actions
        actions_weights=torch.tensor(cfg.actions_weights).to(device=device)
        actions_loss=F.cross_entropy(actions_scores,actions_in,weight=actions_weights)  
        actions_labels=torch.argmax(actions_scores,dim=1)  
        actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())

        # Predict activities
        activities_loss=F.cross_entropy(activities_scores,activities_in)
        activities_labels=torch.argmax(activities_scores,dim=1)  
        activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())

        # Get accuracy
        actions_accuracy=actions_correct.item()/actions_scores.shape[0]
        activities_accuracy=activities_correct.item()/activities_scores.shape[0]
        
        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        total_loss=activities_loss+cfg.actions_loss_weight*actions_loss
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'actions_acc':actions_meter.avg*100
    }
    
    return train_info
        
    
def test_volleyball(data_loader, model, device, epoch, cfg):
    model.eval()
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    
    epoch_timer=Timer()
    with torch.no_grad():
        for batch_data_test in data_loader:
            # prepare batch data
            batch_data_test=[b.to(device=device) for b in batch_data_test]
            batch_size=batch_data_test[0].shape[0]
            num_frames=batch_data_test[0].shape[1]

            actions_in=batch_data_test[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data_test[3].reshape((batch_size,num_frames))
            
            # forward
            actions_scores,activities_scores=model((batch_data_test[0],batch_data_test[1]))
            
            # Predict actions
            actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            activities_in=activities_in[:,0].reshape((batch_size,))
            
            actions_weights=torch.tensor(cfg.actions_weights).to(device=device)
            actions_loss=F.cross_entropy(actions_scores,actions_in,weight=actions_weights)  
            actions_labels=torch.argmax(actions_scores,dim=1) 

            # Predict activities
            activities_loss=F.cross_entropy(activities_scores,activities_in)
            activities_labels=torch.argmax(activities_scores,dim=1) 
            
            actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())
            activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
            
            # Get accuracy
            actions_accuracy=actions_correct.item()/actions_scores.shape[0]
            activities_accuracy=activities_correct.item()/activities_scores.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss=activities_loss+cfg.actions_loss_weight*actions_loss
            loss_meter.update(total_loss.item(), batch_size)

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'actions_acc':actions_meter.avg*100
    }
    
    return test_info


def train_collective(data_loader, model, device, optimizer, epoch, cfg):
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data=[b.to(device=device) for b in batch_data]
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        # forward
        actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
        
        actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
        activities_in=batch_data[3].reshape((batch_size,num_frames))
        bboxes_num=batch_data[4].reshape(batch_size,num_frames)

        actions_in_nopad=[]
        if cfg.training_stage==1:
            actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
            bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
            for bt in range(batch_size*num_frames):
                N=bboxes_num[bt]
                actions_in_nopad.append(actions_in[bt,:N])
        else:
            for b in range(batch_size):
                N=bboxes_num[b][0]
                actions_in_nopad.append(actions_in[b][0][:N])
        actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
        if cfg.training_stage==1:
            activities_in=activities_in.reshape(-1,)
        else:
            activities_in=activities_in[:,0].reshape(batch_size,)
        
        # Predict actions
        actions_loss=F.cross_entropy(actions_scores,actions_in,weight=None)  
        actions_labels=torch.argmax(actions_scores,dim=1)  #B*T*N,
        actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())

        # Predict activities
        activities_loss=F.cross_entropy(activities_scores,activities_in)
        activities_labels=torch.argmax(activities_scores,dim=1)  #B*T,
        activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
        
        
        # Get accuracy
        actions_accuracy=actions_correct.item()/actions_scores.shape[0]
        activities_accuracy=activities_correct.item()/activities_scores.shape[0]
        
        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        total_loss=activities_loss+cfg.actions_loss_weight*actions_loss
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'actions_acc':actions_meter.avg*100
    }
    
    return train_info
        
    
def test_collective(data_loader, model, device, epoch, cfg):
    model.eval()
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    
    epoch_timer=Timer()
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            batch_data=[b.to(device=device) for b in batch_data]
            batch_size=batch_data[0].shape[0]
            num_frames=batch_data[0].shape[1]
            
            actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data[3].reshape((batch_size,num_frames))
            bboxes_num=batch_data[4].reshape(batch_size,num_frames)

            # forward
            actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
            
            actions_in_nopad=[]
            
            if cfg.training_stage==1:
                actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
                bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
                for bt in range(batch_size*num_frames):
                    N=bboxes_num[bt]
                    actions_in_nopad.append(actions_in[bt,:N])
            else:
                for b in range(batch_size):
                    N=bboxes_num[b][0]
                    actions_in_nopad.append(actions_in[b][0][:N])
            actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
            if cfg.training_stage==1:
                activities_in=activities_in.reshape(-1,)
            else:
                activities_in=activities_in[:,0].reshape(batch_size,)

            actions_loss=F.cross_entropy(actions_scores,actions_in)  
            actions_labels=torch.argmax(actions_scores,dim=1)  #ALL_N,
            actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())

            # Predict activities
            activities_loss=F.cross_entropy(activities_scores,activities_in)
            activities_labels=torch.argmax(activities_scores,dim=1)  #B,
            activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
            
            # Get accuracy
            actions_accuracy=actions_correct.item()/actions_scores.shape[0]
            activities_accuracy=activities_correct.item()/activities_scores.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss=activities_loss+cfg.actions_loss_weight*actions_loss
            loss_meter.update(total_loss.item(), batch_size)

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'actions_acc':actions_meter.avg*100
    }
    
    return test_info
