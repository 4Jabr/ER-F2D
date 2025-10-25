#Import libraries
import time
import argparse
import numpy as np
import torch
import torch.nn
from utils.path_utils import ensure_dir
from torch.utils.data import DataLoader
from dataloader import build_dataset
from model_B import *
from misc import *
from metrics import *
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import math
import io
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop

def get_model_size(model):
    # to calculate the model size in MB
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.getbuffer().nbytes
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

# Ensuring checkpoint, tensorboard and logging directories are created    
checkpoint_dir = "experiment/exp_2"
print(checkpoint_dir)
ensure_dir(checkpoint_dir)
tensorboard_logdir = os.path.join(checkpoint_dir, 'tensorboard')
ensure_dir(tensorboard_logdir)
modellog_logdir = os.path.join(checkpoint_dir, 'checkpoints')
ensure_dir(modellog_logdir)
int_path = os.path.join(checkpoint_dir, 'intermediate_results')
ensure_dir(int_path)
writer = SummaryWriter(log_dir=tensorboard_logdir)

from datetime import datetime
now=datetime.now()
date_and_time = now.strftime("%d_%m_%y_%H:%M:%S")
logging_logdir = os.path.join(checkpoint_dir, 'logs')
ensure_dir(logging_logdir)
log_path =os.path.join(logging_logdir, "training_" + date_and_time + ".txt")
logging.basicConfig(filename=log_path,level=logging.INFO, format='')


# defining all the arguments
def get_args_parser():
  parser = argparse.ArgumentParser('Transformer arguments', add_help=False)
  #dataloader args
  parser.add_argument('--data_path', default = "/home/mdl/akd5994/monocular_depth/ramnet/mvsec_dataset/mvsec_dataset_day2/", type=str, help="data folder path")
  parser.add_argument('--val_data_path', default = "/home/mdl/akd5994/monocular_depth/ramnet/mvsec_dataset/mvsec_dataset_day2/", type=str, help="data folder path")
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--num_worker', default=4, type=int)
  parser.add_argument('--device', default='cuda', help='device to use for train and test, cpu or cuda')
  parser.add_argument('--clip_distance', default=80, type=int) # for mvsec
  parser.add_argument('--reg_factor', default=3.7, type=int) # for mvsec
  parser.add_argument('--train_step_size', default=8, type=int) # for mvsec training
  parser.add_argument('--val_step_size', default=16, type=int) # for mvsec training
  parser.add_argument('--filters', default=64, type=int)
  # transformer
  parser.add_argument('--num_enc_dec_layers', default=12, type=int,
                        help="Number of encoding and decoding layers in the transformer (depth)")

  parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
  parser.add_argument('--hidden_dim', default=768, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
  parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
  parser.add_argument('--nheads', default=12, type=int,
                        help ="Number of attention heads inside the transformer's attentions")
  parser.add_argument('--pre_norm', action='store_true')
  parser.add_argument('--num_res_blocks', default=1, type=int,
                        help="Number of residual blocks in RRDB")
  #training parameters
  parser.add_argument('--lr', default=3e-4, type=float)
  parser.add_argument('--weight_decay', default=1e-4, type=float)
  parser.add_argument('--lr_drop', default=200, type=int) 
  #evaluation args
  parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
  #training args
  parser.add_argument('--start_epoch', default=0, type=int)
  parser.add_argument('--epochs', default=70, type=int)
  parser.add_argument('--log_step', default=5, type=int)
  parser.add_argument('--inital_checkpoint', default=1, type=int)
  parser.add_argument('--save_freq', default=4, type=int)
  return parser


#main loop - dataloader, train, test  
def main(args):
  monitor_best = math.inf 
  monitor_mean_error = math.inf
  device = torch.device(args.device)
  print("USING DEVICE:", device)
  print("reg factor, clip distance", args.reg_factor, args.clip_distance)
  logging.info("USING DEVICE %s",device)
  #dataloader
  train_dataset = build_dataset(set="train", transform= Compose([RandomRotationFlip(0.0, 0.5, 0.0),RandomCrop(224)]),args=args)
  val_dataset = build_dataset(set="validation", transform =CenterCrop(224), args=args) 

  train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers = args.num_worker)
  val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers = args.num_worker)
  train_dataloader_len=len(train_dataloader.dataset)  
  val_dataloader_len=len(val_dataloader.dataset) 
     
  # model
  model = build_model(args)

  if args.inital_checkpoint:
    # Loading pretrained weights from vitbase 
    p=torch.load('pretrained_weights_updated_from_vitbase.pth')
    pretrained_model_weights = loading_weights_from_eventscape(model.state_dict(), p)
    model.load_state_dict(pretrained_model_weights)

  model = torch.nn.DataParallel(model)
  model=model.to(device)
  # number of parameters
  n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print('number of params:', n_parameters)
  logging.info("number of params: %d",n_parameters)
  size_mb = get_model_size(model)
  print(f"Model size: {size_mb:.2f} MB")
  # optimizer
  optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
  print("learning_rate", args.lr)
  #loss
  criterion = torch.nn.L1Loss().to(device)
  criterion_normal = NormalLoss()
  grad_criterion = multi_scale_grad_loss
  
  # training
  print("start training")
  logging.info('start_training')
  model.train()
  # randomly choosing few indices from both train and validation dataset for intermediate output visualization after each epoch
  val_preview_indices = np.random.choice(range(len(val_dataloader.dataset)), 2, replace=False)
  preview_indices = np.random.choice(range(len(train_dataloader.dataset)), 2, replace=False)
  
  for epoch in range(args.start_epoch, args.epochs):
    total_loss=0
    for batch_idx, sequence in enumerate(train_dataloader):
      rgb,events, gt= sequence[0]['image'].to(device), sequence[0]['events'].to(device), sequence[0]['depth_image'].to(device)
      optimizer.zero_grad()
      output= model(rgb, events) 
      #loss
      is_nan=torch.isnan(gt)
      l_loss = criterion(output[~is_nan], gt[~is_nan])
      grad_loss =grad_criterion(output, gt)
      gen_features = imgrad_yx(output)
      real_features = imgrad_yx(gt)
      n_loss = criterion_normal(gen_features, real_features)
      loss = 0.5*l_loss+ 0.25*grad_loss#+n_loss
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()  
      #metrics 
      pred, gt = output.cpu().detach().clone(), gt.cpu().detach().clone()
      metrics = eval_metrics(pred,gt)

      if batch_idx % args.log_step == 0: 
        print('Train Epoch:{} [{}/{} ({:.0f}%)] Loss: {:.3f} Mean Error: {:.3f}'.format(epoch, batch_idx*args.batch_size, train_dataloader_len*args.batch_size, 100*(batch_idx/train_dataloader_len), loss.item(), metrics[0]))
        logging.info('Train Epoch:{} [{}/{} ({:.0f}%)] Loss: {:.3f} Mean Error: {:.3f}'.format(epoch, batch_idx*args.batch_size, train_dataloader_len*args.batch_size, 100*batch_idx/train_dataloader_len, loss.item(), metrics[0]))
    writer.add_scalar('train loss', total_loss/train_dataloader_len, epoch) 
    
    with torch.no_grad():
      previews=[]
      for preview_idx in preview_indices:
        data = train_dataloader.dataset[preview_idx]
        rgb,events, gt = torch.unsqueeze(data[0]['image'],0), torch.unsqueeze(data[0]['events'],0), torch.unsqueeze(data[0]['depth_image'],0)
        output = model(rgb, events)
        previews.append(make_preview(int_path, rgb,events,gt,output,preview_idx,epoch,mode='train'))
    
    model.eval()
    total_val_loss=0
    with torch.no_grad():
      total_metrics = []
      for batch_idx, sequence in enumerate(val_dataloader):
        rgb,events, gt= sequence[0]['image'].to(device), sequence[0]['events'].to(device), sequence[0]['depth_image'].to(device)
        output= model(rgb, events)
        val_grad_loss = grad_criterion(output, gt)
        is_nan=torch.isnan(gt)
        val_l_loss = criterion(output[~is_nan], gt[~is_nan])
        gen_features = imgrad_yx(output)
        real_features = imgrad_yx(gt)
        val_n_loss = criterion_normal(gen_features, real_features)
        val_loss = 0.5*val_l_loss + 0.25*val_grad_loss#+ val_n_loss
        total_val_loss+=val_loss.item()
        
        # metrics
        pred, gt = output.cpu().detach().clone(),  gt.cpu().detach().clone()
        metrics = eval_metrics(pred,gt)
        total_metrics.append(metrics)
        if batch_idx % args.log_step == 0:
          print('val Epoch:{} [{}/{} ({:.0f}%)] Loss: {:.3f} Mean Error:{:.3f}'.format(epoch, batch_idx*args.batch_size, val_dataloader_len*args.batch_size, 100*(batch_idx/val_dataloader_len), val_loss.item(), metrics[0]))
          logging.info('val Epoch:{} [{}/{} ({:.0f}%)] Loss: {:.3f} Mean Error:{:.3f}'.format(epoch, batch_idx*args.batch_size, val_dataloader_len*args.batch_size, 100*batch_idx/val_dataloader_len, val_loss.item(), metrics[0]))
      writer.add_scalar('val loss', total_val_loss/val_dataloader_len, epoch)
      writer.add_scalar('mean error',metrics[0],epoch)
      loss_val = total_val_loss/val_dataloader_len
      val_previews=[]
      for preview_idx in val_preview_indices:
          data = val_dataloader.dataset[preview_idx]
          rgb,events, gt = torch.unsqueeze(data[0]['image'],0), torch.unsqueeze(data[0]['events'],0), torch.unsqueeze(data[0]['depth_image'],0)
          output = model(rgb, events)
          val_previews.append(make_preview(int_path,rgb,events,gt,output,preview_idx, epoch, mode='val'))
    avg_metrics = np.sum(np.array(total_metrics),0)/len(total_metrics)
    current_mean_error = avg_metrics[0] # mean depth error
    #saving checkpoints
    states={'epoch':epoch, 'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(),'val_loss':loss_val} 
    if epoch%args.save_freq==0:
      filename = os.path.join(modellog_logdir,'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'.format(epoch, total_loss/train_dataloader_len))
      torch.save(states, filename)
      print('Saving Checkpoints:{} at {}'.format(filename, epoch))
      logging.info('Saving Checkpoints:{} at {}'.format(filename, epoch))
      
    if(current_mean_error < monitor_mean_error):
      monitor_mean_error = current_mean_error
      filename = os.path.join(modellog_logdir,'model_best.pth.tar')
      torch.save(states, filename)
      print('Saving current best:{} at {} with mean error {}'.format(filename, epoch, monitor_mean_error))
      logging.info('Saving current best:{} at {} mean error {}'.format(filename, epoch, monitor_mean_error))
      
      
if __name__=='__main__':
  parser = argparse.ArgumentParser('Transformer Script', parents=[get_args_parser()])
  args = parser.parse_args()
  if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
  main(args)
