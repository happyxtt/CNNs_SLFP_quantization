## resnet50 的wbit控制在resnet50.py里，wbit = 32/ wbit = 8
## python ./imgnet_train_eval.py --Wbits 8 --Abit 8 --wd 1e-4 --lr 0.000008 --pretrain --max_epochs 2
import os
import time
import argparse
from datetime import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
import psutil # monitor CPU utilization
cudnn.benchmark = True
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from tqdm import tqdm
#from nets.img100_VGG import *
from nets.mobilenetv1 import *
from nets.resnet50_official import *
from nets.alexnet import *
from nets.squeezenet1_0 import *
from utils.preprocessing import *
# Training settings
parser = argparse.ArgumentParser(description='DoReFa-Net pytorch')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='/opt/datasets/imagenet-1k')
parser.add_argument('--log_name', type=str, default='imgnet-1k')
parser.add_argument('--pretrain', action='store_true', default=True)
parser.add_argument('--if_train', type = int, default=0)
parser.add_argument('--all_validate', type = int, default=0)
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/resnet-50.pth')

parser.add_argument('--Wbits', type=int, default=32)
parser.add_argument('--Abits', type=int, default=32)

parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--wd', type=float, default=5e-4)

parser.add_argument('--train_batch_size', type=int, default=32)#256
parser.add_argument('--eval_batch_size', type=int, default=16)#100
parser.add_argument('--max_epochs', type=int, default=2)

parser.add_argument('--log_interval', type=int, default=500) #10
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=1) #20

parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args()

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

if not cfg.cluster:
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu


def main():
  # Data loading code
  traindir = os.path.join(cfg.data_dir, 'train')
  valdir = os.path.join(cfg.data_dir, 'val')
  
  train_dataset = datasets.ImageFolder(traindir, imgnet_transform(is_training=True))
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.train_batch_size,
                                             shuffle=True,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True)
  
  val_dataset = datasets.ImageFolder(valdir, imgnet_transform(is_training=False))
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=cfg.eval_batch_size,
                                           shuffle=False,
                                           num_workers=cfg.num_workers,
                                           pin_memory=True)

  # create model
  print("=> creating model...")
  #model = MobileNetV1_Q(ch_in=3, wbit=cfg.Wbits, abit=cfg.Abits).cuda()
  #model = AlexNet(wbit = cfg.Wbits).cuda()
  #model = SqueezeNet(wbit = cfg.Wbits).cuda()
  model = ResNet50(wbit = cfg.Wbits).cuda()

  # optionally resume from a checkpoint
  if cfg.pretrain:
    model.load_state_dict(torch.load(cfg.pretrain_dir))

  # define loss function (criterion) and optimizer
  optimizer = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [5, 10, 15], gamma=0.3)
  criterion = nn.CrossEntropyLoss().cuda()

  summary_writer = SummaryWriter(cfg.log_dir)

  def train(epoch):
    # switch to train mode
    model.train()

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      # compute output
      output = model(inputs.cuda())
      loss = criterion(output, targets.cuda())

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
              (datetime.now(), epoch, batch_idx, loss.item(),
               cfg.train_batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)
        torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'resnet_8perclass_slfp34.pth'))


  def validate(epoch):
    # switch to evaluate mode
    model.eval()
    top1 = 0
    top5 = 0
    if (cfg.all_validate == 1 ):
      num_samples = len(val_dataset)
    else:
      num_samples = 5000
   
    #with tqdm(total=len(val_dataset)) as pbar:
     # for i, (inputs, targets) in enumerate(val_loader):

    with tqdm(total=num_samples) as pbar:
      for i, (inputs, targets) in enumerate(val_loader):
        if i * cfg.eval_batch_size >= num_samples:
          break  # 停止迭代，已经测试了指定数量的样本

        targets = targets.cuda()
        input_var = inputs.cuda()

        # compute output
        output = model(input_var)

        # measure accuracy and record loss
        _, pred = output.data.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        top1 += correct[:1].view(-1).float().sum(0, keepdim=True).item()
        top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
        pbar.update(cfg.eval_batch_size)

    top1 *= 100 / num_samples
    top5 *= 100 / num_samples
    print('%s------------------------------------------------------ '
          'Precision@1: %.2f%%  Precision@1: %.2f%%\n' % (datetime.now(), top1, top5))

    summary_writer.add_scalar('Precision@1', top1, epoch)
    summary_writer.add_scalar('Precision@5', top5, epoch)
    
    return top1, top5
  
  
####################  测试并计算每一层输入(输出)，权重的最大值(Ka，Kw) ################
  
  def test_with_layer_inputs_and_outputs(model, data_loader, total_images ): #共10000输入，只统计1000个，验证最大值缩放的普遍有效性
      model.eval()
      correct = 0
      count = 0
      layer_inputs = {}  # Used to store inputs for each layer
      layer_outputs = {}  # Used to store outputs for each layer
      layer_weights = {}
      #inputimg = {}
      
      for batch_idx, (inputs, targets) in enumerate(data_loader):
          inputs, targets = inputs.cuda(), targets.cuda()
     
          # Record inputs and outputs for each layer
          model.reset_layer_inputs_outputs()
          model.reset_layer_weights()
          outputs = model(inputs)
          # Get inputs and outputs for each layer
          current_layer_inputs = model.get_layer_inputs()
          #print(current_layer_inputs)
          current_layer_outputs = model.get_layer_outputs()
          current_layer_weights = model.get_layer_weights()
          #print(inputimg.shape)
          #print(inputimg[0].cpu().numpy())
          
          for idx, input_tensor in current_layer_inputs.items():
              if idx not in layer_inputs:
                  layer_inputs[idx] = []
              layer_inputs[idx].append(input_tensor.detach().cpu())
              
          for idx, output_tensor in current_layer_outputs.items():
              if idx not in layer_outputs:
                  layer_outputs[idx] = []
              layer_outputs[idx].append(output_tensor.detach().cpu())

          for idx, output_tensor in current_layer_weights.items():
              if idx not in layer_weights:
                  layer_weights[idx] = []
              layer_weights[idx].append(output_tensor.detach().cpu())

          _, predicted = torch.max(outputs.data, 1)
          correct += predicted.eq(targets.data).cpu().sum().item()

          count += len(inputs)
          if count >= total_images:
              break
        
      # Calculate the maximum absolute values for layer inputs and outputs
      max_abs_layer_inputs = {}
      max_abs_layer_outputs = {}
      max_abs_layer_weights = {}

      for idx, inputs_list in layer_inputs.items():
          max_abs_input = torch.max(torch.abs(torch.cat(inputs_list, dim=0)))
          max_abs_layer_inputs[idx] = max_abs_input.item()
          
      for idx, outputs_list in layer_outputs.items():
          max_abs_output = torch.max(torch.abs(torch.cat(outputs_list, dim=0)))
          max_abs_layer_outputs[idx] = max_abs_output.item()

      for idx, weights_list in layer_weights.items():
          max_abs_output = torch.max(torch.abs(torch.cat(weights_list, dim=0)))
          max_abs_layer_weights[idx] = max_abs_output.item()
 
      acc = 100. * correct / total_images
      print('%s------------------------------------------------------ '
            'Precision@1: %.2f%% \n' % (datetime.now(), acc))

      return acc, max_abs_layer_inputs, max_abs_layer_outputs, max_abs_layer_weights
  
  ###############----------------- --------############
  '''
  total_images = 1000
  accuracy, max_abs_layer_inputs, max_abs_layer_outputs ,max_abs_layer_weights = test_with_layer_inputs_and_outputs(model, val_loader, total_images)
 
  print(max_abs_layer_inputs)
  print(max_abs_layer_outputs)
  print(max_abs_layer_weights)
  
  result_filename = "max_inout_squeezenet.txt"
  with open(result_filename, "w") as f:
      for idx, max_abs_input in max_abs_layer_inputs.items():
          f.write(f"Layer {idx} Max Absolute Input:\n")
          f.write(str(max_abs_input) + "\n\n")
      for idx, max_abs_output in max_abs_layer_outputs.items():
          f.write(f"Layer {idx} Max Absolute Output:\n")
          f.write(str(max_abs_output) + "\n\n")

  result_filename_weight = "max_weight_squeezenet.txt"
  with open(result_filename_weight, "w") as f:
      for idx, max_abs_layer_weights in max_abs_layer_weights.items():
          f.write(f"Layer {idx} Max Absolute weight:\n")
          f.write(str(max_abs_layer_weights) + "\n\n")
  

  print(f"Results saved to {result_filename_weight}")
  '''

  ##########------------------------------------############

  for epoch in range(1, cfg.max_epochs):
    lr_schedu.step(epoch)
    if (cfg.if_train == 1):
      train(epoch)
    validate(epoch)
    #torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'mobilenet_finetune.pth'))


if __name__ == '__main__':
  main()
