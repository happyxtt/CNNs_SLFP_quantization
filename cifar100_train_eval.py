# final version
# python ./train_cifar100.py --Wbits 32 --Abits 32 --max_epochs 1 --lr 0.01 --wd 5e-4
import os
import time
import argparse
import torch.optim as optim
from datetime import datetime
import torch
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import torchvision

from tensorboardX import SummaryWriter

from utils.preprocessing import *
from nets.cifar100_shufflenet_v2 import *
from torch.optim.optimizer import required
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=8, suppress=True)

# Training settings
parser = argparse.ArgumentParser(description='SLFP train and finetune pytorch implementation')

""" CustomSGD is used to solve the ununiform of SLFP quantization """
parser.add_argument('--optimizer', type=str, default='SGD')  # options: SGD or CustomSGD

""" Each network corresponds to two models according to two training strategies.
    m1: learnable parameter = w
    m2: learnable parameter = w/kw. (set pretrain_dir as xxxx_m2.pth, get m2 pth by 'pre_xxxnet_weight()' function in autocode.py) """
parser.add_argument('--net', type=str, default='shufflenetv2_m1')  # options: shufflenetv2_m1, shufflenetv2_m2  
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='cifar-100')
parser.add_argument('--if_train', type = int, default=0)
parser.add_argument('--if_save', type = int, default=0)
parser.add_argument('--pretrain', action='store_true', default=True)  #default=True：use pretrained parameters  False：random seed
#parser.add_argument('--pretrain_dir', type=str, default='./ckpt/cifar-100/shufflenetv2_fp32_biasF.t7')  #default='./ckpt/resnet20_baseline 
### file 'checkpoint.t7' is damaged
#parser.add_argument('--pretrain_dir', type=str, default='./ckpt/resnet20_baseline') 


parser.add_argument('--cifar', type=int, default=100)
parser.add_argument('--Wbits', type=int, default=32)  #1
parser.add_argument('--Abits', type=int, default=32)  #32
 
parser.add_argument('--lr', type=float, default=0.001)  #0.1
parser.add_argument('--wd', type=float, default=5e-4)  #1e-4

parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=1) #200

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=5)

parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args()  

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)   

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu

'''
class CustomSGD(optim.Optimizer):   ## 第一次写的
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(CustomSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
 
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
 
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
 
                p.data.add_(-group['lr']*d_p*p.data)
        return loss
'''
class CustomSGD(optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
 
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)
 
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
 
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
 
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
 
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
 
                p.data.add_(-group['lr']*d_p)
        return loss

def main():
  
  if cfg.cifar == 10:
    print('training CIFAR-10 !')
    dataset = torchvision.datasets.CIFAR10
  elif cfg.cifar == 100:
    print('training CIFAR-100 !')
    dataset = torchvision.datasets.CIFAR100
  else:
    assert False, 'dataset unknown !'

  print('==> Preparing data ..')
  train_dataset = dataset(root=cfg.data_dir, train=True, download=True,
                          transform=cifar_transform(is_training=True))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                             num_workers=cfg.num_workers)

  eval_dataset = dataset(root=cfg.data_dir, train=False, download=True,
                         transform=cifar_transform(is_training=False))
  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                            num_workers=cfg.num_workers)
  

  num_samples = len(train_loader) * cfg.train_batch_size
  print("Total number of samples in train_loader:", num_samples)

  print("=> creating model", cfg.net, "..." )
  if cfg.net == "shufflenetv2_m1":
    model = ShuffleNetV2(wbit=cfg.Wbits).cuda()
    #pretrain_dir = './ckpt/cifar-100/shufflenetv2_fp32_biasF.t7'
    pretrain_dir = './ckpt/cifar-100/shufflenetv2_cifar_fp_65.94.pth'
    
  elif cfg.net == "mobilenet_m2":
    model = haimeixie(ch_in=3, wbit=cfg.Wbits, abit=cfg.Abits).cuda()
    pretrain_dir = './ckpt/cifar-100/shufflenetv2_m2_base.pth'

  ################-----MODEL-----################
  #model = MobileNetV3_Large(wbit=cfg.Wbits, abit=cfg.Abits).cuda()
  #model = resnet20(wbits=cfg.Wbits, abits=cfg.Abits).cuda()
  #model = VGG16(wbit=cfg.Wbits, abit=cfg.Abits).cuda()
  #model = MobileNetV1_inference(ch_in=3, n_classes=100,wbit=cfg.Wbits, abit=cfg.Abits).cuda()
  #model = MobileNetV1_scale_train(ch_in=3, n_classes=100,wbit=cfg.Wbits, abit=cfg.Abits).cuda()
  model = ShuffleNetV2(wbit=cfg.Wbits).cuda()
  ################---------------################


  #### define optimizer #######
  if cfg.optimizer == "SGD" :
    print("optimizer => SGD")
    optimizer = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  elif cfg.optimizer == "CustomSGD":
    print("optimizer => CustomSGD")
    optimizer = CustomSGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)

  lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.1)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  summary_writer = SummaryWriter(cfg.log_dir)

  if cfg.pretrain:
    model.load_state_dict(torch.load(pretrain_dir), False)

  # Training
  def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
      outputs = model(inputs.cuda())
      loss = criterion(outputs, targets.cuda())

      optimizer.zero_grad()
      #custom_optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #custom_optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
              (datetime.now(), epoch, batch_idx, loss.item(),
               cfg.train_batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def test(epoch): 
    # pass
    model.eval()   #评估模式： 无dropout，bn参数使用训练时的统计信息
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
      inputs, targets = inputs.cuda(), targets.cuda()
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(targets.data).cpu().sum().item()

    acc = 100. * correct / len(eval_dataset)
    print('%s------------------------------------------------------ '
          'Precision@1: %.2f%% \n' % (datetime.now(), acc))
    acc_data.append(acc)

    summary_writer.add_scalar('Precision@1', acc, global_step=epoch)
  
  
####################  测试并计算每一层输入(输出)，权重的最大值(Ka，Kw) ################
  '''
  total_images = 1000  #共10000输入，只统计1000个，验证最大值缩放的普遍有效性
  
  def test_with_layer_inputs_and_outputs(model, data_loader, total_images):
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

  accuracy, max_abs_layer_inputs, max_abs_layer_outputs ,max_abs_layer_weights = test_with_layer_inputs_and_outputs(model, eval_loader, total_images)
 
  print(max_abs_layer_inputs)
  print(max_abs_layer_outputs)
  print(max_abs_layer_weights)
  
  result_filename = "max_inout_shufflenet.txt"
  with open(result_filename, "w") as f:
      for idx, max_abs_input in max_abs_layer_inputs.items():
          f.write(f"Layer {idx} Max Absolute Input:\n")
          f.write(str(max_abs_input) + "\n\n")
      for idx, max_abs_output in max_abs_layer_outputs.items():
          f.write(f"Layer {idx} Max Absolute Output:\n")
          f.write(str(max_abs_output) + "\n\n")

  result_filename_weight = "max_weight_shufflenet.txt"
  with open(result_filename_weight, "w") as f:
      for idx, max_abs_layer_weights in max_abs_layer_weights.items():
          f.write(f"Layer {idx} Max Absolute weight:\n")
          f.write(str(max_abs_layer_weights) + "\n\n")
  

  print(f"Results saved to {result_filename_weight}")
  '''

##########-------------------------------------------------------############
   
  acc_data = [] 
  acc_max = 0
  for epoch in range(cfg.max_epochs):
    lr_schedu.step(epoch)
    if (cfg.if_train == 1):
      train(epoch)
      print("saving....")
    test(epoch)
    print(acc_data)        
    if (cfg.if_save == 1 and max(acc_data)> acc_max):
      acc_max = max(acc_data)
      torch.save(model.state_dict(), './ckpt/cifar-100/shufflenetv2_m1_selfSGDtmp.pth')
      print("max acc :", acc_max)
  
  

  summary_writer.close()


if __name__ == '__main__':
  main()
