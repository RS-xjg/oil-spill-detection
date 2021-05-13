import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from model.Unet import UNet, Attention_UNet
from model.fcn import *
from modeling.deeplab import *

def get_yaml(path):
    f = open(path,'r')
    config = f.read()
    f.close()

    config = yaml.load(config)
    return config

def get_net(config):
    if config['net'].lower() == 'UNet'.lower():
        net = UNet(n_channels=config['channels'], n_classes=config['n_classes'], bilinear=True)
        return net
    elif config['net'].lower() == 'DeepLab'.lower():
        net = DeepLab(backbone='resnet', output_stride=16, num_classes=config['n_classes'],
                sync_bn=False, freeze_bn=False)
        if config['channels'] != 3:
# 当backbone='resnet'时
             net.backbone.conv1 = nn.Conv2d(config['channels'], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# 当backbone='xception'时
            # net.backbone.conv1 = nn.Conv2d(7, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# 当backbone='mobilenet'时
            # net.backbone.features[0][0] = nn.Conv2d(7, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        return net
    elif config['net'].lower() == 'Attention_UNet'.lower():
        net = Attention_UNet(n_channels=config['channels'], n_classes=config['n_classes'], bilinear=True)
        return net

    elif config['net'].lower() in ['fcns','fcn8s','fcn16s','fcn32s','attn_fcn8s']:
        vgg_extractor =  VGGNet(pretrained=False,requires_grad=True)
        if config['net'].lower() == 'fcns':
            net  =  FCNs(pretrained_net=vgg_extractor, n_class=config['n_classes'])
        elif config['net'].lower() == 'fcn8s':
            net  =  FCN8s(pretrained_net=vgg_extractor, n_class=config['n_classes'])
        elif config['net'].lower() == 'fcn16s':
            net  =  FCN16s(pretrained_net=vgg_extractor, n_class=config['n_classes'])
        elif config['net'].lower() == 'fcn32s':
            net  =  FCN32s(pretrained_net=vgg_extractor, n_class=config['n_classes'])
        elif config['net'].lower() == 'attn_fcn8s':
            net = Attn_FCN8s(pretrained_net=vgg_extractor, n_class=config['n_classes'])
        else:
            raise NotImplementedError
        return net
    else:
        raise NotImplementedError



def get_criterion(config):
    if config['criterion'].lower() == 'crossentropy':
        return nn.CrossEntropyLoss(ignore_index=config["ignore_index"])
    elif config['criterion'].lower() == 'weight_crossentropy':
        weight = torch.FloatTensor([0.05, 0.95])  # 修改损失函数中0与1的比重，以使曲线更平滑，0.8表示1的占比，即更关注“1”类。
        weight = weight.cuda()
        return nn.CrossEntropyLoss(weight = weight, ignore_index=config["ignore_index"])
    elif config['criterion'].lower() == 'bcelogits':
        return nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError

def get_optim(config, net):
    train_params = [{'params': net.get_1x_lr_params(), 'lr': config['lr']},
                {'params': net.get_10x_lr_params(), 'lr': config['lr'] * 10}] if config['net'].lower() == 'DeepLab'.lower() else net.parameters()
    if config['net'].lower()=='deeplab':
        return optim.SGD(train_params,momentum=0.99,weight_decay=1e-5)

    # Define Optimizer
    if config['optimizer'].lower()=='adam':
        return optim.Adam(net.parameters(),lr=config['lr'],betas=(0.9,0.999),eps=1e-8,amsgrad=True)
    #   return optim.Adam(net.parameters(), lr=config['lr'], betas=(0, 0.9), eps=1e-8, amsgrad=True)
    #   return optim.Adam(net.parameters(), lr=config['lr'], betas=(0.5, 0.999), eps=1e-8, amsgrad=True)
    elif config['optimizer'].lower()=='sgd':
        return optim.SGD(net.parameters(),lr=config['lr'],momentum=0.99,weight_decay=1e-5)
    elif config['optimizer'].lower()=='rmsprop':
        return optim.RMSprop(net.parameters(), lr=config['lr'], momentum=0, weight_decay=1e-5)
    else:
        raise NotImplementedError

def get_scheduler(config,optimizer):
    if config['scheduler'].lower()=='steplr':
        return lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
    else:
        raise NotImplementedError





if __name__ == "__main__":
    config = get_yaml('ConfigFiles/config2.yaml')
    print(config)