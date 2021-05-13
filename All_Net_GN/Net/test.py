# 本代码单独用以测试
import os
import sys
import torch
import cv2
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
from utils import decode_segmap
from SegDataFolder import semData
from getSetting import get_yaml, get_criterion, get_optim, get_scheduler, get_net
from metric import AverageMeter, intersectionAndUnion
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from modeling.deeplab import *
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import Dataset
import torchvision.transforms.transforms as _transform
import torch


class Solver(object):
    def __init__(self,configs):
        self.configs = configs
        self.cuda = torch.cuda.is_available()

        self.n_classes = self.configs['n_classes']
        self.ignore_index = self.configs['ignore_index']
        self.channels = self.configs['channels']
        self.net = get_net(self.configs)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = get_criterion(self.configs)
        self.optimizer = get_optim(self.configs, self.net)
        self.scheduler = get_scheduler(self.configs, self.optimizer)
        self.batchsize = self.configs['batchsize']
        self.start_epoch = self.configs['start_epoch']
        self.end_epoch = self.configs['end_epoch']
        self.logIterval = self.configs['logIterval']
        self.valIterval = self.configs['valIterval']

        self.resume = self.configs['resume']['flag']
        if self.resume:
            self.resume_state(self.configs['resume']['state_path'])
        
        if self.cuda:
            self.net = self.net.cuda()
            if self.resume:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        self.trainSet = semData(train=True, channels=self.channels)
        self.valSet = semData(train=False, channels=self.channels)
        self.train_dataloader = torch.utils.data.DataLoader(self.trainSet, batch_size=self.batchsize, shuffle=True) 
        self.val_dataloader = torch.utils.data.DataLoader(self.valSet, 1, shuffle=False)

        self.best_miou = 0.00
        self.result_dir = self.configs['result_dir']
        os.makedirs(self.result_dir,exist_ok=True)
        self.writer = SummaryWriter(self.result_dir)
        with self.writer:
            if not self.resume:
                inp = torch.randn([1,self.channels,512,512]).cuda() if self.cuda else torch.randn([1,self.channels,512,512])
                self.writer.add_graph(self.net, inp)

    def save_state(self, epoch, path):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, path)

    def resume_state(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        print('Resume from epoch{}...'.format(self.start_epoch))

    def test(self, singleImages=[], outName='1_2.jpg'):
        assert len(singleImages) == self.channels
        from PIL import Image
        import transform as T
       # mean = np.array([0.013366839, -1.0583542e-05, -4.3700067e-05, 0.0026632368, 19.917131, 0.6687881, 0.6205375])  # 训练原图的平均值
       # std = np.array([0.20220748, 0.017818337, 0.013740968, 0.006313215, 7.6970053, 0.14615154, 0.15835664])  # 训练原图的方差
        mean = np.array([0.013083419, -9.451074e-06, -3.988814e-05, 0.002662216, 19.957167, 0.6682034, 0.6206584])  # 测试原图的平均值
        std = np.array([0.030120198, 0.008994167, 0.013204422, 0.004353816, 7.761388, 0.14784424, 0.15963696])  # 测试原图的方差

        def get_idx(channels):
            assert channels in [2, 4, 7]
            if channels == 7:
                return list(range(7))
            elif channels == 4:
                return list(range(4))
            elif channels == 2:
                return list(range(6))[-2:]

        mean_, std_ = mean[get_idx(self.channels)], std[get_idx(self.channels)]
        _t = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean_, std=std_)
        ])
        L = []
        for item in singleImages:
            img = Image.open(item)
            img = np.expand_dims(np.array(img), axis=2)
            L.append(img)
        image = np.concatenate(L, axis=-1)
        # img = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0)
        # for t, m, s in zip(img, mean_, std_):
        #         t.sub_(m).div_(s)
        img, _ = _t(image, image[:, :, 0])
        img = img.unsqueeze(0)
        img = img.cuda() if self.cuda else img

        self.net.eval()
        with torch.no_grad():
            outp = self.net(img)
            score = self.softmax(outp)
            pred = score.max(1)[1]

            saved_1 = pred.squeeze().cpu().numpy()
            saved_255 = 255 * saved_1

            cv2.imwrite('test_output/{}_(1).png'.format(outName), saved_1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite('test_output/{}_(255).png'.format(outName), saved_255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def init_logfile(self):
        self.vallosslog = open(os.path.join(self.result_dir,'valloss.csv'),'w')
        self.vallosslog.writelines('epoch,loss\n')   

        self.valallacclog = open(os.path.join(self.result_dir,'valacc.csv'),'w')
        self.valallacclog.writelines('epoch,acc\n')

        self.trainlosslog = open(os.path.join(self.result_dir,'trainloss.csv'),'w')
        self.trainlosslog.writelines('epoch,loss\n')

        self.trainallacclog = open(os.path.join(self.result_dir,'trainacc.csv'),'w')
        self.trainallacclog.writelines('epoch,acc\n')

        self.precisionlog = open(os.path.join(self.result_dir,'presion.csv'),'w')
        self.precisionlog.writelines('epoch,precision\n')

        self.recalllog = open(os.path.join(self.result_dir,'recall.csv'),'w')
        self.recalllog.writelines('epoch,recall\n')

        self.f1log = open(os.path.join(self.result_dir, 'f1.csv'),'w')
        self.f1log.writelines('epoch,f1\n')

    def close_logfile(self):
        self.vallosslog.close()
        self.valallacclog.close()
        self.trainlosslog.close()
        self.trainallacclog.close()
        self.precisionlog.close()
        self.recalllog.close()
        self.f1log.close()  

    def trainer(self):
        try:
            for _ in range(self.start_epoch):
                self.scheduler.step()
            
            for epoch in range(self.start_epoch, self.end_epoch):
                self.train(epoch)
                self.scheduler.step()
                self.save_state(epoch, '{}/{}-ep{}.pth'.format(self.result_dir,self.configs['net'], epoch))
                if (epoch+1)%self.valIterval == 0:
                    self.val(epoch)
            
        except KeyboardInterrupt:
            print('Saving checkpoints from keyboardInterrupt...')
            self.save_state(epoch, '{}/{}-kb_resume.pth'.format(self.result_dir,self.configs['net']))
        
        finally:
            self.writer.close()


        # self.save_pred()
    def visualize(self, img, label, pred):
        label = label.clone().squeeze(0).cpu().numpy()
        label = decode_segmap(label).transpose((2,0,1))
        label = torch.from_numpy(label).unsqueeze(0)
        label = label.cuda() if self.cuda else label

        pred = pred.clone().squeeze(0).cpu().numpy()
        pred = decode_segmap(pred).transpose((2,0,1))
        pred = torch.from_numpy(pred).unsqueeze(0)
        pred = pred.cuda() if self.cuda else pred

        vis = torch.cat([self.denorm(img), label.float(), pred.float()], dim=0)
        vis_cat = vutils.make_grid(vis,nrow=3,padding=5,pad_value=0.8)
        return vis_cat
    

    def denorm(self, x):
        mean_ = torch.Tensor(mean).view(3,1,1)
        std_ = torch.Tensor(std).view(3,1,1)
        mean_ = mean_.cuda() if self.cuda else mean_
        std_ = std_.cuda() if self.cuda else std_
        out = x * std_ + mean_
        out = out / 255.
        return out.clamp_(0,1)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config = get_yaml('ConfigFiles/config-deeplab.yaml')
    print(config)

    solver = Solver(config)

    solver.test(['Data/predict/1-C11/1_2.tif',
                 'Data/predict/2-C12_real/1_2.tif',
                 'Data/predict/3-C12_imag/1_2.tif',
                 'Data/predict/4-C22/1_2.tif',
                 'Data/predict/5-alpha/1_2.tif',
                 'Data/predict/6-anisotropy/1_2.tif',
                 'Data/predict/7-entropy/1_2.tif'])

    # solver.trainer()
