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
import matplotlib.pyplot as plt
import time     # 引入计时机制

time_start = time.time()


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

# 利用tensorboard的视觉模块实现神经网络训练过程的可视化
        self.writer = SummaryWriter(self.result_dir)
        with self.writer:
            if not self.resume:
                inp = torch.randn([1,self.channels,512,512]).cuda() if self.cuda else torch.randn([1,self.channels,512,512])
                self.writer.add_graph(self.net, inp)    # 在tensorboard中创建Graphs，Graphs中存放网络结构

    def save_state(self, epoch, path):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, path)

#  修改程序运行状态，使其被暂停后仍可接着跑
    def resume_state(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        print('Resume from epoch{}...'.format(self.start_epoch))

    def train(self, epoch):
        self.net.train()

        loss_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()


        for i, batch in enumerate(self.train_dataloader):
            img = batch['X'].cuda() if self.cuda else batch['X']
            label = batch['Y'].cuda() if self.cuda else batch['Y']
            path = batch['path']

            self.optimizer.zero_grad()
            outp = self.net(img)

            loss = self.criterion(outp, label)
            loss.backward()
            self.optimizer.step()

            score = self.softmax(outp)
            pred = score.max(1)[1]

            N = img.size()[0]
            
            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), label.cpu().numpy(), self.n_classes, self.ignore_index)
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

#  计算每一次迭代间隔的loss/acc/iou
            iou = sum(intersection_meter.val) / (sum(union_meter.val) + 1e-10)
            acc = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss = loss_meter.update(loss.item(), N)


#  logInfo生成日志信息表，“:.4f”为保留小数点后四位
            if (i+1)%self.logIterval == 0:
                logInfo = ("Train [{}/{}]\t"
                          "Loss {loss_meter.val:.4f}({loss_meter.avg:.4f})\t"
                          "Accuracy {acc:.4f}\t"
                          "IoU {iou:.4f}.").format(i+1, len(self.train_dataloader),loss_meter=loss_meter,acc=acc,iou=iou)
                print(logInfo)

#  计算每一次迭代的allacc/macc/miou
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        acc_class = intersection_meter.sum / (target_meter.sum +1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(acc_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum)+1e-10)

#  logInfo生成日志信息表，“:.4f”为保留小数点后四位
        logInfo = 'Train at epoch [{}/{}] : mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, self.end_epoch, mIoU, mAcc, allAcc)
        mIoU1 = ('{:.4f}'.format(mIoU))
        mAcc1 = ('{:.4f}'.format(mAcc))
        allAcc1 = ('{:.4f}'.format(allAcc))
        Loss1 =  ('{:.4f}'.format(loss))
        file_hand = open('train.txt', mode="a+")
        file_hand.write(Loss1 + ',' +mIoU1 + ',' + mAcc1 + ',' + allAcc1 + '\n')
        file_hand.close()
        print(logInfo)



# 在tensorboard中加入训练时的各类评价指标
        self.writer.add_scalar('train/loss', loss_meter.avg, epoch+1)
        self.writer.add_scalar('train/mIoU', mIoU, epoch+1)
        self.writer.add_scalar('train/mAcc', mAcc, epoch+1)
        self.writer.add_scalar('train/allAcc', allAcc, epoch+1)


    
    def val(self, epoch):
        
        self.net.eval()
        with torch.no_grad():
            loss_meter = AverageMeter()
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            correct = []
            preds = []
            prob = []

            for i, batch in enumerate(self.val_dataloader):
                img = batch['X'].cuda() if self.cuda else batch['X']
                label = batch['Y'].cuda() if self.cuda else batch['Y']
                path = batch['path']

                outp = self.net(img)
                loss = self.criterion(outp, label)
                score = self.softmax(outp)      
                pred = score.max(1)[1]
                
                correct.extend(label.reshape(-1).cpu().numpy())
                preds.extend(pred.reshape(-1).cpu().numpy())
                
                N = img.size()[0]
                
                intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), label.cpu().numpy(), self.n_classes, self.ignore_index)
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

#  计算每一次迭代间隔的loss/acc/iou
                iou = sum(intersection_meter.val) / (sum(union_meter.val) + 1e-10)
                acc = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
                loss = loss_meter.update(loss.item(), N)
                
                if (i+1)%self.logIterval == 0:
                    logInfo = ("Val [{}/{}]\t"
                            "Loss {loss_meter.val:.4f}({loss_meter.avg:.4f})\t"
                            "Accuracy {acc:.4f}\t"
                            "IoU {iou:.4f}.").format(i+1, len(self.val_dataloader),loss_meter=loss_meter,acc=acc,iou=iou)
                    # vis = self.visualize(img, label, pred)
                    # self.writer.add_image(path[0].split('.')[0],vis,epoch+1)
                    print(logInfo)
            
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            acc_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(acc_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum)+1e-10)
        #    precision = precision_score(correct, preds)   # “precision” 准确率 ， "1-precision" 误检率  --找的对
        #    recall = recall_score(correct, preds)         # “recall” 召回率 ， "1-recall" 漏检率  --找的全
        #    f1 = f1_score(correct, preds)                 # F1分数可以看作是模型精确率和召回率的一种调和平均
            
            if mIoU > self.best_miou:
                self.best_miou = mIoU
                self.save_state(epoch, '{}/{}-{:.4f}-ep{}.pth'.format(self.result_dir,self.configs['net'], mIoU, epoch))
                self.save_pred()

          # logInfo = 'Val at epoch [{}/{}] : mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.\n'.format(epoch+1, self.end_epoch, mIoU, mAcc, allAcc)
            logInfo = 'Val at epoch [{}/{}] : mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, self.end_epoch, mIoU, mAcc, allAcc)
          # logInfo += '1-precision : {:.4f} | 1-recall : {:.4f} | f1 : {:.4f}.'.format(1-precision, 1-recall, f1)
            mIoU2 = ('{:.4f}'.format(mIoU))
            mAcc2 = ('{:.4f}'.format(mAcc))
            allAcc2 = ('{:.4f}'.format(allAcc))
            Loss2 = ('{:.4f}'.format(loss))
            file_hand = open('test.txt', mode="a+")
            file_hand.write(Loss2 + ',' +mIoU2 + ',' + mAcc2 + ',' + allAcc2 + '\n')
            file_hand.close()
            print(logInfo)

            for c in range(self.n_classes):
                logInfo = 'Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(c, iou_class[c], acc_class[c])
                self.writer.add_scalar('val/class{}_iou'.format(c), iou_class[c], epoch+1)
                self.writer.add_scalar('val/class{}_acc'.format(c), acc_class[c], epoch+1)
                print(logInfo)

# 在tensorboard中加入训练时的各类评价指标
        self.writer.add_scalar('val/loss', loss_meter.avg, epoch+1)
        self.writer.add_scalar('val/mIoU', mIoU, epoch+1)
        self.writer.add_scalar('val/mAcc', mAcc, epoch+1)
        self.writer.add_scalar('val/allAcc', allAcc, epoch+1)
    #    self.writer.add_scalar('val/precision', precision, epoch+1)
    #    self.writer.add_scalar('val/recall', recall, epoch+1)
    #    self.writer.add_scalar('val/f1', f1, epoch+1)
    
    def test(self, prefix):
        self.net.eval()
        with torch.no_grad():
            for n, img in self.trainSet.TestSetLoader():
                img = img.unsqueeze(0)
                img = img.cuda() if self.cuda else img
                out_p = self.net(img)
                outp = out_p.max(1)[1]
                outp = outp.clone().squeeze(0).cpu().numpy()
                outp = decode_segmap(outp)

                outp = torch.from_numpy((outp).transpose((2,0,1)))
                vutils.save_image(outp,os.path.join('./test_output','{}_{}'.format(prefix,n)))
        
    def init_logfile(self):
        self.vallosslog = open(os.path.join(self.result_dir,'valloss.csv'),'w')
        self.vallosslog.writelines('epoch,loss\n')   

        self.valallacclog = open(os.path.join(self.result_dir,'valacc.csv'),'w')
        self.valallacclog.writelines('epoch,acc\n')

        self.trainlosslog = open(os.path.join(self.result_dir,'trainloss.csv'),'w')
        self.trainlosslog.writelines('epoch,loss\n')

    #    self.trainallacclog = open(os.path.join(self.result_dir,'trainacc.csv'),'w')
    #    self.trainallacclog.writelines('epoch,acc\n')


    #    self.precisionlog = open(os.path.join(self.result_dir,'presion.csv'),'w')
    #    self.precisionlog.writelines('epoch,precision\n')

    #    self.recalllog = open(os.path.join(self.result_dir,'recall.csv'),'w')
    #    self.recalllog.writelines('epoch,recall\n')

    #    self.f1log = open(os.path.join(self.result_dir, 'f1.csv'),'w')
    #    self.f1log.writelines('epoch,f1\n')


    def save_pred(self):
        self.net.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                img = batch['X'].cuda() if self.cuda else batch['X']
                label = batch['Y'].cuda() if self.cuda else batch['Y']
                path = batch['path'][0].split('.')[0]

                outp = self.net(img)
                loss = self.criterion(outp, label)
                score = self.softmax(outp)      
                pred = score.max(1)[1]

                saved_1 = pred.squeeze().cpu().numpy()
                saved_255 = 255 * saved_1

# 保存预测图片，像素值为0&1
                cv2.imwrite('test_output/test_output_1/{}.png'.format(path),saved_1,[int(cv2.IMWRITE_JPEG_QUALITY),95])
# 保存预测图片，像素值为0&255
                cv2.imwrite('test_output/test_output_255/{}.png'.format(path),saved_255,[int(cv2.IMWRITE_JPEG_QUALITY),95])

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
    os.environ["CUDA_VISIBLE_DEVICES"]="0"  # 调用GPU，当只有一块GPU时，应改为“0”
    config = get_yaml('ConfigFiles/config-deeplab.yaml')  # 调用配置文件
    print(config)

    solver = Solver(config)
    solver.trainer()
    # solver.test(prefix='end')

    time_end = time.time()
    print('totally cost', time_end - time_start)