import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.transforms as _transform
import torch
try:
    import transform as T
except:
    import data_utils.transform as T

traindir = "train"
testdir = "test"
imagedir = 'images'
labeldir = 'labels_0-1'

channel_list = ['1_C11', '2_C12_real', '3_C12_imag', '4_C22', '5_alpha', '6_anisotropy', '7_entropy' ]
trainfile = r'E:\XJG\All_net\Net\Data\train\labels_0-1\name.txt'
testfile = r'E:\XJG\All_net\Net\Data\test\labels_0-1\name.txt'


'''
train:                                      test:
1_C11 : 0.020159062 0.20009075                      0.01998466 0.008748512
2_c12_real: -2.9787574e-05 0.017613683              -3.127454e-05 0.0013148686
3_c12_imag: -7.190412e-05 0.020240089               -7.972777e-05 0.0013129638
4_c22 : 0.0027757809 0.005845128                    0.0027730444 0.00074990257
5_alpha : 15.893365 6.5137076                       15.987696 6.592051
6_entropy : 0.53359926 0.14960644                   0.53549945 0.15119201
'''   

mean_train = np.array([0.01308804, -1.1662938e-05, -3.957561e-05, 0.0026290491, 20.119514, 0.6648699, 0.62416124])
std_train = np.array([0.21623433, 0.01678721, 0.012647983, 0.005536756, 7.823991, 0.14901447, 0.16086595])

mean_test = np.array([0.013258977, 1.6523516e-05, -4.353957e-05, 0.0026426313, 19.974094, 0.66738826, 0.6236491])
std_test = np.array([0.12094154, 0.025448063, 0.014785762, 0.009304555, 7.4105687, 0.14177753, 0.15515509])

def get_idx(channels):
    assert channels in [2,4,7]
    if channels == 7:
        return list(range(7))
    elif channels == 4:
        return list(range(4))
    elif channels == 2:
        return list(range(7))[-2:]

def getTransform(train=True, channel_idx=[0,1,2,3]):
    if train:
        transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean_train[channel_idx],std=std_train[channel_idx])
            ]
        )
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean_test[channel_idx],std=std_test[channel_idx])
        ])
    return transform

_transform_test = _transform.Compose([
            _transform.ToTensor(),
            _transform.Normalize(mean=mean_test, std=std_test)
        ])


class semData(Dataset):
    def __init__(self, train=True, root='./Data', channels = 4, transform=None, selftest_dir=None): #modified here!
        self.train = train
        self.root = root
        self.dir = traindir if self.train else testdir
        if selftest_dir is not None:
            self.dir = selftest_dir
        self.channels = channels
        self.c_idx = get_idx(self.channels)
        if selftest_dir is not None: #modified here!
            self.file = os.path.join(self.root,selftest_dir,'labels_0-1/name.txt')
        else:
            self.file = trainfile if train else testfile

        self.img_dir = os.path.join(self.root, self.dir, imagedir)
        self.label_dir = os.path.join(self.root, self.dir, labeldir)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = getTransform(self.train, self.c_idx)
        
        self.data_list = pd.read_csv(self.file).values
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,index):
        L = []
        lbl_name = self.data_list[index][0] # 1081.png
        p = lbl_name.split('.')[0]          # 1081
        for k in self.c_idx:
            img_path = p + '.tif'
            img_path = os.path.join(self.img_dir, channel_list[k],img_path)
            img = Image.open(img_path)
            img = np.expand_dims(np.array(img),axis=2)
            L.append(img)

        # image = cv2.imread(os.path.join(self.root,img_path), cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.float32(image)
        image = np.concatenate(L, axis=-1)
        label = cv2.imread(os.path.join(self.label_dir, lbl_name), cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + p + " " + lbl_name + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        
        return {
            'X':image,
            'Y':label,
            'path': lbl_name       
        }
    
    def TestSetLoader(self,root='./data/test1',file='test.txt'):
        l = pd.read_csv(os.path.join(root,file)).values
        for i in l:
            filename = i[0]
            path = os.path.join(root, filename)
            image = Image.open(path)
            image = _transform_test(image)
            yield filename,image

if __name__ == "__main__":
    trainset = semData(train=False, channels=6, transform=None, selftest_dir='test')
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        img = data['X']
        label = data['Y']
        path = data['path']
        print(img.size(),label.max())



    

