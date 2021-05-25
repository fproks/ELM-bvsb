import numpy as np
import os
import cv2
from typing import Tuple
import torch
import torch.nn
import torchvision.models as models
from Algorithms import Algorithms
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio

'''
https://bbs.csdn.net/topics/392551610
需要安装cuda ，安装带cuda 的pytorch
import torch
print(torch.cuda.is_available())
这个代码可以看cuda 是否可用
conda 安装带CUDA的pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
'''
IMAGE_SIZE = 244


class Pretreatment():
    @staticmethod
    def readGrayImageToDATA(rootPath: str) -> Tuple[list, list]:
        imageList = Pretreatment._getfileList(rootPath)
        dataList = []
        targetList = []
        for img in imageList:
            _data = cv2.imread(img)
            _data = cv2.resize(_data, (IMAGE_SIZE, IMAGE_SIZE))
            dataList.append(_data)
            filename = os.path.basename(img)
            targetList.append(int(filename.split("_")[0]))
        return dataList, targetList

    @staticmethod
    def _getfileList(rootPath) -> list:
        if os.path.isabs(rootPath) is False:
            rootPath = os.path.abspath(rootPath)
        assert os.path.exists(rootPath)
        if os.path.isdir(rootPath) is False:
            return []
        result = []
        for file in os.listdir(rootPath):
            file_path = os.path.join(rootPath, file)
            if os.path.isfile(file_path):
                result.append(file_path)
            else:
                result.extend(Pretreatment._getfileList(file_path))
        return result

    @staticmethod
    def featureExtractionVGG(imageList: list) -> list:
        algorithm = Algorithms()
        model = algorithm.make_model()
        img_to_tensor = transforms.ToTensor()
        resultList = []
        for img in imageList:
            tensor = img_to_tensor(img)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
            result = model(Variable(tensor))
            result_npy = result.data.cpu().numpy()
            resultList.append(result_npy[0].tolist())
        return resultList


print('start extract features by VGG16')
[a, b] = Pretreatment.readGrayImageToDATA("../image/source")
print('得到图像数据,开始特征提取')
data = Pretreatment.featureExtractionVGG(a)
print('特征提取完成，开始保存')
data = np.array(data)
target = np.array(b)
sio.savemat('vggdata.mat', {'data': data, 'target': target})
print("特征数据保存为vggdata.mat")
