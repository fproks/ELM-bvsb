# -*- coding:utf-8 -*-
import numpy as np
import cv2
from sklearn import svm
import operator
from sklearn.metrics import accuracy_score
from Utils import *

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from PIL import Image

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()


class Algorithms():
    def __init__(self):
        utils = Utils()
        self.label_list = utils.get_Label_List()
        pass

    def get_SIFT_Features(self, image_path, num):  # 提取图片的SIFT特征
        '''
        :param image_path: 图像的路径
        :param num: 截取每张图片的SIFT的特征点数
        :return: SIFT特征数组，行数为num，列为128
        '''
        img = cv2.imread(image_path)
        # print(image_path, img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # SIFT特征计算
        sift = cv2.xfeatures2d.SIFT_create()
        key_point, description = sift.detectAndCompute(gray, None)
        # print(description)

        if num > len(description):
            print('图片' + image_path + '的SIFT特征点数小于' + str(num) + '个，该幅图像不予考虑！')
            return []
        else:
            return description[0:num]

    def get_HOG_Features(self, image_path):  # 提取图片的HOG特征
        '''
        :param image_path: 图片的完整路径
        :return: 返回该图片的HOG特征值
        '''
        img = cv2.imread(image_path)
        # print(image_path, img)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cell_size = (6, 6)

        num_cells_per_block = (2, 2)

        block_size = (num_cells_per_block[0] * cell_size[0],
                      num_cells_per_block[1] * cell_size[1])


        x_cells = gray_image.shape[1] // cell_size[0]
        y_cells = gray_image.shape[0] // cell_size[1]

        h_stride = 1

        # Vertical distance between blocks in units of Cell Size. Must be an integer and it must
        # be set such that (y_cells - num_cells_per_block[1]) / v_stride = integer.
        v_stride = 1

        # Block Stride in pixels (horizantal, vertical). Must be an integer multiple of Cell Size
        block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)

        # Number of gradient orientation bins
        num_bins = 9

        # Specify the size of the detection window (Region of Interest) in pixels (width, height).
        # It must be an integer multiple of Cell Size and it must cover the entire image. Because
        # the detection window must be an integer multiple of cell size, depending on the size of
        # your cells, the resulting detection window might be slightly smaller than the image.
        # This is perfectly ok.
        win_size = (x_cells * cell_size[0], y_cells * cell_size[1])


        # Set the parameters of the HOG descriptor using the variables defined above
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

        # Compute the HOG Descriptor for the gray scale image
        hog_descriptor = hog.compute(gray_image)
        return hog_descriptor.reshape((-1,))

    def make_model(self):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.vgg16(pretrained=False)  # 其实就是定位到第28层，对照着上面的key看就可以理解
        pre=torch.load(r"./vgg16-397923af.pth").to(device)
        # if torch.cuda.is_available():
        #     pre = torch.load(r'./vgg16-397923af.pth')
        # else:
        #     pre = torch.load(r'./vgg16-397923af.pth', map_location=torch.device('cpu'))
        model.load_state_dict(pre)

        # # 获取原始模型中去掉最后一个全连接层的classifier. 输出的是4096维特征
        # new_classifier = torch.nn.Sequential(*list(model.children())[-1][:5])
        # # 替换原始模型中的classifier.
        # model.classifier = new_classifier

        model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
        if torch.cuda.is_available():
            model.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
        return model

    # 特征提取
    def get_VGG_Features(self, model, imgpath):
        model.eval()  # 必须要有，不然会影响特征提取结果
        img = Image.open(imgpath)  # 读取图片
        # print('图像通道数：', len(img.getbands()))
        if len(img.getbands()) == 3:
            img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            tensor = img_to_tensor(img)  # 将图片转化成tensor
            tensor = tensor.cuda()
            tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)

            tensor = tensor.cuda()

            result = model(Variable(tensor))
            result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错
            # print(result_npy)

            return result_npy[0].tolist()  # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]
        else:
            return []


if __name__ == '__main__':
    algorithms = Algorithms()
    model = algorithms.make_model()
    for i in range(1, 9):
        feature = algorithms.get_VGG_Features(model, '../image/source/001.ak47/001_000' + str(1) + '.jpg')
        print(i, len(feature), feature)
