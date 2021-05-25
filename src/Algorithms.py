# -*- coding:utf-8 -*-
u'''
Created on 2021年4月19日
@author: xianyu
@description：主程序中需要调用的若干算法函数
'''
__author__ = 'xianyu'
__version__ = '1.0.0'
__company__ = u'STDU'
__updated__ = '2021-04-19'

import random
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

        # Specify the parameters for our HOG descriptor

        # Cell Size in pixels (width, height). Must be smaller than the size of the detection window
        # and must be chosen so that the resulting Block Size is smaller than the detection window.
        cell_size = (6, 6)

        # Number of cells per block in each direction (x, y). Must be chosen so that the resulting
        # Block Size is smaller than the detection window
        num_cells_per_block = (2, 2)

        # Block Size in pixels (width, height). Must be an integer multiple of Cell Size.
        # The Block Size must be smaller than the detection window
        block_size = (num_cells_per_block[0] * cell_size[0],
                      num_cells_per_block[1] * cell_size[1])

        # Calculate the number of cells that fit in our image in the x and y directions
        x_cells = gray_image.shape[1] // cell_size[0]
        y_cells = gray_image.shape[0] // cell_size[1]

        # Horizontal distance between blocks in units of Cell Size. Must be an integer and it must
        # be set such that (x_cells - num_cells_per_block[0]) / h_stride = integer.
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

        # # Print the shape of the gray scale image for reference
        # print('\nThe gray scale image has shape: ', gray_image.shape)
        # print()
        #
        # # Print the parameters of our HOG descriptor
        # print('HOG Descriptor Parameters:\n')
        # print('Window Size:', win_size)
        #
        # print('Cell Size:', cell_size)
        # print('Block Size:', block_size)
        # print('Block Stride:', block_stride)
        # print('Number of Bins:', num_bins)
        # print()

        # Set the parameters of the HOG descriptor using the variables defined above
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

        # Compute the HOG Descriptor for the gray scale image
        hog_descriptor = hog.compute(gray_image)
        return hog_descriptor.reshape((-1,))

    def make_model(self):
        model = models.vgg16(pretrained=False)  # 其实就是定位到第28层，对照着上面的key看就可以理解
        # model = models.vgg16(pretrained=False).features[:28]  # 其实就是定位到第28层，对照着上面的key看就可以理解
        if torch.cuda.is_available():
            pre = torch.load(r'./vgg16-397923af.pth')
        else:
            pre = torch.load(r'./vgg16-397923af.pth', map_location=torch.device('cpu'))
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
            tensor = tensor.cuda()  # 如果只是在cpu上跑的话要将这行去掉
            # print(tensor.shape)  # [3, 224, 224]
            tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)

            # print(tensor.shape)  # [1,3, 224, 224]
            tensor = tensor.cuda()

            result = model(Variable(tensor))
            result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错
            # print(result_npy)

            return result_npy[0].tolist()  # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]
        else:
            return []

    def RSM_SIFT(self, features_in, part):  # SIFT特征的RSM
        '''
        :param features_in:  128维的SIFT特征向量
        :param part: SIFT特征降到维数，必须为8的整数倍
        :return: 将为后的SIFT特征向量
        '''

        cell_len = 8
        # print(features_in)
        if len(features_in) % cell_len == 0:
            index = np.arange(0, int(len(features_in) / cell_len))
            # print('index = ', index)
            random.shuffle(index)  # 随机打乱数组顺序

            sub_index = index[0:int(part / cell_len)]
            sub_index = np.sort(sub_index)

            features_out = [features_in[i * cell_len:i * cell_len + 8] for i in sub_index]  # 按照抽样的子空间索引并输出
            features_out = np.array(features_out).flatten().tolist()

            # print('sub_index: ')
            # print(list(sub_index), sep=",")
            # print('features_out: ')
            # print(features_out)

            return features_out
        else:
            print(str(part) + '必须为8的整数倍! 请查证，并重新输入')
            return 0

    def RSM_HOG(self, features_in, num):  # HOG特征的RSM算法
        index = np.arange(0, len(features_in))
        # print('index = ', index)
        random.shuffle(index)  # 随机打乱数组顺序

        sub_index = index[0:num]
        sub_index = np.sort(sub_index)

        features_out = [features_in[i] for i in sub_index]  # 按照抽样的子空间索引并输出
        return features_out

    def RSM_VGG(self, features_in, num):  # VGG特征的RSM算法
        index = np.arange(0, len(features_in))
        # print('len(features_in)=', len(features_in), 'num=', num)
        # print('index = ', index)
        random.shuffle(index)  # 随机打乱数组顺序

        sub_index = index[0:num]
        sub_index = np.sort(sub_index)

        features_out = [features_in[i] for i in sub_index]  # 按照抽样的子空间索引并输出
        return features_out

    def BvSB(self, Y_pred_proba_1, Y_pred_proba_2, ratio, threshold_BvSB):  # BvSB算法实现
        # 分类器1给出的预测值处理
        diff_B_SB_1 = []  # 二维数组，每行如下：[样本序号, best和second best的差值, best的下标]
        i = 0
        for proba_pic in Y_pred_proba_1:
            best_index = proba_pic.index(max(proba_pic))
            best_val = max(proba_pic)
            proba_pic[best_index] = 0  # 将最大值的部分清零，剩下的部分继续寻找最大值即为second best
            second_best_index = proba_pic.index(max(proba_pic))
            second_best_val = max(proba_pic)
            # print(best_index, best_val, second_best_index, second_best_val)
            # 获取best和second best的差值，按顺序存入diff_B_SB中，并存入best的下标
            diff_B_SB_1.append([i, (best_val - second_best_val), best_index])
            i = i + 1

        # print(diff_B_SB_1)
        diff_B_SB_1.sort(key=operator.itemgetter(1), reverse=True)  # 将差值由高到低进行排序
        diff_B_SB_1 = diff_B_SB_1[0:max(int(ratio * len(diff_B_SB_1)), 1)]  # 取前5%，要是少于20张，20x5%小于1，则取1
        diff_B_SB_1 = [x for x in diff_B_SB_1 if x[1] > threshold_BvSB]  # best-secondbest的差值取满足阈值的样本

        # 分类器2给出的预测值处理
        diff_B_SB_2 = []  # 二维数组，每行如下：[样本序号, best和second best的差值, best的下标]
        i = 0
        for proba_pic in Y_pred_proba_2:
            best_index = proba_pic.index(max(proba_pic))
            best_val = max(proba_pic)
            proba_pic[best_index] = 0  # 将最大值的部分清零，剩下的部分继续寻找最大值即为second best
            second_best_index = proba_pic.index(max(proba_pic))
            second_best_val = max(proba_pic)
            # print(best_index, best_val, second_best_index, second_best_val)
            # 获取best和second best的差值，按顺序存入diff_B_SB中，并存入best的下标
            diff_B_SB_2.append([i, (best_val - second_best_val), best_index])
            i = i + 1

        # print(diff_B_SB_2)
        diff_B_SB_2.sort(key=operator.itemgetter(1), reverse=True)  # 将差值由高到低进行排序
        diff_B_SB_2 = diff_B_SB_2[0:int(ratio * len(diff_B_SB_2))]  # 取前5%
        diff_B_SB_2 = [x for x in diff_B_SB_2 if x[1] > threshold_BvSB]  # best-secondbest的差值取满足阈值的样本

        return diff_B_SB_1, diff_B_SB_2

    def SVM_Training_And_Testing(self, X_train_1, X_train_2, Y_train, X_test, Y_test):
        # 分别利用X_train_1和X_train_2以及Y_train训练两个支持向量机分类器svm_classifier_1和svm_classifier_2
        svm_classifier_1 = svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovr', gamma=0.0001, probability=True)
        svm_classifier_1.fit(X_train_1, Y_train)
        # print('svm_classifier_1 = ', svm_classifier_1)
        svm_classifier_2 = svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovr', gamma=0.0001, probability=True)
        svm_classifier_2.fit(X_train_2, Y_train)
        # print('svm_classifier_2 = ', svm_classifier_2)

        # 分别使用svm_classifier_1和svm_classifier_2对测试集进行测试，输出每个样本对应各类的概率值
        Y_pred_proba_1 = svm_classifier_1.predict_proba(X_test)
        Y_pred_proba_2 = svm_classifier_2.predict_proba(X_test)
        Y_pred_proba_1 = Y_pred_proba_1.tolist()
        Y_pred_proba_2 = Y_pred_proba_2.tolist()

        Y_pred_1 = [self.label_list[i.index(max(i))] for i in Y_pred_proba_1]
        Y_pred_2 = [self.label_list[i.index(max(i))] for i in Y_pred_proba_2]
        # print('Y_pred_1 = ', Y_pred_1, 'Y_pred_2 = ', Y_pred_2)
        score_1 = accuracy_score(Y_test, Y_pred_1)
        score_2 = accuracy_score(Y_test, Y_pred_2)
        # print('score_1 = ', score_1, 'score_2 = ', score_2)

        # print('Y_pred', 'Y_test', 'Y_pred_proba')
        # for i in range(len(Y_test)):
        #     print('SVM1', Y_pred_proba_1[i].index(max(Y_pred_proba_1[i])), Y_test[i], Y_pred_proba_1[i])
        #     print('SVM2', Y_pred_proba_2[i].index(max(Y_pred_proba_2[i])), Y_test[i], Y_pred_proba_2[i])
        #     # print(Y_pred[i], Y_test[i])
        return Y_pred_proba_1, Y_pred_proba_2, score_1, score_2


if __name__ == '__main__':
    algorithms = Algorithms()
    model = algorithms.make_model()
    for i in range(1, 9):
        feature = algorithms.get_VGG_Features(model, '../image/source/001.ak47/001_000' + str(1) + '.jpg')
        print(i, len(feature), feature)