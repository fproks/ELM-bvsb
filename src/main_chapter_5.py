# -*- coding:utf-8 -*-
u'''
Created on 2021年4月19日
@author: xianyu
@description：主程序，训练流程实现
'''
__author__ = 'xianyu'
__version__ = '1.0.0'
__company__ = u'STDU'
__updated__ = '2021-04-19'


from Algorithms import *
from Utils import *

import matplotlib.pyplot as plt
import time

import sys

# 定义全局变量
num_SIFT = 32   # 每张图片中提取的SIFT特征数，目前取8
m = 64  # SIFT特征降维后的维数，必须为8的倍数，目前取32

# 修改RSM随机抽取的维数
len_HOG = 3000
len_VGG = 900

ratio_BvSB = 0.05
threshold_BvSB = 0.5

ratio_marked_unmarked = 0.5

image_batch = 200

feature_type = 'vgg'    # 可以改成'sift'或'hog'，程序便运行相应的特征提取算法

if __name__ == '__main__':

    # 记录程序开始运行的时间
    start_time = time.time()

    # 导入所需的工具和算法包
    utils = Utils()
    algorithms = Algorithms()

    # 创建marked和unmarked文件夹下的图片图片，0.7是两个文件夹中每一类图片中的分配比例
    utils.generate_Image(ratio_marked_unmarked)

    # 获取marked和unmarked文件夹下的图片名称
    utils.init_Image_Weight()
    image_names_marked_select, image_names_unmarked = utils.get_Image(image_batch)
    if image_names_marked_select == image_names_unmarked == 0:
        print('初始获取图片错误，请查证！')
        sys.exit()

    # print('image_names_marked_select = ', image_names_marked_select)
    # print('image_names_unmarked = ', image_names_unmarked)
    # print('num_all_img = ', num_all_img)

    num_unmarked_img_orig = len(image_names_unmarked)  # 记录unmarked文件夹下原始的图片数量
    num_loop = 1    # 记录训练的论数
    unchanged_times = 0  # 记录训练结果不变的次数，由此表征训练结果是否达到稳定
    threshold_satble = 10  # 训练结果不变的次数超过threshold_satble次，则证明训练已达稳定，训练终止
    accuracy_score_buf_1 = []  # 记录分类器1的精度
    accuracy_score_buf_2 = []  # 记录分类器2的精度
    moving_ratio = []

    model = algorithms.make_model()  # 创建VGG网络模型，使用官方训练好的网络进行特征提取

    # 打印程序现在的信息
    print('提取特征的方法是：', feature_type)
    print('RSM抽取的维数是：', len_VGG)
    print('BvSB取前', ratio_BvSB*100, '%的样本')

    while True:
        # 存储训练和测试的样本
        X_Train_1 = []
        X_Train_2 = []
        Y_Train = []
        X_Test = []
        Y_Test = []

        if feature_type == 'sift':
            print('------------------------------第' + str(num_loop) + '轮训练--------------------------------')
            print('正在提取SIFT特征，生成训练样本......')
            # 根据marked文件夹下的图像生成训练样本
            for i in image_names_marked_select:
                # 提取每幅图像的8个SIFT特征
                features_full = algorithms.get_SIFT_Features(i, num_SIFT)
                if len(features_full) > 0:  # 有些图像的SIFT特征点很少，小于8个，这种图像不予考虑
                    # 将8个SIFT特征每个进行抽样，降至32位；并组成1维的向量，长度为32x8=256
                    features_M_1 = []
                    for sift in features_full:
                        features_M_1 = features_M_1 + algorithms.RSM_SIFT(sift, m)
                    features_M_2 = []
                    for sift in features_full:
                        features_M_2 = features_M_2 + algorithms.RSM_SIFT(sift, m)
                    # print(i, len(features_M), features_M)
                else:
                    image_names_marked_select.remove(i)  # 在数组中移除该图像的名字
                    continue
                # 样本生成， X_Train, Y_train
                X_Train_1.append(features_M_1)
                X_Train_2.append(features_M_2)

                Y_Train.append(int(os.path.basename(i).split('_')[0]))
            print('训练样本生成完成！')

            # 根据unmarked文件夹下的图像生成测试样本
            print('正在提取SIFT特征，生成测试样本......')
            for i in image_names_unmarked:
                # 提取每幅图像的8个SIFT特征
                features_full = algorithms.get_SIFT_Features(i, num_SIFT)
                if len(features_full) > 0:  # 有些图像的SIFT特征点很少，小于8个，这种图像不予考虑
                    # 将8个SIFT特征每个进行抽样，降至32位；并组成1维的向量，长度为32x8=256
                    features_M = []
                    for sift in features_full:
                        features_M = features_M + algorithms.RSM_SIFT(sift, m)
                    # print(i, len(features_M), features_M)
                else:
                    image_names_unmarked.remove(i)  # 在数组中移除该图像的名字
                    continue
                # 样本生成， X_Train, Y_train
                X_Test.append(features_M)
                Y_Test.append(int(os.path.basename(i).split('_')[0]))
            print('测试样本生成完成！')

            # print(len(X_Train_1), len(X_Train_1[0]))
            # print(len(X_Train_2), len(X_Train_2[0]))
            # print(len(X_Test), len(X_Test[0]))
            # print(len(Y_Train))
            # print(len(Y_Test))

        if feature_type == 'hog':
            print('------------------------------第' + str(num_loop) + '轮训练--------------------------------')
            print('正在提取HOG特征，生成训练样本......')
            for i in image_names_marked_select:
            # 提取每幅图像的HOG特征
                features_full = algorithms.get_HOG_Features(i)
                features_out = algorithms.RSM_HOG(features_full, len_HOG)
                X_Train_1.append(features_out)
                features_out = algorithms.RSM_HOG(features_full, len_HOG)
                X_Train_2.append(features_out)
                Y_Train.append(int(os.path.basename(i).split('_')[0]))
            print('训练样本生成完成！')

            print('正在提取HOG特征，生成测试样本......')
            for i in image_names_unmarked:
            # 提取每幅图像的HOG特征
                features_full = algorithms.get_HOG_Features(i)
                features_out = algorithms.RSM_HOG(features_full, len_HOG)

                X_Test.append(features_out)
                Y_Test.append(int(os.path.basename(i).split('_')[0]))
            print('测试样本生成完成！')

        if feature_type == 'vgg':
            print('------------------------------第' + str(num_loop) + '轮训练--------------------------------')
            print('正在提取VGG特征，生成训练样本......')
            for i in image_names_marked_select:
                # 提取每幅图像的VGG特征
                # print(i)
                features_full = algorithms.get_VGG_Features(model, i)
                if len(features_full) != 0:
                    features_out = algorithms.RSM_VGG(features_full, len_VGG)
                    # print('len(features_full)', len(features_full), 'len(features_out)=', len(features_out))
                    X_Train_1.append(features_out)
                    features_out = algorithms.RSM_VGG(features_full, len_VGG)
                    X_Train_2.append(features_out)
                    Y_Train.append(int(os.path.basename(i).split('_')[0]))
            print('训练样本生成完成！')

            print('正在提取VGG特征，生成测试样本......')
            for i in image_names_unmarked:
                # 提取每幅图像的VGG特征
                features_full = algorithms.get_VGG_Features(model, i)

                if len(features_full) != 0:
                    features_out = algorithms.RSM_VGG(features_full, len_VGG)

                    X_Test.append(features_out)
                    Y_Test.append(int(os.path.basename(i).split('_')[0]))
            print('测试样本生成完成！')

        # 输入SVM进行训练和预测
        Y_pred_proba_1, Y_pred_proba_2, score_1, score_2 = algorithms.SVM_Training_And_Testing(
            np.array(X_Train_1), np.array(X_Train_2), np.array(Y_Train), np.array(X_Test), np.array(Y_Test))
        print('score_1 = ', score_1, 'score_2 = ', score_2)

        # 记录两个分类器的精度
        accuracy_score_buf_1.append(score_1)
        accuracy_score_buf_2.append(score_2)

        # BvSB算法输出前ratio的候选值
        diff_B_SB_1, diff_B_SB_2 = algorithms.BvSB(Y_pred_proba_1, Y_pred_proba_2, ratio_BvSB, threshold_BvSB)
        # print(Y_pred_proba_1, Y_pred_proba_2)
        # print(diff_B_SB_1, diff_B_SB_2)

        # 判断两个预测结果是否相同且是否正确
        # label_list = utils.get_Label_List()
        diff_B_SB_1_index = [i[0] for i in diff_B_SB_1]   # 获取BvSB得到的样本下标
        diff_B_SB_2_index = [i[0] for i in diff_B_SB_2]   # 获取BvSB得到的样本下标
        same_index = [x for x in diff_B_SB_1_index if x in diff_B_SB_2_index]  # 寻找相同的元素
        candidate_img = [image_names_unmarked[i] for i in same_index]  # 获取两个预测结果中相同的图片信息

        # for i in range(min(len(diff_B_SB_1), len(diff_B_SB_2))):
        #     if diff_B_SB_1[i][0] == diff_B_SB_2[i][0] and label_list[diff_B_SB_1[i][2]] ==
        #            label_list[diff_B_SB_1[i][2]] == Y_Test[diff_B_SB_1[i][0]]:
        #         candidate_img.append(image_names_unmarked[diff_B_SB_1[i][0]])
        print(candidate_img)

        # 判断是否还有待移动的图片，若超过10次没有图片可以移动，则证明训练已达稳定，训练终止
        if len(candidate_img) == 0:
            unchanged_times = unchanged_times + 1
        else:
            unchanged_times = 0
        # 训练终止的条件:训练达到稳定
        if unchanged_times >= threshold_satble:
            print('训练已达到稳定，结束训练！')
            break

        # 移动符合条件的unmarked图片到marked文件夹中，以继续训练
        utils.update_Image_Folder(candidate_img)

        # 重新获取marked和unmarked文件夹下的图片名称
        image_names_marked_select, image_names_unmarked = utils.get_Image(image_batch)
        if image_names_marked_select == image_names_unmarked == 0:
            print('获取图片错误，请查证')
            break

        # 训练终止的条件：超过85%的无标签样本被移动
        if (num_unmarked_img_orig-len(image_names_unmarked))/num_unmarked_img_orig > 0.85:
            print('超过85%的无标签样本被移动，结束训练！')
            break

        num_loop = num_loop + 1
        moving_ratio.append((num_unmarked_img_orig-len(image_names_unmarked))/num_unmarked_img_orig)
        print('{:.2%}的无标签样本被移动'.format((num_unmarked_img_orig-len(image_names_unmarked))/num_unmarked_img_orig))

        # 打印程序运行时间
        end_time = time.time()
        dtime = end_time - start_time
        print("程序运行时间：%.8s s" % dtime)  # 显示到微秒
        print()


    # 绘制精度变化曲线
    plt.figure(0)
    plt.plot(accuracy_score_buf_1)
    plt.title('classifier 1')
    plt.xlabel('times')
    plt.ylabel('accuracy')

    plt.figure(1)
    plt.plot(accuracy_score_buf_2)
    plt.title('classifier 2')
    plt.xlabel('times')
    plt.ylabel('accuracy')

    # 绘制无标签样本被移动比例图
    plt.figure(2)
    plt.plot(moving_ratio)
    plt.title('moving ratio')
    plt.xlabel('times')
    plt.ylabel('ratio')

    # 打印程序运行时间
    end_time = time.time()
    dtime = end_time - start_time
    print("程序运行时间：%.8s s" % dtime)  # 显示到微秒

    plt.show()

