# -*- coding:utf-8 -*-
u'''
Created on 2021年4月19日
@author: xianyu
@description：主程序中需要调用的若干工具函数
'''
__author__ = 'xianyu'
__version__ = '1.0.0'
__company__ = u'STDU'
__updated__ = '2021-04-19'

import os
import shutil
import random


class Utils():
    def __init__(self):
        self.img_path = './image'
        self.img1 = '../image/img1.jpg'
        self.img_path_marked = '../image/marked/'
        self.img_path_unmarked = '../image/unmarked/'
        self.source_img_path = '../image/source/'
        self.image_names_marked_dic = []
        self.image_names_unmarked_dic = []
        self.image_names_marked_selected_dic = []
        pass

    # 创建marked和unmarked文件夹及其内部图片
    def generate_Image(self, ratio):
        '''
        :param ratio:创建比例，范围是0~1
        :return:
        '''
        if ratio < 1:
            # 清除 marked和unmarked 两个文件夹中的文件(删除并重新创建)
            # 删除文件夹
            if os.path.exists(self.img_path_marked):
                shutil.rmtree(self.img_path_marked)
            if os.path.exists(self.img_path_unmarked):
                shutil.rmtree(self.img_path_unmarked)
            # 创建问价夹
            os.mkdir(self.img_path_marked)
            os.mkdir(self.img_path_unmarked)

            # 复制self.source_img_path路径下ratio比例的图像到marked文件夹下，1-ratio比例的图像到unmarked文件夹下
            path_img_cource = [self.source_img_path + x for x in os.listdir(self.source_img_path)]
            # print(path_img_cource)
            for category_name in path_img_cource:
                num_img = len(os.listdir(category_name))
                # print(num_img)
                index = 0
                for file_name in os.listdir(category_name):
                    # print(file_name)
                    if file_name.split('.')[-1] == 'jpg':  # 必须是jpg文件才做处理
                        index = index + 1
                        if index < num_img * ratio:
                            shutil.copy(os.path.join(category_name, file_name),
                                        os.path.join(self.img_path_marked, file_name))
                        else:
                            shutil.copy(os.path.join(category_name, file_name),
                                        os.path.join(self.img_path_unmarked, file_name))
                        pass

        else:
            print('输入的参数值必须小于1，请查证！')

    def init_Image_Weight(self):
        '''
        准备图像数据，记录图片的名称和分类
        :return: True
        '''
        image_names_marked = [self.img_path_marked + x for x in os.listdir(self.img_path_marked)]
        image_names_unmarked = [self.img_path_unmarked + x for x in os.listdir(self.img_path_unmarked)]
        # num_all_img = len(image_names_marked) + len(image_names_unmarked)
        # print(image_names_unmarked, image_names_marked)

        self.image_names_marked_dic = [[i, 0] for i in image_names_marked]
        self.image_names_unmarked_dic = [[i, 0] for i in image_names_unmarked]

        return True

    # 获取用于训练的图片路径及其名称、权重值
    def get_Image(self, num_pic):
        '''
        :param num_pic: 输入要迭代的图片的数量
        :return: 带有权重值的图片信息路径信息
        '''

        if num_pic > len(self.image_names_marked_dic):
            print('要迭代的图片数量大于图片总量，请查证后在输入！')
            return 0, 0
        else:
            # image_names_marked = [self.img_path_marked+x for x in os.listdir(self.img_path_marked)]
            image_names_unmarked = [self.img_path_unmarked+x for x in os.listdir(self.img_path_unmarked)]
            # # num_all_img = len(image_names_marked) + len(image_names_unmarked)
            # # print(image_names_unmarked, image_names_marked)
            #
            # self.image_names_marked_dic = [(i, 0) for i in image_names_marked]
            # self.image_names_unmarked_dic = [(i, 0) for i in image_names_unmarked]
            # 随机抽取num_pic个样本
            self.image_names_marked_selected_dic = random.sample(self.image_names_marked_dic, int(num_pic))
            # print('self.image_names_marked_selected_dic = ', self.image_names_marked_selected_dic)
            # 对抽取的样本权重值加1
            for i in self.image_names_marked_dic:
                if i in self.image_names_marked_selected_dic:
                    if i[1] >= 9:
                        self.image_names_marked_dic.remove(i)
                        self.image_names_marked_selected_dic.remove(i)
                        # 后续可以加入图片的删除操作
                    else:
                        self.image_names_marked_dic[self.image_names_marked_dic.index(i)][1] = i[1] + 1

            print(self.image_names_marked_dic)

            # return image_names_marked, image_names_unmarked
            return [i[0] for i in self.image_names_marked_selected_dic], image_names_unmarked

    # 将unmarked文件夹中的图片移动到marked文件夹中
    def update_Image_Folder(self, candidate_img):
        '''
        :param candidate_img: 待移动的图片的名称数组
        :return: 无
        将未标记的图像变成已标记的图像后，更新图像文件夹内的图像位置
        '''
        image_names_to_move = [i.split('/')[-1] for i in candidate_img]
        for file in image_names_to_move:
            shutil.move(self.img_path_unmarked+file, self.img_path_marked+file)
            self.image_names_marked_dic.append([self.img_path_marked+file, 0])

    # 获得样本名称数组，与支持向量机输出的概率表进行对应
    # def get_Label_List(self, Y_test):
    #     label_list = list(set(Y_test))
    #     label_list.sort()
    #     return label_list
    # 获得样本名称数组，与支持向量机输出的概率表进行对应
    def get_Label_List(self):
        pic_name_folder = os.listdir(self.source_img_path)
        print('当前使用的样本名称为： ', pic_name_folder)
        label_list = [int(i.split('.')[0]) for i in pic_name_folder]
        return label_list


if __name__ == '__main__':
    utils = Utils()
    result = utils.init_Image_Weight()
    a, b = utils.get_Image(30)
    print(a)
    print(b)