import sys
import cv2
import numpy
import os
from tqdm import tqdm

image_test = cv2.imread('data/train/Charlock/0edcd02cd.png')
# image_test = cv2.imread('data/train/Black-grass/0be707615.png')
# 绿色的HSV范围：色调-饱和度-明度
green_lower = numpy.array([35, 43, 46])  # 绿色下限为[35, 43, 46]
green_upper = numpy.array([77, 255, 255])  # 绿色上限为[77, 255, 255]
is_equalize = True
is_sharpening = False


def Image_Equalize(image):
    """将image彩色图像均衡化：对RGB通道分别均衡化后合并"""
    (B, G, R) = cv2.split(image)
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    # 合并每一个通道
    result = cv2.merge((B, G, R))
    return result


def Image_Sharpening(image):
    """对图像使用拉普拉斯算子进行锐化"""
    sharpen_op = numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 'float32')
    result = cv2.filter2D(image, cv2.CV_32F, sharpen_op)
    result = cv2.convertScaleAbs(result)
    return result


def Image_Split(mask, image, is_display=False):
    """将图像转换到HSV空间以筛选出图像中的绿色区域，其他区域置为0"""
    # 先将mask模糊
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    # mask转到HSV空间进行颜色分割
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    # 根据阈值找到对应颜色
    mask = cv2.inRange(mask, green_lower, green_upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    # 展示图片
    if is_display:
        print(image.shape)
        cv2.imshow("images", numpy.hstack([image, result]))
        cv2.waitKey(0)
        exit()
    return result


def Process_All(file_from, file_to, number_limit, is_sharpening, is_equalize, is_display=False):
    """处理并保存图片至新目录。
    file_from：源根目录，”file_to+file_from“：目标目录，number_limit：每个最底层文件夹下文件个数（None时为无限制）"""
    # 夹带私货……将后续要用到的res文件夹也先创建好
    if not os.path.exists('res'):
        os.makedirs('res')
    if not os.path.exists(file_from):
        print('未找到原始数据集，请完成以下2点操作以启动程序：\n'
              f'（1）请将原始数据集放置于当前路径下({sys.path[0]})\n'
              '（2）并将其根目录命名为data')
        exit()

    for cur_dir, sub_dir, files in tqdm(os.walk(file_from)):
        # 通过count计数最底层文件夹下文件个数
        count = 0
        for file_name in files:
            if ((number_limit is None) or (count < number_limit)) and file_name.endswith(('.png', '.jpg')):
                count += 1
                # 原路径：cur_dir + os.sep + file_name
                # 新路径：file_to + cur_dir + os.sep + file_name
                image = cv2.imread(cur_dir + os.sep + file_name)  # 从原地址处获取
                result = image
                # 进行图片处理
                # 均衡化
                if is_equalize:
                    result = Image_Equalize(result)
                # 锐化
                if is_sharpening:
                    result = Image_Sharpening(result)
                # 去除背景
                result = Image_Split(image, result, is_display)

                # 若新目录不存在，则需要先创建
                temp_file = file_to + cur_dir
                if not os.path.exists(temp_file):
                    os.makedirs(temp_file)
                cv2.imwrite(temp_file + os.sep + file_name, result)  # 保存入新文件夹下


def Process_All_StartUp():
    number_limit = input('你需要产生并保存一份处理后的图片数据吗？\n'
                         '  -----------------------------------\n'
                         '  *修改Process_All()的file_from、file_to参数以更改路径*\n'
                         '  -----------------------------------\n'
                         '> 查看处理效果、或若不需要请输入0\n'
                         '> 若需要限制每个文件夹内图片个数请输入一个正整数\n'
                         '> 若要处理并产生所有图片，请输入None\n'
                         '> 请输入数量限制：')
    if number_limit == '0':
        Image_Split(image_test, image_test, is_display=True)
    else:
        if number_limit == 'None':
            number_limit = None
            file_to = 'all_'
        else:
            number_limit = int(number_limit)
            file_to = 'small_'
        Process_All(file_from='data', file_to=file_to, number_limit=number_limit,
                    is_sharpening=is_sharpening, is_equalize=is_equalize, is_display=False)
        print('处理完成！')


if __name__ == '__main__':
    # result = Image_Equalize(image_test)
    # # result=Image_Split(image_test,result)
    # detector = cv2.xfeatures2d.SIFT_create()
    # kp, sift = detector.detectAndCompute(result, None)  # des是描述子 sift是关键点特征，每张图关键点不一样多，每个关键点是128维
    # ret = cv2.drawKeypoints(result, kp, result)
    # cv2.imshow('image', ret)
    # cv2.waitKey(0)
    # exit()

    # Image_Split(image_test, image_test, True)
    # Process_All_StartUp()

    Process_All(file_from='data', file_to='all_', number_limit=None,
                is_sharpening=is_sharpening, is_equalize=is_equalize, is_display=False)
