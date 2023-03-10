import numpy
import torch.utils.data
import torchvision

import os
from PIL import Image
from tqdm import tqdm

image_depth = 3
image_size = 256


def Data_Load(path):
    """读取给定路径下的train、test文件夹内所有图片
    返回三个numpy，依次为：train所有图片、train所有标签、test所有图片"""
    # Train
    print('train文件夹文件读取中……')
    transforms = torchvision.transforms.Compose([
        # 缩放图像以创建 image_size x image_size 的新图像
        torchvision.transforms.Resize((image_size, image_size))])
    # 数据读取
    dataset = torchvision.datasets.ImageFolder(path + '/train', transforms)
    # 转化为numpy
    train_dataset = numpy.array([numpy.array(image) for image, _ in tqdm(dataset)])
    train_label = numpy.array([label for _, label in tqdm(dataset)])
    # Test
    print('test文件夹文件读取中……')
    files = os.listdir(path + '/test')
    test_dataset = list()
    for i in tqdm(files):
        image = Image.open(path + '/test/' + i)
        image = image.resize((image_size, image_size), Image.ANTIALIAS)
        test_dataset.append(numpy.array(image))
    test_dataset = numpy.array(test_dataset)

    print(f'数据加载完成，train图片数量为：{len(dataset)}')
    print(f'train_dataset: {train_dataset.shape},train_label: {train_label.shape}\ntest_dataset: {test_dataset.shape}')
    return train_dataset, train_label, test_dataset


def Data_Load_CNN(path, validation_size=0.2):
    """从测试图片目录下读取所有文件，返回CNN网络的数据集train_dataset、val_dataset
    path：源根目录，validation_size：验证集比例（默认20%）
    dtype：返回numpy（[n,size,size,depth]0~255整数）或tensor（[n,depth，size,size]0~1浮点数）"""
    transforms = torchvision.transforms.Compose([
        # 缩放图像以创建 image_size x image_size 的新图像
        torchvision.transforms.Resize((image_size, image_size)),
        # torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Train
    print('train文件夹文件读取中……')
    dataset = torchvision.datasets.ImageFolder(path + '/train', transforms)
    # 训练集、验证集划分
    val_size = int(validation_size * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f'数据加载完成，train图片数量为：{len(dataset)}')

    # Test
    files = os.listdir(path + '/test')
    test_dataset = list()
    for i in tqdm(files):
        image = transforms(Image.open(path + '/test/' + i))
        test_dataset.append(image)
    print(f'test图片数量为：{len(test_dataset)}')
    return train_dataset, val_dataset, test_dataset


number2label = {0: 'Black-grass',
                1: 'Charlock',
                2: 'Cleavers',
                3: 'Common Chickweed',
                4: 'Common wheat',
                5: 'Fat Hen',
                6: 'Loose Silky-bent',
                7: 'Maize',
                8: 'Scentless Mayweed',
                9: 'Shepherds Purse',
                10: 'Small-flowered Cranesbill',
                11: 'Sugar beet'}

if __name__ == '__main__':
    Data_Load('all_data')
