import cv2
import joblib
import numpy
from skimage import feature as ft
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Data_Load import image_size, Data_Load
from Defines import pca, LBP_PCA_size, is_preload, path, Clf_SVC, accuracy


# LBP特征提取
#######################################################
def LBP_detect(image_dataset):
    """输入所有图片
    计算所有图片的特征
    并降维后返回"""
    radius = 1
    n_point = radius * 8
    hist = numpy.zeros((len(image_dataset), image_size), "float32")
    print('\n下面开始LBP特征提取')
    for i in tqdm(range(len(image_dataset))):
        gray = cv2.cvtColor(image_dataset[i], cv2.COLOR_RGB2GRAY)
        lbp = ft.local_binary_pattern(gray, n_point, radius, 'default')
        max_bins = int(lbp.max() + 1)
        hist[i], _ = numpy.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    return preprocessing.normalize(pca(hist, LBP_PCA_size), norm='l2')


# 获得特征并保存
#######################################################


# 计算LBP特征并保存
def Train_LBP(train_data, train_label, test_data):
    # 将train和test集合并到一起进行特征提取（train, test）
    all_image = numpy.concatenate((train_data, test_data))
    all_features = LBP_detect(all_image)
    # 保存时，将train和test集合分开
    joblib.dump((all_features[:len(train_data)], train_label, all_features[len(train_data):]),
                f'res/lbp_features_pca={LBP_PCA_size}.pkl')


if __name__ == '__main__':
    # is_preload = False
    if not is_preload:
        train_data, train_label, test_data = Data_Load(path)
        Train_LBP(train_data, train_label, test_data)

    # 使用LBP特征进行预测
    train_data, train_label, test_data = joblib.load(f'res/lbp_features_pca={LBP_PCA_size}.pkl')
    train_X, val_X, train_Y, val_Y = train_test_split(train_data, train_label, test_size=0.2, random_state=1)
    # 训练分类器、进行预测、计算正确率
    clf = Clf_SVC(train_X, train_Y)
    pred = clf.predict(val_X)
    accuracy(val_Y, pred)
