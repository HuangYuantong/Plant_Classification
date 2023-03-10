import cv2
import joblib
import numpy
from skimage import feature as ft
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Data_Load import Data_Load
from Defines import pca, HOG_PCA_size, is_preload, path, Clf_SVC, accuracy


# HOG特征提取
#######################################################
def HOG_detect(image_dataset):
    """输入所有图片
    计算所有图片的HOG特征（纹理特征）
    并降维后返回"""
    temp = ft.hog(cv2.cvtColor(image_dataset[0], cv2.COLOR_RGB2GRAY),
                  orientations=6, pixels_per_cell=[8, 8], cells_per_block=[3, 3])
    hist = numpy.zeros((len(image_dataset), temp.shape[0]), "float32")
    print('\n下面开始HOG特征提取')
    for i in tqdm(range(len(image_dataset))):
        gray = cv2.cvtColor(image_dataset[i], cv2.COLOR_RGB2GRAY)
        hist[i] = ft.hog(gray, orientations=6, pixels_per_cell=[8, 8], cells_per_block=[3, 3])
    return preprocessing.normalize(pca(hist, HOG_PCA_size), norm='l2')


# 获得特征并保存
#######################################################
# 计算HOG特征并保存
def Train_HOG(train_data, train_label, test_data):
    # 将train和test集合并到一起进行特征提取（train, test）
    all_image = numpy.concatenate((train_data, test_data))
    all_features = HOG_detect(all_image)
    # 保存时，将train和test集合分开
    joblib.dump((all_features[:len(train_data)], train_label, all_features[len(train_data):]),
                f'res/hog_features_pca={HOG_PCA_size}.pkl')


if __name__ == '__main__':
    # is_preload = False
    if not is_preload:
        train_data, train_label, test_data = Data_Load(path)
        Train_HOG(train_data, train_label, test_data)

    # 使用HOG特征进行预测
    train_data, train_label, test_data = joblib.load(f'res/hog_features_pca={HOG_PCA_size}.pkl')
    train_X, val_X, train_Y, val_Y = train_test_split(train_data, train_label, test_size=0.2, random_state=1)
    # 训练分类器、进行预测、计算正确率
    clf = Clf_SVC(train_X, train_Y)
    pred = clf.predict(val_X)
    accuracy(val_Y, pred)
