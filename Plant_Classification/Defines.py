import os
import numpy as np
import pandas

from sklearn.decomposition import PCA  # PCA降维

# XGBooster分类器
from xgboost import XGBClassifier
# 支持向量机
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, SVR
# 随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
# K近邻
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from Data_Load import number2label

path = 'all_data'  # 数据集位置
is_preload = False  # 是否读取预处理的数据

SIFT_BOW_size = 80  # Sift中,词袋大小(降维后的特征数量)设置
HOG_PCA_size = 60  # HOG降维后的特征数量设置
LBP_PCA_size = 80  # LBP降维后的特征数量设置

n_splits = 5  # K折交叉验证的K数


# 功能函数
#######################################################
# 降维，将一个图片的n维特征，提取出n个重要特征，减少特征的数量
def pca(hist, n):
    print('\nPCA降维中……')
    print(f'{hist.shape[1]}降至{n}维')
    pca = PCA(n_components=n)
    pca.fit(hist)
    hist_new = pca.transform(hist)  # 降维结果
    return hist_new


# 将多个特征，合并为一个特征
def merge(*args):
    return np.hstack(args)


# 正确率计算
def accuracy(actual, predict):
    # 计算正确的标签个数
    correct = (predict == actual).sum()
    total = len(actual)
    print(f'正确个数：{correct},总数：{total}, 正确率{correct / total:.6f}')
    return correct / total


def evaluate(actual, predict):
    """计算真实值和预测值间的精确率"""
    # 精确率 = 所有分类正确的正例/所有正例
    m_precision = metrics.precision_score(actual, predict, average="macro")
    print('m_precision: ', m_precision)
    # 召回率 = 提取出的正确信息条数 /样本中的信息条数
    m_recall = metrics.recall_score(actual, predict, average="macro")
    print('m_recall: ', m_recall)
    # F1 = 2 * (precision * recall) / (precision + recall)
    m_f1 = metrics.f1_score(actual, predict, average="macro")
    print('f1-score: ', m_f1)
    return m_precision, m_f1


# 生成test集的标签csv文档
def Make_CSV(predict_label, filename):
    """传入对test集图片的预测标签，生成并保存用于提交的csv文件"""
    # 将标签转换为植物名字
    predict_label = [number2label[i] for i in predict_label]
    # 保存到csv文件中
    files = os.listdir(path + '/test')
    file = pandas.DataFrame(columns=['file', 'species'], data=list(zip(files, predict_label)))
    file.to_csv(path_or_buf=filename, index=False)


# 分类器生成
#######################################################
def Clf_SVC(train_data, train_label):
    # SVC(核函数kernel='linear'时变为为SVM)
    clf = SVC(C=12, probability=True)  # default with 'rbf'
    print('选用分类器：', clf)
    clf.fit(train_data, train_label)
    return clf


def Clf_SVR(train_data, train_label):
    # SVR
    clf = OneVsRestClassifier(SVR(C=12))
    print('选用分类器：', clf)
    clf.fit(train_data, train_label)  # train_hist为特征，根据特征和标签训练样本
    return clf


def Clf_RandomForestClassifier(train_data, train_label):
    # 随机森林分类器
    clf = RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=40,
                                 bootstrap=False, oob_score=False, random_state=10)
    # n_estimators（树棵数）、bootstrap（有放回采样，会有37%左右不会被使用）
    print('选用分类器：', clf)
    clf.fit(train_data, train_label)
    return clf


def Clf_ExtraTreesClassifier(train_data, train_label):
    # ExtraTree分类器集合
    clf = ExtraTreesClassifier(n_estimators=400, max_features='sqrt', max_depth=50)
    print('选用分类器：', clf)
    clf.fit(train_data, train_label)
    return clf


def Cls_XGBooster(train_data, train_label):
    # xgboost
    clf = XGBClassifier(learning_rate=0.13, max_depth=5, n_estimators=300, nthread=10,
                        use_label_encoder=False, eval_metric='mlogloss')
    print('选用分类器：', clf)
    clf.fit(train_data, train_label)
    return clf


def Clf_KNN(train_data, train_label):
    # KNN
    clf = KNeighborsClassifier()
    print('选用分类器：', clf)
    clf.fit(train_data, train_label)
    return clf


__all__ = [path, is_preload, SIFT_BOW_size, HOG_PCA_size, LBP_PCA_size, n_splits,
           accuracy, merge, evaluate, Make_CSV,
           Clf_SVC, Clf_ExtraTreesClassifier, Clf_RandomForestClassifier, Cls_XGBooster]
