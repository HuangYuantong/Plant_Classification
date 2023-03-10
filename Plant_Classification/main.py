import os
import numpy
from sklearn.model_selection import KFold
import joblib

from Image_Process import Process_All, is_equalize, is_sharpening

from SIFT import Train_Sift
from HOG import Train_HOG
from LBP import Train_LBP

from Data_Load import Data_Load
from Defines import SIFT_BOW_size, is_preload, path, Clf_SVC, accuracy, HOG_PCA_size, LBP_PCA_size, merge, \
    Clf_ExtraTreesClassifier, Clf_RandomForestClassifier, Cls_XGBooster, n_splits, evaluate, Make_CSV


def Step1_Get_Feature(is_preload=is_preload):
    """获取并计算train特征、train标签、test特征（特征为SIFT、HOG、LBP特征融合）
    is_preload=True则从已保存的文件直接读取特征"""
    # 如果is_preload=False，重新计算SIFT、HOG、LBP特征并保存
    if not is_preload:
        # 没有all_data文件夹则先进行一下图片预处理
        if not os.path.exists(path):
            Process_All(file_from='data', file_to='all_', number_limit=None,
                        is_sharpening=is_sharpening, is_equalize=is_equalize, is_display=False)
        train_data, train_label, test_data = Data_Load(path)
        # 计算所有特征并保存（train、test共同进行特征提取）
        Train_Sift(train_data, train_label, test_data)
        Train_HOG(train_data, train_label, test_data)
        Train_LBP(train_data, train_label, test_data)
    # 读取所有train、test集合图片的特征
    sift_train_data, train_label, sift_test_data = joblib.load(f'res/sift_features_bow={SIFT_BOW_size}.pkl')
    hog_train_data, _, hog_test_data = joblib.load(f'res/hog_features_pca={HOG_PCA_size}.pkl')
    lbp_train_data, _, lbp_test_data = joblib.load(f'res/lbp_features_pca={LBP_PCA_size}.pkl')

    # 特征融合
    train_features = merge(sift_train_data, hog_train_data, lbp_train_data)
    test_features = merge(sift_test_data, hog_test_data, lbp_test_data)
    return train_features, train_label, test_features


def Step2_Train_Classification(train_data, train_label):
    """选取多个分类器，并全部用输入的(data, label)训练好
    返回列表包含任意个训练好的分类器"""
    # 选取并训练分类器
    cls_list = list()
    cls_list.append(Clf_SVC(train_data, train_label))
    cls_list.append(Clf_RandomForestClassifier(train_data, train_label))
    cls_list.append(Clf_ExtraTreesClassifier(train_data, train_label))
    cls_list.append(Cls_XGBooster(train_data, train_label))
    return cls_list


def Step3_Predict(data_inputs, cls_list):
    """用传入的列表中的多个分类器，依次对传入的数据进行预测
    并集成选出最终预测，输出预测标签 (0~11)"""
    # 使用概率累加的方式进行分类器集成
    probability = cls_list[0].predict_proba(data_inputs)
    for cls in cls_list[1:]:
        probability = numpy.add(probability, cls.predict_proba(data_inputs))
    # 取出最大的概率作为其标签
    predict = numpy.argmax(probability, axis=1)
    return predict


if __name__ == '__main__':
    # 获取所有特征
    train_features, train_labels, test_features = Step1_Get_Feature(is_preload)
    # K折交叉验证
    ############################################################################
    # 存交叉验证数据
    accuracy_list = []
    precision_list = []
    f1_list = []
    # K折交叉验证
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=46)
    for train_index, val_index in kfold.split(train_features):
        print('==================================================')
        train_X, val_X = train_features[train_index], train_features[val_index]
        train_Y, val_Y = train_labels[train_index], train_labels[val_index]
        # 训练好所有分类器
        cls_list = Step2_Train_Classification(train_X, train_Y)
        # 用验证集计算正确率等参数
        predict = Step3_Predict(val_X, cls_list)
        temp1 = accuracy(val_Y, predict)
        temp2, temp3 = evaluate(val_Y, predict)
        # 储存画图
        accuracy_list.append(temp1)
        precision_list.append(temp2)
        f1_list.append(temp3)
        # 使用模型进行预测
        predict = Step3_Predict(test_features, cls_list)
        Make_CSV(predict, f'submission_features{temp2}.csv')

    # 画图
    ############################################################################
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.plot(range(1, n_splits + 1), accuracy_list, '-^', label="accuracy")
    plt.plot(range(1, n_splits + 1), precision_list, '-.o', label="precision")
    plt.plot(range(1, n_splits + 1), f1_list, ':.', label="L1得分")

    plt.xticks(range(1, n_splits + 1))
    plt.ylim(0.5, 1)
    plt.xlabel('次数', loc='right')
    plt.ylabel('得分（0-1）', loc='top')
    plt.title("预测准确率等得分")
    plt.savefig(fname="预测准确率等得分.png", dpi=440)
    plt.legend()
    plt.show()
