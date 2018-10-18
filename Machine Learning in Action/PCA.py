'''
时间： 2018/10/18
作者： 楼浩然
功能： Numpy 实现PCA 算法
伪码：
1. 去除平均值
2. 计算协方差矩阵
3. 计算协方差矩阵的特征值和特征向量
4. 将特征值从大到小进行排序
5. 保留最前面的N个特征向量
7. 将数据转换到上述N个特征向量构建的新空间中
'''

from numpy import *

def loadDataSet(filename, delim = "\t"):
    fr = open(filename);
    stringArr = [line.strip().split(delim) for line in fr.readlines()];
    dataArr = [map(float,line) for line in stringArr];
    return mat(dataArr);

def PCA(datamat,topNfeat = 99999999):
    meanVal = mean(datamat,axis = 0);
    mearRemoved = datamat - meanVal;  #去除平均值
    covMat = cov(mearRemoved,rowvar=0);   #求协方差
    eigVals,eigVects = linalg.eig(mat(covMat));  #求协方差矩阵的特征值和特征向量
    eigValInd = argsort(eigVals); #对协方差的特征值索引从小到大进行排序的
    eigValInd = eigValInd[:-(topNfeat+1):-1]; #对协方差的特征值索引从大到小进行排序的，取出前topN个
    redEigVects = eigVects[:,eigValInd]; #取出前N个特征向量
    lowDataMat = mearRemoved * redEigVects;
    reconmat = (lowDataMat * redEigVects.T) + meanVal;  # 将数据转换到新的坐标系下
    return lowDataMat,reconmat;