


import numpy as np
import matplotlib.pyplot as plt



def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    #求平均值
    meanVals = np.mean(dataMat, axis=0)    
    meanRemoved = dataMat - meanVals
    
    #计算协方差矩阵
    covMat = np.cov(meanRemoved, rowvar=0)
    #计算特征值，特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    #从小到大对N个值排序，可以得到topNfeat个最大的特征向量；用于对数据进行转换的矩阵
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    
    #矩阵利用N个特征将原始数据转换到新的空间
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat



if __name__ == "__main__":
    dataMat = loadDataSet('testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    print(np.shape(lowDMat))
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
    
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
    
    plt.show()
    
    
    
    
    














