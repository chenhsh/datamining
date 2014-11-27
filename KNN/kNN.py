# -*- coding: UTF-8 –*-
#coding=utf-8
from numpy import * 
import operator

global dataSet

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A','A','B','B']
    return group, labels


def  classify0(inX, dataSet, labels, k):    
    #1距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    #2 选择距离最小的k个点
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #3排序
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)        
    return sortedClassCount[0][0]
    
    

if __name__ == '__main__':
    group,labels = createDataSet()
    rs = classify0([0.5,0.5], group, labels, 3)
    print(rs)
    

