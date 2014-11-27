# -*- coding: UTF-8 –*-
#coding=utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from kNN import *
    
#讲文本记录转换为Numpy的解析程序
def file2matrix(filename):    
    fr = open(filename)
    #得到文本行数
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)    
    #创建返回的Numpy矩阵
    returnMat = zeros((numberOfLines,3))    
    classLabelVector = []
    index = 0
    #解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1        
    fr.close()
    return returnMat,classLabelVector

#----------------------------------------------------------------------
#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals
     
#----------------------------------------------------------------------
def datingClassTest():    
    hoRatio = 0.08
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], \
                                     datingLabels[numTestVecs:m], 5)
        print("the classifier came back with: %d, the realanseris: %d"\
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0            
    print("the total error rate is : %f " %(errorCount/float(numTestVecs)))    
    

if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(normMat[:,1], normMat[:,2],
               15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()
    
    datingClassTest()
    
    


