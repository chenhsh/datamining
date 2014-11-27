# -*- coding: UTF-8 –*-
#coding=utf-8
from numpy import *
from kNN import *
from os import listdir

#----------------------------------------------------------------------
def  img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i + j] = int(lineStr[j])
    fr.close()    
    return returnVect
        
#----------------------------------------------------------------------
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        filenameStr = trainingFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s'%filenameStr)
    
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        filenameStr = testFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorunderTest = img2vector('testDigits/%s'%filenameStr)
        #设置k值为：3
        classifierResult = classify0(vectorunderTest, \
                                     trainingMat, hwLabels, 3)
        
        if (classifierResult != classNumStr) :
            print("---------------error----------------")
            print('the classifier came back with:%d, the real anser is:%d %s'\
                          %(classifierResult, classNumStr, filenameStr))            
            errorCount += 1.0
        else:
            print('the classifier came back with:%d, the real anser is:%d %s'\
                                      %(classifierResult, classNumStr, filenameStr))                                    
            
    print('\nthe total number of errors is:%d' %errorCount)
    print('\nthe total error rate is : %f' %(errorCount/float(mTest)))
    
    
if __name__ == '__main__':
    #手写体识别测试
    handwritingClassTest()
    

