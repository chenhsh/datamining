# -*- coding: UTF-8 –*-
#coding=utf-8
from math import log
import operator
import treePlotter

#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1        
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#待划分数据集，划分数据集的特征，需要返回的特征值
def splitDataSet(dataSet, axis, value):
    retDatSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDatSet.append(reduceFeatVec)
    return retDatSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1    
    for i in range(numFeatures):
        #创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        #计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i ,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            #计算最好的信息增益
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
    return bestFeature

#出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.iterkeys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel : {}}
    del(labels[bestFeat])
    #得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
        
def createDataSet():
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    #dataSet = [[1, 1, "yes"],
                   #[1, -1, "yes"],
                   #[1, 0, "no"],
                   #[0, 2, "no"],
                   #[0, 1, "no"]]    
    labels = ["no surfacing", "flippers"]
    return dataSet, labels

#使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    #将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel= classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

#使用pickle模块存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, "w")
    pickle.dump(inputTree, fw)
    fw.close()

def  grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    

if __name__ == "__main__":
    myDat, labels = createDataSet()
    #rs = calcShannonEnt(myDat)
    #rs = splitDataSet(myDat, 0, 1)
    #rs = chooseBestFeatureToSplit(myDat)   
    #myTree = createTree(myDat, labels)
    #print(myTree)
    
    myTree = treePlotter.retrieveTree(0)
    #myTree["no surfacing"][3] = "maybe"
    #treePlotter.createPlot(myTree)
    
    rs = classify(myTree, labels, [1,0])
    print(rs)
    
    rs = classify(myTree, labels, [1,1])
    print(rs)
    
    
    
    
    


