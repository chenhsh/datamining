
import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

#计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

#为给定数据集构建包含k个随机质心的集合
def randCent(dataSet, k):
    #n是dataSet有多少列
    #从每一列范围内随机选取一个点
    n = np.shape(dataSet)[1]
    #k行：随机选取k个点
    #n列：
    centroids = np.mat(np.zeros((k, n)))
    #构建质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ) #随机范围在整个集合的最小值和最大值之间        
        centroids[:,j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    
    #存储每个簇的分配结果
    clusterAssment = np.mat(np.zeros((m, 2)))
    #存储质心
    centroids = createCent(dataSet, k)
    
    #判断聚类是否达到稳定???可能会陷入死循环吗？不会陷入死循环，判断距离总是会稳定的
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            #寻找最近的质心
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        #print(centroids)
        #print("\n")
        #更新质心的位置
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0] .A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment

#二分K均值聚类
def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    #每个点的簇分配结果，以及平方误差
    clusterAssment = np.mat(np.zeros((m,2)))    
    
    #创建一个初始簇，把所有数据当作一个簇，初始质心为平均值    
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]    
    #缓存质心
    centList =[centroid0]
    
    #计算初始距离
    for j in range(m):
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
        
    while (len(centList) < k):
        #初始化最小SSE为无穷大
        lowestSSE = np.inf
        #尝试划分每一个簇
        for i in range(len(centList)):
            #将这个簇内的点当作数据集dataSet
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            #将数据集ptsInCurrCluster进行二分处理
            #返回两个质心，对应的误差值
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #保存这次划分的误差（1）
            sseSplit = np.sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            #剩余数据集的误差（2）
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            
            #sseSplit + sseNotSplit这两个之和作为本次划分的误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #更新簇的分配结果
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        #新的质心，添加到原来的质心列表中
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return np.mat(centList), clusterAssment


if __name__ == "__main__":
    datMat = np.mat(loadDataSet('testSet.txt'))
    #randCent(datMat, 2)
    
    myCentroids, clustAssing = kMeans(datMat, 4)
    print(myCentroids)
    
    #二分K均值
    datMat3 = np.mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(datMat3, 3)
    print(centList)




