
from numpy import *
from numpy import linalg as la


def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


#欧式距离
def ecludSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))

#皮尔逊相关系数
def pearsSim(inA, inB):
    #inA是列向量，如果小于3个点，此时两个向量完全相关
    if len(inA) < 3:
        return 1.0;
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar = 0)[0][1]

#余弦相似度
def cosSim(inA, inB):
    #向量内积
    num = float(inA.T * inB)
    #向量范数的积
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num/denom)

#根据用户对其他物品的评分，估计他没有评分的评分值

#数据矩阵，用户编号，相似度计算方法，物品编号
def standEst(dataMat, user, simMeas, item):
    #取出一共的物品数量
    n = shape(dataMat)[1]
    #总相关性
    simTotal = 0.0
    #总评分
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        #用户没有评分的跳过
        if userRating == 0:
            continue
        #寻找两个用户都评级的物品
        overLap = nonzero(logical_and(dataMat[:,item].A>0, dataMat[:,j].A>0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

#数据矩阵，用户编号，相似度计算方法，物品编号
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    #建立对角矩阵
    Sig4 = mat(eye(4)*Sigma[:4])
    
    #利用U矩阵将物品转换到低维空间中
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    
    #在低维空间中进行相似性计算
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        print('the %d and %d similarity is: %f ' %(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal
    

#推荐
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    #寻找未评级的物品
    #返回满足(dataMat[user,:].A==0)这个条件的数组，
    #(array([0, 0]), array([3, 4]))
    #        行，行         列，列
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        #给没有评级的物品，估计一个值
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    #寻找前N个未评级物品
    return sorted(itemScores, key = lambda jj: jj[1], reverse=True)[:N]


if __name__ == "__main__":
    """
    myMat = mat(loadExData())

    print("ecludSim\n")
    print(ecludSim(myMat[:,0], myMat[:,4]))
    print(ecludSim(myMat[:,0], myMat[:,0]))
    
    print("cosSim\n")
    print(cosSim(myMat[:,0], myMat[:,4]))
    print(cosSim(myMat[:,0], myMat[:,0]))    
    
    print("pearsSim\n")
    print(pearsSim(myMat[:,0], myMat[:,4]))
    print(pearsSim(myMat[:,0], myMat[:,0]))  
    
    myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
    myMat[3,3]=2
    print(recommend(myMat, 2))
    print(recommend(myMat, 2, simMeas=ecludSim))
    print(recommend(myMat, 2, simMeas=pearsSim))
    """
    """
    U, Sigma, VT = la.svd(mat(loadExData2()))
    print(Sigma)
    Sig2 = Sigma ** 2
    print(Sig2)
    print(sum(Sig2))
    print(sum(Sig2) * 0.9)
    print(sum(Sig2[:2]))
    print(sum(Sig2[:3]))
    """
    myMat = mat(loadExData2())
    print(recommend(myMat, 1, estMethod=svdEst))
    print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))




