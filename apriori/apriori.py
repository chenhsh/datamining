


def loadDataSet() :
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#生成初始的候选项集
def createC1(dataset) :
    C1 = []
    for transaction in dataset :
        for item in transaction :
            if not [item] in C1 :
                C1.append([item])
    C1.sort()
    # 对C1中每个项构建一个不变的集合,frozenset用户不能被修改
    return map(frozenset, C1)

#输入候选项集，返回频繁项集
def scanD(D, Ck, minSupport) :
    ssCnt = {}
    #统计所有候选项集出现的词数
    for tid in D:
        for can in Ck:
            if can.issubset(tid) :
                if ssCnt.has_key(can) : 
                    ssCnt[can] += 1
                else:
                    ssCnt[can] = 1
                    
    numItems = float(len(D))
    retList = []
    supportData = {}
    #计算所有项集的支持度
    for key in ssCnt : 
        support = ssCnt[key] / numItems
        if support >= minSupport :
            retList.insert(0, key)
        supportData[key] = support
        
    #返回频繁项集
    return retList, supportData

#频繁项集列表，项集元素个数;返回构建的候选项集，每个项集有k元素
#这种方法只是为了由k的频繁项集，生成k+1的候选项集
#没有考虑向下闭合特性
def aprioriGen(Lk, k) :
    retList = []
    lenLk = len(Lk)
    #前k-2个项相同时，将两个集合合并
    for i in range(lenLk) :
        L1 = list(Lk[i])[:k-2]
        L1.sort()
        for j in range(i+1, lenLk) :
            L2 = list(Lk[j])[:k-2]
            L2.sort()
            if L1 == L2 :
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0) :
        Ck = aprioriGen(L[k-2], k)
        #扫描数据集，从Ck得到Lk
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

#频繁项集列表；包含那些频繁项集支持数据的字典；最小可信度阀值
#返回包含置信度的规则列表
#supportData是所有候选项集的支持度
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    #从1开始，元素大于2
    #range(0)是{0},{1}
    #range(1)是{1,3}
    for i in range(1, len(L)) :
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i>1):
                #频繁项集的元素超过2
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                #频繁项集的元素只有两个{1,3}，直接计算置信度
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

#计算置信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    #满足最小置信度规则列表
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

#生成候选规则集合
#参数：频繁项集；可以出现在规则右边的元素列表H
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

if __name__ == "__main__":
    """
    1，挖掘频繁项集
    2，挖掘关联规则
    """
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet)
    print(L)
    print(supportData)
    print("\n")
    rules = generateRules(L, supportData, minConf=0.5)
    print(rules)
    
    
    """
    #测试
    dataSet = loadDataSet()
    print(dataSet)
    
    C1 = createC1(dataSet)
    print(C1)
    
    D = map(set, dataSet)
    print(D)
    
    L1, suppData0 = scanD(D, C1, 0.5)
    
    print(L1)
    print(suppData0)
    """
    















