

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
        
    def inc(self, numOccur):
        self.count += numOccur
    
    def disp(self, ind=1):
        print ' '*ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)
            

def createTree(dataSet, minSup=1):
    headerTable = {}
    #第一次遍历数据集；统计每个元素项出现的频度
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
            
    #移除不满足最小支持度的元素项    
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])
    
    
    freqItemSet = set(headerTable.keys())
    
    if len(freqItemSet) == 0:
        return None, None
    
    #保存计数值；及指向每种类型第一个元素项的指针
    #{'s': 3, 'r': 3, 't': 3, 'y': 3, 'x': 4, 'z': 5}
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # item:[个数，指针列表]    
    #{'s': [3, None], 'r': [3, None], 't': [3, None], 'y': [3, None], 'x': [4, None], 'z': [5, None]}
    retTree = treeNode('Null Set', 1, None)
    
    #根据全局频率对每个事务中的元素进行排序
    for tranSet, count in dataSet.items():
        localD = {}        
        #筛选项集中，过滤非频繁项，然后，对频繁项进行排序
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]            
            #使用排序后的频率项集，构建FP树
            #orderedItem:元素个数排序列表 ['z', 'x', 'y', 's', 't']
            #retTree:返回的结果树
            #headerTable:头指针列表 {'s': [3, None], 'r': [3, None], 't': [3, None], 'y': [3, None], 'x': [4, None], 'z': [5, None]}
            #count:当前项集个数 1
            updateTree(orderedItems, retTree, headerTable, count)
            
    return retTree, headerTable

#生成FP树
def updateTree(items, inTree, headerTable, count):
    #树上是否包含此节点
    if items[0] in inTree.children:
        #已经包含节点，增加节点树加count
        inTree.children[items[0]].inc(count)
    else:
        #不包含节点，创建树节点，更新头指针列表
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    #对剩下的元素项迭代调用updateTree
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

#更新头指针链表
def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode           

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat
    
    
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

#迭代获取节点的完整路径
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

#给定元素生成条件模式基
def findPrefixPath(basePat, treeNode):
    #模式基字典
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


#递归查找频繁项集的mineTree函数
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    #从头指针表的低端开始,从小到大
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        #创建条件模式基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #从条件模式基，来构建条件模式树
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            print 'conditional tree for: ',newFreqSet
            myCondTree.disp(1)
            #挖掘条件FP树
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


if __name__ == "__main__":
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    print(initSet)
    myFPtree, myHeaderTab = createTree(initSet, 3)    
    myFPtree.disp()    
    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)
    
    """
    condPats = findPrefixPath('x', myHeaderTab['x'][1])
    print(condPats)
    condPats = findPrefixPath('r', myHeaderTab['r'][1])
    print(condPats)
    condPats = findPrefixPath('t', myHeaderTab['t'][1])
    print(condPats)    
    
    
    rootNode = treeNode('pyramid', 9, None)
    rootNode.children['eye'] = treeNode('eye', 13, None)
    rootNode.children['phoenix'] = treeNode('phoenix', 3, None)    
    rootNode.disp()
    """


