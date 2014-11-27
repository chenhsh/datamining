# -*- coding: UTF-8 –*-
#coding=utf-8
import tree
import treePlotter

fr = open("lenses.txt","r")
lenses = [inst.strip().split("\t") for inst in fr.readlines()]

lensesLabels = ["age", "prescript", "astigmatic", "tearRate"]

lensesTree = tree.createTree(lenses, lensesLabels)

treePlotter.createPlot(lensesTree)


