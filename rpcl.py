# coding=utf-8
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


# 计算两个向量的距离，用的是欧几里得距离
def distEclud(vecA, vecB):

    return sqrt(sum(power(vecA - vecB, 2)))


# 随机生成初始的质心（ng的课说的初始方式是随机选K个点）
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.array(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(array(dataSet)[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k)
    return centroids

def emptyRivalList(k):
    rivalList = []
    for i in range(k):
        rivalList.append([])
    return rivalList

def rpcl(dataSet, k, penalty, initcenter,distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.array(zeros((m, 2)))
    centroids = initcenter
    #centroids = randCent(dataSet, 4)
    plt.plot(centroids[:, 0], centroids[:, 1], '+b', markersize=12)
    clusterChanged = True
    iret = 0

    while clusterChanged:
        clusterChanged = False
        rivalList = emptyRivalList(k)
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            rivalIndex = -1 #？？？？？
            rivalDist = inf
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < rivalDist and j != minIndex:
                    rivalDist = distJI
                    rivalIndex = j

            #print(rivalIndex)

            rivalList[rivalIndex].append(i)#第i个点使得cluster的中心有penalty了

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2

        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[np.where(clusterAssment[:, 0] == cent)]
            rival = []
            for item in rivalList[cent]:
                rival.append(dataSet[item])
            centroids[cent, :] = mean(ptsInClust, axis=0) - penalty * np.mean(rival, axis = 0)  # assign centroid to mean
        for cent in range(k):
            sample = np.shape(dataSet[np.where(clusterAssment[:, 0] == cent)])[0]
            if sample < 10:
                centroids = np.delete(centroids, cent, axis=0)
                k = np.shape(centroids)[0]
                print k
                clusterChanged = True
                break
    return centroids, clusterAssment, k

def kmeans(dataSet, k, penalty, initcenter, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.array(zeros((m, 2)))
    centroids = initcenter
    #centroids = createCent(dataSet, k)
    plt.plot(centroids[:,0], centroids[:,1], '+b', markersize=12)
    clusterChanged = True
    iret = 0

    while clusterChanged:
        clusterChanged = False
        rivalList = emptyRivalList(k)
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            rivalIndex = -1 #？？？？？
            rivalDist = inf
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2

        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[np.where(clusterAssment[:, 0] == cent)]
            rival = []
            for item in rivalList[cent]:
                rival.append(dataSet[item])
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
        for cent in range(k):
            sample = np.shape(dataSet[np.where(clusterAssment[:, 0] == cent)])[0]
            if sample < 10:
                centroids = np.delete(centroids, cent, axis=0)
                k = np.shape(centroids)[0]
                print k
                clusterChanged = True
                break

    return centroids, clusterAssment, k

def show(dataSet, k, centroids, clusterAssment):
    from matplotlib import pyplot as plt
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['+y', '+y', '+y', '+y', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()


def main():
    X, y = make_blobs(n_samples=500, n_features=2, centers=[[1, 0], [0, 1], [-1,0]], cluster_std=[0.1, 0.2, 0.1])
    initcenter = randCent(X, 4)
    myCentroids1, clustAssing1, k1 = rpcl(X, 4, 0.05,np.copy(initcenter))
    show(X, k1, myCentroids1, clustAssing1)


    myCentroids2, clustAssing2, k2 = kmeans(X, 4, 0.08,np.copy(initcenter))

    show(X, k2, myCentroids2, clustAssing2)


if __name__ == '__main__':
    main()