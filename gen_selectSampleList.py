import pickle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
import os
from random import sample
from sklearn.cluster import DBSCAN, OPTICS
import sklearn.utils

dsInfoCULane = {'embeddingFile': 'DatasetEmbedding_CULane.pkl',
                'imgRoot': r'E:\CULane'}
dsInfoTusimple = {
    'embeddingFile': 'DatasetEmbedding_Tusimple.pkl', 'imgRoot': r'E:\Tusimple'}

datasetInfo = dsInfoTusimple


def plot_embedding(X, labels, imgPathList, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)

    # Black removed and is used for noise instead.
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    colors.append([0.7, 0.7, 0.7])  # for noise

    # matplotlib.rcParams['agg.path.chunksize'] = 10000
    # matplotlib.rcParams.update(matplotlib.rc_params())
    plt.figure(figsize=(150, 150))
    ax = plt.subplot(111)
    # fig, ax = plt.subplots()
    # fig.set_size_inches(21.6, 14.4)
    # plt.axis('off')
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0.7, 0.7, 0.7, 0.2]

    #     class_member_mask = (labels == k)

    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', c=tuple(col), markersize=28)
    #     # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #     #          markeredgecolor='k', markersize=7)

    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', c=tuple(col), markersize=18)
    #     # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #     #          markeredgecolor='k', markersize=3)

    # plt.plot(X[:, 0], X[:, 1], 'o', c='red', markersize=18)

    # only print thumbnails with matplotlib > 1.0
    shown_images = np.array([[x_max, x_max]])  # just something big
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 1e-6*x_max:
            # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        img = cv2.imread(imgPathList[i])
        img = cv2.resize(img, (120, 68))
        img = cv2.rectangle(img, (0, 0), (120, 68),
                            colors[labels[i]][::-1], 10)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(img), X[i])
        ax.add_artist(imagebox)
    ax.update_datalim(X)
    ax.autoscale()
    # plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(title+'.png')
    # plt.show()
    plt.close()


def genSelectSampleList(labels, imageFileNameList, proportion):
    clusterGroup = []
    selectSampleList = []
    unique_labels = set(labels)
    for uniqueLbl in unique_labels:
        cluster = []
        for i, lbl in enumerate(labels):
            if uniqueLbl == lbl:
                cluster.append(imageFileNameList[i])
        clusterGroup.append(cluster)

    clusterGroup.sort(lambda clusterA, clusterB: len(clusterA) < len(clusterB))
    clusterNum = len(clusterGroup)
    totalSampleNum = len(labels)
    selectSampleNum = int(totalSampleNum*proportion)
    everyClusterSampleNum = selectSampleNum//clusterNum
    everyClusterSampleNum_rem = selectSampleNum % clusterNum
    for cluster in clusterGroup:
        if len(cluster) < everyClusterSampleNum:
            resampleCluster = sklearn.utils.resample(
                cluster, n_samples=everyClusterSampleNum, replace=True, random_state=0)
        else:
            resampleCluster = sklearn.utils.resample(
                cluster, n_samples=everyClusterSampleNum, replace=False, random_state=0)
        selectSampleList += resampleCluster

    if everyClusterSampleNum_rem is not 0:
        resampleCluster = sklearn.utils.resample(
            clusterGroup[-1], n_samples=everyClusterSampleNum_rem, replace=True, random_state=0)
        selectSampleList += resampleCluster

    return selectSampleList


if __name__ == "__main__":
    IMG_ROOT = datasetInfo['imgRoot']
    with open(datasetInfo['embeddingFile'], 'rb') as f:
        embeddingDictDict = pickle.load(f)
        embeddingDict = embeddingDictDict['embedding']
        dataName = embeddingDictDict['name']
        imagePathList = list(embeddingDict.keys())
        X = np.array(embeddingDict.values())

    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca',
                         random_state=0, n_jobs=16, perplexity=100)
    t0 = time()
    X_tsne = tsne.fit_transform(X)

    print("t-SNE embedding (time %.2fs)" % (time() - t0))

    db = DBSCAN(eps=0.3*5, min_samples=5).fit(X_tsne)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    sampleList = genSelectSampleList(labels, imagePathList, 0.1)
    with open(os.path.join(IMG_ROOT, 'selectedlist.txt'), 'w') as f:
        for samplePath in sampleList:
            f.writelines(sampleList)

    plot_embedding(X_tsne, labels, [os.path.join(
        IMG_ROOT, imagePath) for imagePath in imagePathList], title='t-SNE embedding')
