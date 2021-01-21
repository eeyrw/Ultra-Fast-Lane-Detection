# Authors: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2011
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

dsInfoCULane = {'embeddingFile': 'DatasetEmbedding_CULane.pkl',
                'imgRoot': r'E:\CULane'}
dsInfoTusimple = {
    'embeddingFile': 'DatasetEmbedding_Tusimple.pkl', 'imgRoot': r'E:\Tusimple'}

datasetInfo = dsInfoTusimple

with open(datasetInfo['embeddingFile'], 'rb') as f:
    embeddingDictDict = pickle.load(f)
    embeddingDict = embeddingDictDict['embedding']
    dataName = embeddingDictDict['name']
    # _class = list(dataset.classes)
    key_arr = list(embeddingDict.keys())

    # 采样一部分向量可视化
    visual_arr_key = sample(key_arr, min(9000, len(key_arr)))
    image_arr = np.concatenate([[embeddingDict[v]]
                                for v in visual_arr_key], axis=0)
    # for i in range(9000):
    #    image_arr.append(embeddingDict[visual_arr[i]])

IMG_ROOT = datasetInfo['imgRoot']
X_IMG = visual_arr_key
X = image_arr
n_samples, n_features = X.shape
n_neighbors = 30

# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors


def plot_embedding(X, title=None):
    # Compute DBSCAN
    clust = OPTICS(min_samples=5).fit(X)
    # db = DBSCAN(eps=0.3*5, min_samples=5).fit(X)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_

    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    # matplotlib.rcParams['agg.path.chunksize'] = 10000
    # matplotlib.rcParams.update(matplotlib.rc_params())
    plt.figure(figsize=(150, 150))
    ax = plt.subplot(111)

    for klass, color in zip(unique_labels, colors):
        Xk = X[clust.labels_ == klass]
        plt.plot(Xk[:, 0], Xk[:, 1], 'o', c=color, markersize=28)
    plt.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1],
             'k+', alpha=0.1, markersize=28)

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

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 1e-6:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         img = cv2.imread(os.path.join(IMG_ROOT, X_IMG[i]))
    #         img = cv2.resize(img, (480//4, 272//4))
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(img), X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(title+'.png')
    # plt.show()
    plt.close()

# ----------------------------------------------------------------------
# Plot images of the digits
# n_img_per_row = 20
# img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')


# ----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
# print("Computing random projection")
# rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
# X_projected = rp.fit_transform(X)
# plot_embedding(X_projected, "Random Projection of the digits")


# ----------------------------------------------------------------------
# Projection on to the first 2 principal components

# print("Computing PCA projection")
# t0 = time()
# X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
# plot_embedding(X_pca,
#                "Principal Components projection of the digits (time %.2fs)" %
#                (time() - t0))

# ----------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components

# print("Computing Linear Discriminant Analysis projection")
# X2 = X.copy()
# X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
# t0 = time()
# X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2
#                                                          ).fit_transform(X2, y)
# plot_embedding(X_lda,
#                "Linear Discriminant projection of the digits (time %.2fs)" %
#                (time() - t0))


# ----------------------------------------------------------------------
# Isomap projection of the digits dataset
# print("Computing Isomap projection")
# t0 = time()
# X_iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=2, n_jobs=16
#                         ).fit_transform(X)
# print("Done.")
# plot_embedding(X_iso,
#                "Isomap projection of the digits (time %.2fs)" %
#                (time() - t0))


# ----------------------------------------------------------------------
# Locally linear embedding of the digits dataset
# print("Computing LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2,
#                                       method='standard')
# t0 = time()
# X_lle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_lle,
#                "Locally Linear Embedding of the digits (time %.2fs)" %
#                (time() - t0))


# ----------------------------------------------------------------------
# Modified Locally linear embedding of the digits dataset
# print("Computing modified LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2,
#                                       method='modified')
# t0 = time()
# X_mlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_mlle,
#                "Modified Locally Linear Embedding of the digits (time %.2fs)" %
#                (time() - t0))


# ----------------------------------------------------------------------
# HLLE embedding of the digits dataset
# print("Computing Hessian LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2,
#                                       method='hessian')
# t0 = time()
# X_hlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_hlle,
#                "Hessian Locally Linear Embedding of the digits (time %.2fs)" %
#                (time() - t0))


# ----------------------------------------------------------------------
# LTSA embedding of the digits dataset
# print("Computing LTSA embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2,
#                                       method='ltsa')
# t0 = time()
# X_ltsa = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_ltsa,
#                "Local Tangent Space Alignment of the digits (time %.2fs)" %
#                (time() - t0))

# ----------------------------------------------------------------------
# MDS  embedding of the digits dataset
# print("Computing MDS embedding")
# clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
# t0 = time()
# X_mds = clf.fit_transform(X)
# print("Done. Stress: %f" % clf.stress_)
# plot_embedding(X_mds,
#                "MDS embedding of the digits (time %.2fs)" %
#                (time() - t0))

# ----------------------------------------------------------------------
# Random Trees embedding of the digits dataset
# print("Computing Totally Random Trees embedding")
# hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
#                                        max_depth=5)
# t0 = time()
# X_transformed = hasher.fit_transform(X)
# pca = decomposition.TruncatedSVD(n_components=2)
# X_reduced = pca.fit_transform(X_transformed)

# plot_embedding(X_reduced,
#                "Random forest embedding of the digits (time %.2fs)" %
#                (time() - t0))

# ----------------------------------------------------------------------
# Spectral embedding of the digits dataset
# print("Computing Spectral embedding")
# embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
#                                       eigen_solver="arpack")
# t0 = time()
# X_se = embedder.fit_transform(X)

# plot_embedding(X_se,
#                "Spectral embedding of the digits (time %.2fs)" %
#                (time() - t0))

# ----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca',
                     random_state=0, n_jobs=16, perplexity=100)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

# ----------------------------------------------------------------------
# NCA projection of the digits dataset
# print("Computing NCA projection")
# nca = neighbors.NeighborhoodComponentsAnalysis(init='random',
#                                                n_components=2, random_state=0)
# t0 = time()
# X_nca = nca.fit_transform(X, y)

# plot_embedding(X_nca,
#                "NCA embedding of the digits (time %.2fs)" %
#                (time() - t0))

plt.show()
