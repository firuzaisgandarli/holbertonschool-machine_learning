#!/usr/bin/env python3
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering with Ward linkage.
    """

    # compute linkage matrix (Ward method)
    Z = sch.linkage(X, method='ward')

    # plot dendrogram with distance threshold
    sch.dendrogram(
        Z,
        color_threshold=dist
    )

    plt.show()

    # assign clusters based on threshold
    clss = sch.fcluster(Z, t=dist, criterion='distance') - 1

    return clss
