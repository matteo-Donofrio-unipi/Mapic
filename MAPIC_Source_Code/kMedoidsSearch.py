from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import pdist, squareform

from FileManager import WriteCsvComparison


def runKMedoids(tree,dataset,n_clusters):

    #columnsList=list(['idTs','IdCandidate,'startingPosition','M/D']) LISTA DELLE PRIME 4 COLONNE BASE DI DATASET
    #DA QUESTE DEVO ESTRARRE TUTTE LE SUCCESSIVE


    numColumnsDataset=list(dataset.columns)
    numColumnsDataset=numColumnsDataset[4::]
    X=dataset[numColumnsDataset]
    #calcolo kmeans tra candidati
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)

    centroids = kmeans.cluster_centers_
    tree.SseList.append(kmeans.inertia_)
    tree.IterationList.append(kmeans.n_iter_)


    #per ogni centroide, estraggo medoide (record nel cluster piu vicino al centroide )
    medoids = list()
    for label in np.unique(kmeans.labels_):
        X_cluster = X[kmeans.labels_ == label]
        X_cluster.reset_index(inplace=True)
        dist = pdist(X_cluster)
        index = np.argmin(np.sum(squareform(dist), axis=0))
        medoids.append(X_cluster.iloc[index]['index'])


    #retituisco gli indici, di dataset, che sono stati scelti
    return medoids
