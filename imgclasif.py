from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


class ClassifMethod(Enum):
    """docstring for ClassificationMethod."""
    EUC = 'Euclidean'
    MAH = 'Mahalanobis'
    PRB = 'Probabilistic'
    KNN = 'KNN'


class UnsupervisedMethod(Enum):
    HCL = 'Hierarchical'
    CHA = 'Chain'
    KMN = 'K-Means'


class EvalMethod(Enum):
    """docstring for Evaluation."""
    SUB = 'Substitution'
    CRS = 'Cross'
    LOO = 'Leave One Out'


def classify(x, pixels, classes, method=ClassifMethod.EUC):
    x = np.array(x)
    pixels = np.array(pixels)
    classes = np.array(classes)

    if method == ClassifMethod.EUC:
        return _classify_distance(x, pixels, classes)
    elif method == ClassifMethod.MAH:
        return _classify_distance(x, pixels, classes, 'mahalanobis')
    elif method == ClassifMethod.PRB:
        return _classify_probabilistic(x, pixels, classes)
    elif method == ClassifMethod.KNN:
        return _classify_knn(x, pixels, 5)
    else:
        raise ValueError(f'Unkown method: {method}')


def clusterize(pixels, k=None, method=None, thres=None):
    pixels = np.array(pixels)

    if method == UnsupervisedMethod.HCL:
        return _classify_hierarchical(pixels, k)
    elif method == UnsupervisedMethod.CHA:
        return _classify_chain(pixels, thres)
    elif method == UnsupervisedMethod.KMN:
        return _classify_kmeans(pixels, k)
    else:
        raise ValueError(f'Unkown method: {method}')


def validate(X, y, spc, class_method=ClassifMethod.EUC,
             eval_method=EvalMethod.SUB):
    clasif = classification_function[class_method]
    class_labels = np.unique(y)
    num_classes = len(class_labels)

    if eval_method == EvalMethod.SUB:
        x_train = x_test = X
        y_train = y_test = y
        y_hat = clasif(x_test, x_train, y_train)
        cm = confusion_matrix(y_test, y_hat)
    elif eval_method == EvalMethod.CRS:
        cms = np.zeros((20, num_classes, num_classes))

        for i in range(cms.shape[0]):
            selected_x = []
            selected_y = []

            for cl in class_labels:
                subx = X[y == cl]
                suby = y[y == cl]

                subsize = subx.shape[0] // 2
                idxs = np.random.choice(
                    np.arange(subsize), subsize, replace=False)
                selected_x.extend(subx[idxs])
                selected_y.extend(suby[idxs])

            x_train, x_test, y_train, y_test = train_test_split(
                np.array(selected_x), np.array(selected_y), test_size=0.5)

            y_hat = clasif(x_test, x_train, y_train)
            cms[i] = confusion_matrix(y_test, y_hat)
        cm = cms.mean(axis=0)
    elif eval_method == EvalMethod.LOO:
        xmask = np.ma.array(X, mask=False)
        ymask = np.ma.array(y, mask=False)

        y_hat = np.zeros(y.shape)
        mask = np.tile(True, y_hat.size)
        for i, xt in enumerate(X):
            mask[i] = False
            y_hat[i] = clasif(X[i][np.newaxis], X[mask], y[mask])
            mask[i] = True
        cm = confusion_matrix(y, y_hat)
    else:
        raise ValueError(f'Unkown evaluation method: {eval_method}')

    # Convert to percentage
    cm = (cm / cm.sum(axis=0)) * 100
    print(cm)

    ef = cm.diagonal().mean()
    plt.bar([f'Class-{i + 1}' for i in range(num_classes)],
            [cm[i, i] for i in range(num_classes)])
    plt.title('{} Evaluation\n{} method\nEfficiency: {}%'.format(
        eval_method.value, class_method.value, '%.4f' % ef))
    plt.ylabel('Class efficiency')
    plt.show()


def calculate_centers(x_train, y_train):
    class_labels = np.unique(y_train)
    centers = np.zeros((len(class_labels), x_train.shape[1]))

    # Calculate means for all classes
    for i, c in enumerate(class_labels):
        centers[i] = x_train[y_train == c].mean(axis=0)

    return centers, class_labels


def nearest_cluster(x, centers):
    dists = cdist(x, centers)
    return dists.argmin(axis=1)


def _classify_distance(x, x_train, y_train, met='euclidean'):
    centers, class_labels = calculate_centers(x_train, y_train)
    dists = cdist(x, centers, metric=met)

    return class_labels[dists.argmin(axis=1)]


# Deprecate this crap
def _classify_probabilistic(x, x_train, y_train):
    '''
    Deprecated: This one sucks
    '''
    centers, class_labels = calculate_centers(x_train, y_train)
    m_distances = cdist(x[np.newaxis], centers, metric='mahalanobis')[0]
    covs = [np.cov(x_train[y_train == cl]) for cl in class_labels]

    idx = np.argmax((m_distances / sum(m_distances)) * 100)

    return class_labels[idx]


def _classify_knn(x, x_train, y_train, k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train, y_train)
    return clf.predict(x)


########################
# Unsupervised Methods #
########################


def _classify_hierarchical(x_train, k):
    cluster = AgglomerativeClustering(n_clusters=k)
    return cluster.fit_predict(x_train)


def _classify_chain(x_train, thres=100):
    groups = [x_train[0][np.newaxis]]
    centers = [x_train[0]]

    for xi in x_train:
        distances = cdist(xi[np.newaxis], centers)

        if np.all(xi > thres):
            # Create a new group
            groups.append(xi[np.newaxis])
            centers.append(xi)
        else:
            idx = distances.argmin()
            groups[idx] = np.vstack((xi, groups[idx]))
            centers[idx] = groups[idx].mean(axis=0)

    return cdist(x_train, centers).argmin(axis=1)

def _classify_kmeans(x_train, k):
    kmeans = KMeans(n_clusters=k).fit(x_train)
    return kmeans.labels_


# For lack of a better name
def _lloyd_correction(x, x_train, y_train, kcen, alpha=0.1, epochs=20, thres=100):
    '''
    This is supposed to improve KMeans by readjusting the centers.

    kcen is a dictionary that has the label as a key that maps to a cluster
    center.
    '''
    # TODO: Optimize by verifying instance
    le = LabelEncoder()
    le.fit(y_train)
    y_labels = le.classes_
    k_labels = kcen.keys()

    # Verify that all data is correctly labeled
    if y_labels.shape[0] != len(k_labels):
        raise ValueError(
            'the amount of labels must correspond to the cluster centers.')
    if not np.all([yl in k_labels for yl in y_labels]):
        raise ValueError(
            'label of data must correspond to the labels of the clusters')

    for i in range(epochs):
        for xi in x_train:
            pass


classification_function = {
    # Supervised
    ClassifMethod.EUC: _classify_distance,
    ClassifMethod.MAH: _classify_distance,
    ClassifMethod.PRB: _classify_probabilistic,
    ClassifMethod.KNN: _classify_knn,
    # Unsupervised
    UnsupervisedMethod.HCL: _classify_hierarchical,
    UnsupervisedMethod.KMN: _classify_kmeans
}

if __name__ == '__main__':
    xt = np.random.randint(0, 256, (20, 3))
    y = np.random.randint(0, 3, 20)
    kc = {
        0: np.array([10, 10, 10]),
        1: np.array([128, 128, 128]),
        2: np.array([200, 200, 200])
    }
    _lloyd_correction(
        np.array([[20, 20, 1], [6, 0, 0], [100, 3, 100]]), xt, y, kc)
