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
    # PRB = 'Probabilistic'
    KNN = 'KNN'
    HCL = 'Hierarchical'
    ITE = 'Iterative'
    KMN = 'K-Means'


class EvalMethod(Enum):
    """docstring for Evaluation."""
    SUB = 'Substitution'
    CRS = 'Cross'
    LOO = 'Leave One Out'


# TODO: Might not be useful
def classify(x, pixels, classes, method=ClassifMethod.EUC):
    x = np.array(x)
    pixels = np.array(pixels)
    classes = np.array(classes)

    if method == ClassifMethod.EUC:
        return _classify_distance(x, pixels, classes)
    elif method == ClassifMethod.MAH:
        return _classify_distance(x, pixels, classes, 'mahalanobis')
    # elif method == ClassifMethod.PRB:
    #     return _classify_probabilistic(x, pixels, classes)
    elif method == ClassifMethod.KNN:
        return _classify_knn(x, pixels, classes)
    elif method == ClassifMethod.HCL:
        return _classify_hierarchical(x, pixels, classes)
    elif method == ClassifMethod.ITE:
        return _classify_iterative(x, pixels, classes)
    elif method == ClassifMethod.KMN:
        return _classify_kmeans(x, pixels, classes)
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


def _calculate_centers(x_train, y_train):
    class_labels=np.unique(y_train)
    centers=[]

    # Calculate means for all classes
    for c in class_labels:
        x_class=x_train[y_train == c]
        # Mean of columns
        centers.append(x_class.mean(axis=0))

    return centers, class_labels


def _classify_distance(x, x_train, y_train, met='euclidean'):
    centers, class_labels = _calculate_centers(x_train, y_train)
    dists = cdist(x, centers, metric=met)

    return class_labels[dists.argmin(axis=1)]


# Deprecate this crap
def _classify_probabilistic(x, x_train, y_train):
    '''
    Deprecated: This one sucks
    '''
    centers, class_labels = _calculate_centers(x_train, y_train)
    m_distances = cdist(x[np.newaxis], centers, metric='mahalanobis')[0]
    covs = [np.cov(x_train[y_train == cl]) for cl in class_labels]

    pr_dist = []
    for i, cl in enumerate(class_labels):
        temp = 1 / (np.pi ** (class_labels.shape[0] / 2))
        temp *= np.sqrt(np.linalg.det(covs[i]))
        temp *= np.exp(m_distances[i] * -0.5)
        pr_dist.append(temp)
    print(pr_dist, np.argmax((pr_dist / sum(pr_dist)) * 100))
    idx = np.argmax((pr_dist / sum(pr_dist)) * 100)

    return class_labels[idx]


def _classify_knn(x, pixels, classes, k=5):
    le = LabelEncoder()
    classes_train = le.fit_transform(classes)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(pixels, classes_train)
    y_hat = clf.predict(x)
    return le.inverse_transform(y_hat)


def _classify_hierarchical(x, x_train, y_train):
    y_labels = np.unique(y_train)
    cluster = AgglomerativeClustering(n_clusters=len(y_labels))
    cluster_labels = cluster.fit_predict(x_train)

    centers = []
    cen_labels = []
    for cl in np.unique(cluster_labels):
        x_class = x_train[cluster_labels == cl]
        center = np.array([np.mean(x_class[:, x_col])
                           for x_col in range(x_class.shape[1])])
        centers.append(center)
        cen_labels.append(y_train[list(cluster_labels).index(cl)])
    distances = cdist(x, centers)
    cen_labels = np.array(cen_labels)
    return cen_labels[distances.argmin(axis=1)]


def _classify_iterative(x, x_train, y_train, thres=100):
    groups = [x_train[0][np.newaxis]]
    centers = [x_train[0]]

    for i in range(1, x_train.shape[0]):
        xt = x_train[i]
        distances = cdist(xt[np.newaxis], centers)

        if np.all(xt >= thres):
            # Create a new group
            groups.append(xt)
            centers.append(xt)
        else:
            idx = np.argmin(distances)
            groups[idx] = np.vstack((xt, groups[idx]))
            centers[idx] = groups[idx].mean(axis=0)

    print(centers)


def _classify_kmeans(x, x_train, y_train):
    class_labels = np.unique(y_train)
    kmeans = KMeans(n_clusters=len(class_labels)).fit(x_train)
    km_labels = kmeans.labels_
    y_hat = kmeans.predict(x)

    return y_train[km_labels == y_hat][0]


classification_function = {
    ClassifMethod.EUC: _classify_distance,
    ClassifMethod.MAH: _classify_distance,
    # ClassifMethod.PRB: _classify_probabilistic,
    ClassifMethod.KNN: _classify_knn,
    # Classify Iterative
    ClassifMethod.KMN: _classify_kmeans
}

if __name__ == '__main__':
    a = np.random.randint(0, 100, 40).reshape(20, 2)
    b = np.repeat([1, 2, 3], [6, 7, 7])
    print(_classify_probabilistic(
        np.array([1, 1]), a, b))
