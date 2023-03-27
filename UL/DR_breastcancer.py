import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn.decomposition import FastICA, PCA
from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance

warnings.filterwarnings('ignore')


def SelBest(arr: list, X: int) -> list:
    dx = np.argsort(arr)[:X]
    return arr[dx]


def gmm_js(gmm_p, gmm_q, n_samples=10 ** 5):
    # https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4

    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
                    + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)


def generate_kmeans_SV_ICD_plots(X, k):
    plot_nums = len(k)
    fig, axes = plt.subplots(plot_nums, 2, figsize=[25, 40])
    col_ = 0
    for i in k:
        kmeans = KMeans(n_clusters=i, algorithm="full")
        visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=axes[col_][0])
        visualizer.fit(X)
        visualizer.finalize()

        kmeans = KMeans(n_clusters=i, algorithm="full")
        visualizer = InterclusterDistance(kmeans, ax=axes[col_][1])
        visualizer.fit(X)
        visualizer.finalize()

        col_ += 1
    name = "output3/" + "k=" + str(k) + ".jpg"
    plt.savefig(name)
    plt.cla()


def generate_silhoutte_score_plot(X, k, model):
    n_clusters = np.arange(2, k)
    sils = []
    sils_err = []
    iterations = k
    for n in n_clusters:
        tmp_sil = []
        for _ in range(iterations):
            clf = model(n).fit(X)
            labels = clf.predict(X)
            sil = metrics.silhouette_score(X, labels, metric='euclidean')
            tmp_sil.append(sil)
        val = np.mean(SelBest(np.array(tmp_sil), int(iterations / 5)))
        err = np.std(tmp_sil)
        sils.append(val)
        sils_err.append(err)
    plt.errorbar(n_clusters, sils, yerr=sils_err)
    plt.title("Silhouette Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters ({})".format("Test"))
    plt.ylabel("Score")
    name = "output3/" + model.__name__ + " k=" + str(k) + ".jpg"
    plt.savefig(name)
    plt.cla()

def data_processXY(data):
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    # Transform the 'yes' and 'no' values (target variable) to 1 and 0 respectively
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Scalling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Split the data to train and test sets
    x = scaled_data.loc[:, scaled_data.columns != 'diagnosis']
    y = scaled_data['diagnosis']
    return x, y



def data_process(data):
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    # Transform the 'yes' and 'no' values (target variable) to 1 and 0 respectively
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Scalling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Split the data to train and test sets
    x = scaled_data.loc[:, scaled_data.columns != 'diagnosis']
    y = scaled_data['diagnosis']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=322, shuffle=True)
    return x_train, x_test, y_train, y_test


def pca(x_train, x_test):
    print("this is PCA")
    pca = PCA().fit(x_train)
    # https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_,
            label='Individual explained variance')
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_),
             label='Cumulative explained variance')
    plt.title("Component-wise and Cumulative Explained Variance on Breast cancer dataset")
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig("output3/PCA.jpg")
    plt.cla()
    # n= 5
    n_components = [3, 4, 5, 6, 7, 8, 9, 10]
    reconstruction_error = []
    for comp in n_components:
        pca = PCA(n_components=comp)
        X_transformed = pca.fit_transform(x_train)
        X_projected = pca.inverse_transform(X_transformed)
        reconstruction_error.append(((x_train - X_projected) ** 2).mean())

        # if(comp == gridSearch.best_estimator_.named_steps['pca'].n_components):
        # chosen_error = ((x_train - X_projected) ** 2).mean()
    plt.cla()
    fig2, ax2 = plt.subplots()
    ax2.plot(n_components, reconstruction_error, linewidth=2)
    # ax2.axvline(gridSearch.best_estimator_.named_steps['pca'].n_components, linestyle=':', label='n_components chosen', linewidth = 2)
    plt.axis('tight')
    plt.xlabel('Number of components')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction error for n_components chosen')
    plt.savefig("output3/PCA Reconstruction Error.jpg")
    plt.cla()

    # TIME
    n = 5
    start = time.time()
    pca = PCA(n_components=n).fit(x_train)
    print("Time(s) " + str(time.time() - start))
    print("original shape:", x_train.shape)
    X_pca = pca.transform(x_train)
    print("transformed shape:", X_pca.shape)
    print("----------------")

    plt.figure(figsize=(6, 4))
    plt.title('PCA Components')

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            plt.scatter(X_pca[:, i], X_pca[:, j])
    plt.savefig("output3/PCA Components.jpg")
    plt.cla()

    # RECONSTRUCTION ERROR
    pca = PCA(n_components=5)
    X_test_transformed = pca.fit_transform(x_test)
    X_test_projected = pca.inverse_transform(X_test_transformed)
    test_reconstruction_error = ((x_test - X_test_projected) ** 2).mean()
    print("test_reconstruction_error: ", test_reconstruction_error)

    return X_pca


def run_ICA(X, title):
    dims = list(np.arange(2, (X.shape[1] - 1), 3))
    dims.append(X.shape[1])
    ica = FastICA(random_state=322)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: " + title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'b-')
    plt.grid(False)
    plt.savefig("output3/ICA.jpg")
    plt.cla()


def ICA(x_train, x_test):
    print("this is ICA")
    run_ICA(x_train, "ICA: Breast cancer Dataset Average Kurtosis")

    n = 30
    ica = FastICA(n_components=n, max_iter=10000, tol=0.1)
    X_ica = ica.fit_transform(x_train)

    plt.figure(figsize=(6, 4))
    plt.title('ICA Components on Breast cancer dataset')

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            plt.scatter(X_ica[:, i], X_ica[:, j])
    plt.savefig("output3/ICA Components.jpg")
    plt.cla()

    start = time.time()
    ica = FastICA(n_components=n, max_iter=10000, tol=0.1)
    print("Time(s) " + str(time.time() - start))
    print("original shape:", x_train.shape)
    X_ica = ica.fit_transform(x_train)
    print("transformed shape:", X_ica.shape)
    print("--------------------------------")

    # RECONSTRUCTION ERROR
    ica = FastICA(n_components=30)
    X_test_transformed = ica.fit_transform(x_test)
    X_test_projected = ica.inverse_transform(X_test_transformed)
    test_reconstruction_error = ((x_test - X_test_projected) ** 2).mean()
    print("test_reconstruction_error: ", test_reconstruction_error)

    return X_ica


def inverse_transform_rp(rp, X_transformed, X_train):
    # print(X_transformed.dot(rp.components_).shape)
    # print(np.mean(X_train, axis=0).shape)
    # print("666")
    return X_transformed.dot(rp.components_) + X_train


def RP(x_train, x_test):
    print("this is rp")
    print("Calculating Reconstruction Error")
    n_components = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    reconstruction_error = []
    for comp in n_components:
        rp = random_projection.GaussianRandomProjection(n_components=comp)
        X_transformed = rp.fit_transform(x_train)
        # print("888", X_transformed.shape, rp.components_.shape, x_train.shape)
        X_projected = inverse_transform_rp(rp, X_transformed, x_train)
        reconstruction_error.append(((x_train - X_projected) ** 2).mean())

    fig2, ax2 = plt.subplots()
    ax2.plot(n_components, reconstruction_error, linewidth=2)

    plt.axis('tight')
    plt.xlabel('Number of components')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction error for n_components chosen on Heart disease dataset')
    plt.savefig("output3/RP Reconstruction error.jpg")
    plt.cla()

    n = 17
    rp = random_projection.GaussianRandomProjection(n_components=n)
    X_rp = rp.fit_transform(x_train)

    plt.figure(figsize=(6, 4))
    plt.title('RP Components on Heart disease dataset')

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            plt.scatter(X_rp[:, i], X_rp[:, j])

    plt.savefig("output3/RP Components error.jpg")
    plt.cla()

    start = time.time()
    rp = random_projection.GaussianRandomProjection(n_components=n)

    print("Time(s) " + str(time.time() - start))
    print("original shape:", x_train.shape)
    X_rp = rp.fit_transform(x_train)
    print("transformed shape:", X_rp.shape)
    print("---------------------------")

    # RECONSTRUCTION ERROR
    rp = random_projection.GaussianRandomProjection(n_components=17)
    X_test_transformed = rp.fit_transform(x_test)
    X_test_projected = inverse_transform_rp(rp, X_test_transformed, x_test)
    test_reconstruction_error = ((x_test - X_test_projected) ** 2).mean()
    print("test_reconstruction_error:n ", test_reconstruction_error)

    return X_rp


def SVD(x_train, x_test):
    print("this is SVD")
    tsvd = TruncatedSVD(n_components=x_train.shape[1] - 1)
    X_tsvd = tsvd.fit(x_train)
    # https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/
    plt.bar(range(1, len(X_tsvd.explained_variance_ratio_) + 1), X_tsvd.explained_variance_ratio_,
            label='Individual explained variance')
    plt.step(range(1, len(X_tsvd.explained_variance_ratio_) + 1), np.cumsum(X_tsvd.explained_variance_ratio_),
             label='Cumulative explained variance')
    plt.title("SVD Component-wise and Cumulative Explained Variance on Breast cancer data")
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig("output3/SVD.jpg")
    plt.cla()

    # Reconstruction Error

    n_components = [4,5,6,7, 8, 9, 10]
    reconstruction_error = []

    for comp in n_components:
        tsvd = TruncatedSVD(n_components=comp)
        X_transformed = tsvd.fit_transform(x_train)
        X_projected = tsvd.inverse_transform(X_transformed)
        reconstruction_error.append(((x_train - X_projected) ** 2).mean())

    fig2, ax2 = plt.subplots()
    ax2.plot(n_components, reconstruction_error, linewidth=2)

    plt.axis('tight')
    plt.xlabel('Number of components')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction error for n_components chosen')
    plt.savefig("output3/SVD Reconstruction error.jpg")
    plt.cla()

    n = 5
    tsvd = TruncatedSVD(n_components=n)
    X_tsvd = tsvd.fit_transform(x_train)

    plt.figure(figsize=(6, 4))
    plt.title('SVD Components on Breast cancer dataset')

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                continue
            plt.scatter(X_tsvd[:, i], X_tsvd[:, j])

    plt.savefig("output3/SVD Components.jpg")
    plt.cla()

    start = time.time()
    tsvd = TruncatedSVD(n_components=n)
    print("Time(s) " + str(time.time() - start))
    print("original shape:", x_train.shape)
    X_tsvd = tsvd.fit_transform(x_train)
    print("transformed shape:", X_tsvd.shape)
    print("------------------------------------")
    # RECONSTRUCTION ERROR
    tsvd = TruncatedSVD(n_components=5)
    X_test_transformed = tsvd.fit_transform(x_test)
    X_test_projected = tsvd.inverse_transform(X_test_transformed)
    test_reconstruction_error = ((x_test - X_test_projected) ** 2).mean()
    print("test_reconstruction_error: ",test_reconstruction_error)
    return X_tsvd



if __name__ == '__main__':
    # load the data
    data = pd.read_csv("../input/breast-cancer.csv")

    X, Y = data_processXY(data)
    # Process data and distribute it according to 3:7
    x_train, x_test, y_train, y_test = data_process(data)

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, y_test_encoded.shape, y_train_encoded.shape)

    x_pca = pca(x_train, x_test)
    x_ica = ICA(x_train, x_test)
    x_rp = RP(x_train, x_test)
    x_svd = SVD(x_train, x_test)


