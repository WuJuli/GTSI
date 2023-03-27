from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from scipy import linalg
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import warnings

warnings.filterwarnings('ignore')


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
    return x, y


def SelBest(arr: list, X: int) -> list:
    dx = np.argsort(arr)[:X]
    return arr[dx]


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
    name = "output1/" + model.__name__ +" k="+ str(k)+ ".jpg"
    plt.savefig(name)
    plt.cla()


def k_means(X, Y):
    wcss = []
    for i in range(1, 40):
        kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 40), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("output1/k-mean default.jpg")
    plt.cla()
    # 10
    print(kmeans.n_iter_)
    # k = 40 not generate,20 ok, see 30
    generate_silhoutte_score_plot(X, 30, KMeans)

    range_n_clusters = [8]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=322)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X.iloc[:, 8], X.iloc[:, 2], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 8st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.savefig("output1/k-mean Cluster.jpg")
    plt.cla()
    clusterNum = 7
    k_means = KMeans(n_clusters=clusterNum, max_iter=300, n_init=10, random_state=0)
    k_means.fit(X)
    labels = k_means.labels_
    print("labels", labels)
    print("rand_score(Y, labels):", rand_score(Y, labels))

    return


def EM(X, Y):
    print(X.shape, Y.shape)
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 8)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    bars = []
    print("clf", clf)
    print("covariance_type",clf.covariance_type)

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                bic[i * len(n_components_range): (i + 1) * len(n_components_range)],
                width=0.2,
                color=color,
            )
        )
    plt.xticks(n_components_range)
    plt.ylim([bic.max(), bic.min() * 1.01 - 0.01 * bic.max()])
    plt.title("BIC score per model")
    xpos = (
            np.mod(bic.argmin(), len(n_components_range))
            + 0.65
            + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    )
    plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    print("cov shape",clf.covariances_.shape)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
        # print("999", cov)
        v, w = linalg.eigh(cov)
        print(v.shape,w.shape)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X.iloc[Y_ == i, 8], X.iloc[Y_ == i, 2], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[10], v[11], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(
        f"Selected GMM: {best_gmm.covariance_type} model, "
        f"{best_gmm.n_components} components"
    )
    plt.subplots_adjust(hspace=0.35, bottom=0.02)
    plt.savefig("output1/EM Cluster.jpg")
    plt.cla()
    print(clf)
    print("rand_score: ", rand_score(Y, best_gmm.predict(X)))

    return


if __name__ == '__main__':
    data = pd.read_csv("../input/breast-cancer.csv")
    X, Y = data_process(data)
    # print(X.shape, Y.shape)
    # print(X.head())
    # k_means(X, Y)
    # EM(X, Y)
