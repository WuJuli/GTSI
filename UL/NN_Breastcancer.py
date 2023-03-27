import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, learning_curve, cross_val_score, train_test_split, \
    RandomizedSearchCV, ShuffleSplit, validation_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=322, shuffle=True)
    return x_train, x_test, y_train, y_test


def nnetwork(str, x_train, x_test, y_train, y_test):
    y_train_encoded = to_categorical(y_train, 6)
    y_test_encoded = to_categorical(y_test, 6)

    # Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_test = sc.transform(x_test)

    # Initialising the ANN second method
    classifier2 = Sequential()

    # Adding the input layer and the first hidden layer
    classifier2.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))
    # Adding dropout to prevent overfitting
    classifier2.add(Dropout(rate=0.1))

    # Adding the second hidden layer
    classifier2.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
    # Adding dropout to prevent overfitting
    classifier2.add(Dropout(rate=0.1))

    # Adding the output layer
    classifier2.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    # Compiling the ANN
    classifier2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the ANN to the Training set
    history2 = classifier2.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=450)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    t = f.suptitle('ANN Performance using Second Method', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    ax1.plot(range(450), history2.history['accuracy'], label='Train Accuracy')
    ax1.plot(range(450), history2.history['val_accuracy'], label='Validation Accuracy')
    plt.tick_params(direction="in", labelsize=12)
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(range(450), history2.history['loss'], label='Train Loss')
    ax2.plot(range(450), history2.history['val_loss'], label='Validation Loss')
    plt.tick_params(direction="in", labelsize=12)
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
    name = "./output7/NN - " + str + " - Accuracy&Loss.jpg"
    f.savefig(name)

    return


if __name__ == '__main__':
    # load the data
    data = pd.read_csv("../input/breast-cancer.csv")

    # Process data and distribute it according to 3:7
    x_train, x_test, y_train, y_test = data_process(data)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # nnetwork("default ",x_train, x_test, y_train, y_test)

    # pca = PCA().fit(x_train)
    # x_pca = pca.transform(x_train)
    # x_test_pca = pca.transform(x_test)
    # nnetwork("pca", x_pca, x_test_pca, y_train, y_test)

    # ica = FastICA().fit(x_train)
    # x_ica = ica.transform(x_train)
    # x_test_ica = ica.transform(x_test)
    # nnetwork("ica", x_ica, x_test_ica, y_train, y_test)

    # rp = random_projection.GaussianRandomProjection(n_components=30)
    # x_rp = rp.fit_transform(x_train)
    # x_test_rp = rp.fit_transform(x_test)
    # nnetwork("rp", x_rp, x_test_rp, y_train, y_test)

    # tsvd = TruncatedSVD(n_components=x_train.shape[1])
    # x_tsvd = tsvd.fit_transform(x_train)
    # x_test_tsvd = tsvd.fit_transform(x_test)
    # nnetwork("svd", x_tsvd, x_test_tsvd, y_train, y_test)

    # k_means = KMeans(n_clusters=30, max_iter=300, n_init=10, random_state=322)
    # k_means.fit(x_train)
    # x_train_kmeans = k_means.transform(x_train)
    # x_test_kmeans = k_means.transform(x_test)
    # nnetwork("kmeans", x_train_kmeans, x_test_kmeans, y_train, y_test)

    gmm = mixture.GaussianMixture(n_components=30, random_state=322)
    gmm.fit(x_train)
    x_train_gmm = gmm.predict_proba(x_train)
    x_test_gmm = gmm.predict_proba(x_test)
    nnetwork("GMM", x_train_gmm, x_test_gmm, y_train, y_test)



