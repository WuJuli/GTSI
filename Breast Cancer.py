from random import seed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, learning_curve, cross_val_score, train_test_split, \
    RandomizedSearchCV, ShuffleSplit, validation_curve, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import os
import graphviz
from time import time
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def display(df):
    # print(data.head())
    # print(data.info())
    # print(data.shape)

    # create the correlation matrix heat map
    plt.figure(figsize=(10, 6))
    sns.heatmap(data[[data.columns[0], data.columns[1], data.columns[2], data.columns[3],
                      data.columns[4], data.columns[5]]].corr(), linewidths=.1, cmap="YlGnBu", annot=True)
    plt.yticks(rotation=0)
    plt.suptitle('Correlation Matrix')
    plt.savefig("./output2/display - corr.jpg")
    plt.cla()
    return


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


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


def decisionTree(x_train, x_test, y_train, y_test):
    X = pd.concat([x_train, x_test], axis=1)
    Y = pd.concat([y_train, y_test], axis=1)
    X_scaled = scale(X)

    # imply the default model
    dt = DecisionTreeClassifier(criterion='entropy', random_state=322)
    dt.fit(x_train, y_train)
    y_pre_dt = dt.predict(x_test)
    ac_score = dt.score(x_test, y_test)
    cv_score = cross_val_score(dt, x_train, y_train, cv=5).mean()
    print("dt_default", ac_score, cv_score)

    # random search
    param_dist = {"max_depth": [*range(1, 15)],
                  "splitter": ("best", "random"),
                  "criterion": ("gini", "entropy"),
                  "min_samples_leaf": [*range(1, 15)],
                  "min_samples_split": [*range(2, 15)]
                  }
    n_iter_search = 300

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=322)
    random_search = RandomizedSearchCV(dt, param_distributions=param_dist, cv=cv, n_iter=n_iter_search,
                                       random_state=322)
    start = time()
    random_search.fit(x_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    best_estimator_DT = random_search.best_estimator_
    print("best_estimator_DT :", best_estimator_DT, random_search.best_score_)

    # validation curve
    # Vary the max_depth parameter from 1 to 15
    max_depth = range(2, 15)
    # random_search decision tree results
    DT_randomsearched = DecisionTreeClassifier(min_samples_leaf=2, min_samples_split=11, random_state=322)
    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(DT_randomsearched, x_train, y_train,
                                                 param_name="max_depth", param_range=max_depth, cv=cv)

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Validation Curve with Decision trees')
    plt.plot(max_depth, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(max_depth, test_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(max_depth, train_mean - train_std, train_mean + train_std, alpha=0.15, color='r')
    plt.fill_between(max_depth, test_mean - test_std,
                     test_mean + test_std, alpha=0.15, color='g')

    # Visual aesthetics
    plt.legend(loc='best')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Score')
    plt.ylim([-0.05, 1.05])
    plt.savefig("./output2/DT - MaxDepth.jpg")
    plt.cla()

    # Vary the min_samples_splits parameter from 1 to 15
    min_samples_split = range(2, 15)
    # random_search decision tree results
    DT_maxdepth_tuned = DecisionTreeClassifier(max_depth=14, min_samples_leaf=2, min_samples_split=11,
                                               random_state=322)
    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(DT_maxdepth_tuned, x_train, y_train,
                                                 param_name="min_samples_split", param_range=min_samples_split, cv=cv)

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Validation Curve with Decision trees')
    plt.plot(min_samples_split, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(min_samples_split, test_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(min_samples_split, train_mean - train_std, train_mean + train_std, alpha=0.15, color='r')
    plt.fill_between(min_samples_split, test_mean - test_std,
                     test_mean + test_std, alpha=0.15, color='g')

    # Visual aesthetics
    plt.legend(loc='best')
    plt.xlabel('min_samples_splits')
    plt.ylabel('Score')
    plt.ylim([-0.05, 1.05])
    plt.savefig("./output2/DT - MinSSplits.jpg")
    plt.cla()

    # final dt
    dt_final = DecisionTreeClassifier(max_depth=14, criterion='entropy', min_samples_leaf=2, min_samples_split=11,
                                      random_state=322, splitter='random').fit(x_train, y_train)

    # learning curve
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(dt_final, x_train, y_train, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), return_times=True)

    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Decision trees")
    plt.savefig("./output2/DT - final.jpg")
    plt.cla()

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=322)
    title = "Learning Curves (Untuned Decision Tree)"
    plot_learning_curve(dt, title, X_scaled, Y, axes=axes[:, 0], ylim=(0.0, 1.01), cv=cv)

    title = r"Learning Curves (Tuned Decision Tree)"
    plot_learning_curve(dt_final, title, X_scaled, Y, axes=axes[:, 1], ylim=(0.0, 1.01), cv=cv)
    plt.savefig("./output2/DT - learning curve.jpg")
    plt.cla()
    return


def nnetwork(x_train, x_test, y_train, y_test):
    # y_train_encoded = to_categorical(y_train, 6)
    # y_test_encoded = to_categorical(y_test, 6)
    #
    # # Initialize the ANN
    # classifier = Sequential()
    # # Adding the input layer and the first hidden layer
    # classifier.add(Dense(units=9, kernel_initializer='uniform', activation='relu'))
    # # Adding the second hidden layer
    # classifier.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))
    # # Adding the output layer
    # classifier.add(Dense(units=6, kernel_initializer='uniform', activation='softmax'))
    # # Compiling the ANN
    # classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # print(x_train.shape, y_train_encoded.shape, x_test.shape, y_test_encoded.shape, "3333")
    # # Fitting the ANN to the Training set
    # history = classifier.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=100,
    #                          epochs=1150)
    #
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    # t = f.suptitle('ANN Performance', fontsize=12)
    # f.subplots_adjust(top=0.85, wspace=0.3)
    #
    # # epochs = list(range(1,1151))
    # ax1.plot(range(1150), history.history['accuracy'], label='Train Accuracy')
    # ax1.plot(range(1150), history.history['val_accuracy'], label='Validation Accuracy')
    # plt.tick_params(direction="in", labelsize=12)
    # ax1.set_ylabel('Accuracy Value')
    # ax1.set_xlabel('Epoch')
    # ax1.set_title('Accuracy')
    # l1 = ax1.legend(loc="best")
    #
    # ax2.plot(range(1150), history.history['loss'], label='Train Loss')
    # ax2.plot(range(1150), history.history['val_loss'], label='Validation Loss')
    # plt.tick_params(direction="in", labelsize=12)
    # ax2.set_ylabel('Loss Value')
    # ax2.set_xlabel('Epoch')
    # ax2.set_title('Loss')
    # l2 = ax2.legend(loc="best")
    # f.savefig("./output2/NN - Accuracy&Loss.jpg")
    #

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
    f.savefig("./output2/NN - Accuracy&Loss2.jpg")

    return


def boosting(x_train, x_test, y_train, y_test):
    # defalut model
    GBC = GradientBoostingClassifier().fit(x_train, y_train)
    print("defalut model GBC: ", GBC.score(x_test, y_test))

    # Random search hyperparameter tuning
    parameters = {
        "n_estimators": [5, 50, 100, 150, 200, 250, 300],
        "max_depth": [1, 3, 5, 7, 9],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.1, 1]
    }

    n_iter_search = 300
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=322)
    random_search = RandomizedSearchCV(GBC, param_distributions=parameters, cv=cv, n_iter=n_iter_search,
                                       random_state=322)
    start = time()
    random_search.fit(x_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    best_estimator_GBC = random_search.best_estimator_
    print("best_estimator_GBC : ", best_estimator_GBC, random_search.best_score_)

    # final GBC
    GBC_final = GradientBoostingClassifier(max_depth=1).fit(x_train, y_train)

    # validation curve
    # Vary the max_depth parameter from 1 to 15
    max_depth = [0, 1, 5, 10, 15, 20, 25, 30]

    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(GBC_final, x_train, y_train, param_name="max_depth",
                                                 param_range=max_depth, cv=cv)

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Validation Curve with GBC on breast cancer dataset')
    plt.plot(max_depth, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(max_depth, test_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(max_depth, train_mean - train_std, train_mean + train_std, alpha=0.15, color='r')
    plt.fill_between(max_depth, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')

    # Visual aesthetics
    plt.legend(loc='best')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Score')
    plt.ylim([-0.05, 1.05])
    plt.savefig("./output2/GBC - MaxDepth.jpg")
    plt.cla()

    # learning curve
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(GBC_final, x_train, y_train, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Gradient Boosting Classifier")
    plt.savefig("./output2/GBC - learning curve.jpg")
    plt.cla()
    return


def SVM(x_train, x_test, y_train, y_test):
    # linear
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    scores = []
    for i in kernels:
        clf = svm.SVC(kernel=i)
        clf.fit(x_train, y_train)
        scores.append(clf.score(x_test, y_test))

    plt.figure(figsize=(10, 5))
    # creating the bar plot:inear
    plt.bar(kernels, scores, width=0.4)
    plt.xlabel("Kernals")
    plt.ylabel("Accuracy")
    plt.title("SVM performance of different kernels on breast cancer dataset")
    plt.savefig("./output2/SVM - kernals.jpg")
    plt.cla()

    # defining parameter range
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['linear']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid.fit(x_train, y_train)
    # print best parameter after tuning
    print("grid.best_params: ", grid.best_params_, grid.best_estimator_)

    # best
    print("best score:", grid.best_estimator_.score(x_test, y_test))
    my_best = SVC(C=10, gamma=1)
    my_best.fit(x_train, y_train)
    print(my_best.score(x_test, y_test))

    # different gamma
    param_range = np.logspace(-6, -0, 6)
    SVC_best = SVC(C=10, gamma=1, random_state=322)
    train_scores, test_scores = validation_curve(
        SVC(C=10), x_train, y_train, param_name="gamma", param_range=param_range, scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM on breast cancer dataset")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("./output2/SVM - gamma.jpg")
    plt.cla()

    # different C
    param_range = np.logspace(-3, 3, 7)
    SVC_best = SVC(C=10, gamma=1, random_state=322)
    train_scores, test_scores = validation_curve(
        SVC(), x_train, y_train, param_name="gamma", param_range=param_range, scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM on breast cancer dataset")
    plt.xlabel(r"C")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig("./output2/SVM - C.jpg")
    plt.cla()

    # learning curve
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=322)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(SVC_best, x_train, y_train, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Support Vector Classifier on breast cancer data")
    plt.savefig("./output2/SVM - learning curve1.jpg")
    plt.cla()
    # learning curves
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    title = "Learning Curve for svc"
    plot_learning_curve(SVC(), title, x_train, y_train, axes=axes[:, 0], ylim=(0.0, 1.01), cv=cv)
    title = r"Learning Curve for SVC"
    plot_learning_curve(SVC(), title, x_train, y_train, axes=axes[:, 1], ylim=(0.0, 1.01), cv=cv)
    plt.savefig("./output2/SVM - learning curve2.jpg")
    plt.cla()
    return


def knn(x_train, x_test, y_train, y_test):
    # DEFAULT module
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    print("DEFAULT knn module : ", knn.score(x_test, y_test))
    # validation curve
    # Vary the max_depth parameter from 1 to 15
    n_neighbors = range(1, 20)
    # random_search decision tree results

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=322)
    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(KNeighborsClassifier(), x_train, y_train, param_name="n_neighbors",
                                                 param_range=n_neighbors, cv=cv)
    print("train_scores", train_scores)
    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Validation Curve with kNN')
    plt.plot(n_neighbors, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(n_neighbors, test_mean, 'o-', color='g', label='Validation Score')

    # Visual aesthetics
    plt.legend(loc='best')
    plt.xlabel('n_neighbors')
    plt.ylabel('Score')
    plt.ylim([-0.05, 1.05])
    plt.savefig("./output2/KNN - N.jpg")
    plt.cla()

    # learning curve
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(KNeighborsClassifier(n_neighbors=14), x_train, y_train, cv=cv,
                       train_sizes=np.linspace(.1, 1.0, 5), return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("kNN Classifier on breast cancer data")
    plt.savefig("./output2/KNN - learning curve1.jpg")
    plt.cla()

    # learning curves
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    title = "Learning Curve for Untuned KNN"
    plot_learning_curve(KNeighborsClassifier(), title, x_train, y_train, axes=axes[:, 0], ylim=(0.0, 1.01), cv=cv)

    title = r"Learning Curve for Tuned KNN"
    plot_learning_curve(KNeighborsClassifier(n_neighbors=14), title, x_train, y_train, axes=axes[:, 1],
                        ylim=(0.0, 1.01),
                        cv=cv)
    plt.savefig("./output2/KNN - learning curve2.jpg")
    plt.cla()
    return


if __name__ == '__main__':
    # load the data
    data = pd.read_csv("input/breast-cancer.csv")

    # Process data and distribute it according to 3:7
    x_train, x_test, y_train, y_test = data_process(data)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # display the data
    # display(data)

    # Implementation    of    five    algorithms
    # decisionTree(x_train, x_test, y_train, y_test)

    nnetwork(x_train, x_test, y_train, y_test)

    # boosting(x_train, x_test, y_train, y_test)

    # SVM(x_train, x_test, y_train, y_test)

    # knn(x_train, x_test, y_train, y_test)
