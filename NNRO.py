from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, ShuffleSplit, GridSearchCV, RandomizedSearchCV, train_test_split, \
    learning_curve, ShuffleSplit


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


def RHCNN(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot):
    # Initialize neural network object and fit object - attempt 1
    nn_model1 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                           algorithm='random_hill_climb',
                                           max_iters=3000, bias=True, is_classifier=True,
                                           learning_rate=0.1, early_stopping=True, curve=True,
                                           clip_max=5, max_attempts=100, random_state=3)

    nn_model1.fit(X_train_scaled, y_train_hot)

    y_train_pred1 = nn_model1.predict(X_train_scaled)

    y_train_accuracy1 = accuracy_score(y_train_hot, y_train_pred1)

    print("y_train_accuracy1", y_train_accuracy1)

    # Predict labels for test set and assess accuracy
    y_test_pred1 = nn_model1.predict(X_test_scaled)

    y_test_accuracy1 = accuracy_score(y_test_hot, y_test_pred1)

    print("y_test_accuracy", y_test_accuracy1)

    # restart=5
    nn_model1r5 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                             algorithm='random_hill_climb',
                                             max_iters=3000, bias=True, is_classifier=True,
                                             learning_rate=0.1, restarts=5, early_stopping=True, curve=True,
                                             clip_max=5, max_attempts=100, random_state=3)

    nn_model1r5.fit(X_train_scaled, y_train_hot)

    y_train_pred1r5 = nn_model1r5.predict(X_train_scaled)

    y_train_accuracy1r5 = accuracy_score(y_train_hot, y_train_pred1r5)

    print("y_train_accuracy1r5", y_train_accuracy1r5)

    # Predict labels for test set and assess accuracy
    y_test_pred1r5 = nn_model1r5.predict(X_test_scaled)

    y_test_accuracy1r5 = accuracy_score(y_test_hot, y_test_pred1r5)

    print("y_test_accuracy1r5", y_test_accuracy1r5)
    print("nn_model1r5.loss", nn_model1r5.loss)

    nn_model1r10 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                              algorithm='random_hill_climb',
                                              max_iters=3000, bias=True, is_classifier=True,
                                              learning_rate=0.1, restarts=10, early_stopping=True, curve=True,
                                              clip_max=5, max_attempts=100, random_state=3)

    nn_model1r10.fit(X_train_scaled, y_train_hot)

    y_train_pred1r10 = nn_model1r10.predict(X_train_scaled)

    y_train_accuracy1r10 = accuracy_score(y_train_hot, y_train_pred1r10)

    print("y_train_accuracy1r10", y_train_accuracy1r10)
    # Predict labels for test set and assess accuracy
    y_test_pred1r10 = nn_model1r10.predict(X_test_scaled)

    y_test_accuracy1r10 = accuracy_score(y_test_hot, y_test_pred1r10)

    print("y_test_accuracy1r10", y_test_accuracy1r10)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(range(0, len(list(nn_model1.fitness_curve))), nn_model1.fitness_curve[..., 0], '-',
            label='RHC restarts=0')  # Plot some data on the axes.
    ax.plot(range(0, len(list(nn_model1r5.fitness_curve))), nn_model1r5.fitness_curve[..., 0], '-',
            label='RHC restarts=5')  # Plot more data on the axes...
    ax.plot(range(0, len(list(nn_model1r10.fitness_curve))), nn_model1r10.fitness_curve[..., 0], '-',
            label='RHC restarts=10')  # ... and some more.
    # ax.plot(range(0,len(list(nn_model4.fitness_curve))),nn_model4.fitness_curve,'-',label='Gradient Descent')
    ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    ax.set_title("RHC fitness/loss curve on different restarts")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    fig.savefig("./output6/RHCNN.jpg")
    cv_method = ShuffleSplit(n_splits=4, test_size=0.2, random_state=6666)
    # # Calculate the training and testing scores
    # train_scores, test_scores = validation_curve(nn_model1, X_train_scaled, y_train_hot, cv = cv)

    # learning curve
    train_sizes, train_scores, test_scores = learning_curve(nn_model1, X_train_scaled, y_train_hot, cv=cv_method)

    # fit_times_mean = np.mean(fit_times,axis = 1)
    # fit_times_std = np.std(fit_times,axis = 1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.cla()
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
    plt.title("Learning curve for NN using RHC")
    fig.savefig("./output6/LC- RHCNN.jpg")
    return nn_model1


def SANN(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot):
    nn_model2 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                           algorithm='simulated_annealing',
                                           max_iters=3000, bias=True, is_classifier=True,
                                           learning_rate=0.1, early_stopping=True, curve=True,
                                           clip_max=5, max_attempts=100, random_state=4)

    nn_model2.fit(X_train_scaled, y_train_hot)

    y_train_pred = nn_model2.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print("y_train_accuracy", y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model2.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print("y_test_accuracy", y_test_accuracy)

    Schedules = mlrose_hiive.GeomDecay(init_temp=1.0, decay=0.5, min_temp=0.001)
    nn_model2s5 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                             algorithm='simulated_annealing', schedule=Schedules,
                                             max_iters=3000, bias=True, is_classifier=True,
                                             learning_rate=0.1, early_stopping=True, curve=True,
                                             clip_max=5, max_attempts=100, random_state=4)

    nn_model2s5.fit(X_train_scaled, y_train_hot)

    y_train_pred = nn_model2s5.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print("y_train_accuracy", y_train_accuracy)
    y_test_pred = nn_model2s5.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print("y_test_accuracy", y_test_accuracy)

    Schedules = mlrose_hiive.GeomDecay(init_temp=1.0, decay=0.1, min_temp=0.001)
    nn_model2s1 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                             algorithm='simulated_annealing', schedule=Schedules,
                                             max_iters=3000, bias=True, is_classifier=True,
                                             learning_rate=0.1, early_stopping=True, curve=True,
                                             clip_max=5, max_attempts=100, random_state=4)

    nn_model2s1.fit(X_train_scaled, y_train_hot)

    y_train_pred = nn_model2s1.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print("y_train_accuracy", y_train_accuracy)
    y_test_pred = nn_model2s1.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print("y_test_accuracy", y_test_accuracy)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    # Plot some data on the axes.
    ax.plot(range(0, len(list(nn_model2.fitness_curve))), nn_model2.fitness_curve[..., 0], '-',
            label='SA decay = 0.95')  # Plot more data on the axes...
    ax.plot(range(0, len(list(nn_model2s5.fitness_curve))), nn_model2s5.fitness_curve[..., 0], '-',
            label='SA decay = 0.5')  # ... and some more.
    ax.plot(range(0, len(list(nn_model2s1.fitness_curve))), nn_model2s1.fitness_curve[..., 0], '-',
            label='SA decay = 0.1')
    ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    ax.set_title("SA fitness/loss curve with different decay")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    fig.savefig("./output6/SANN.jpg")
    ax.cla()
    # cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=6666)
    # # Calculate the training and testing scores
    # train_scores, test_scores = validation_curve(nn_model1, X_train_scaled, y_train_hot, cv = cv)

    # learning curve
    cv_method = ShuffleSplit(n_splits=5, test_size=0.2, random_state=322)
    train_sizes, train_scores, test_scores = learning_curve(nn_model2, X_train_scaled, y_train_hot, cv=cv_method)

    # fit_times_mean = np.mean(fit_times,axis = 1)
    # fit_times_std = np.std(fit_times,axis = 1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.cla()
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
    plt.title("NN Learning Curve using SA")
    fig.savefig("./output6/LC-SANN.jpg")
    return nn_model2s5





def GANN(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot):
    nn_model3 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                           algorithm='genetic_alg',
                                           max_iters=3000, bias=True, is_classifier=True,
                                           learning_rate=0.1, early_stopping=True, curve=True,
                                           clip_max=5, max_attempts=100, random_state=5)

    nn_model3.fit(X_train_scaled, y_train_hot)

    y_train_pred = nn_model3.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print("y_train_accuracy", y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model3.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print("y_test_accuracy", y_test_accuracy)

    nn_model3p100 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                               algorithm='genetic_alg', pop_size=100,
                                               max_iters=3000, bias=True, is_classifier=True,
                                               learning_rate=0.1, early_stopping=True, curve=True,
                                               clip_max=5, max_attempts=100, random_state=5)

    nn_model3p100.fit(X_train_scaled, y_train_hot)

    y_train_predp100 = nn_model3p100.predict(X_train_scaled)

    y_train_accuracyp100 = accuracy_score(y_train_hot, y_train_predp100)

    print("y_train_accuracyp100", y_train_accuracyp100)

    y_test_predp100 = nn_model3p100.predict(X_test_scaled)

    y_test_accuracyp100 = accuracy_score(y_test_hot, y_test_predp100)

    print("y_test_accuracyp100", y_test_accuracyp100)

    nn_model3p300 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                               algorithm='genetic_alg', pop_size=300,
                                               max_iters=3000, bias=True, is_classifier=True,
                                               learning_rate=0.1, early_stopping=True, curve=True,
                                               clip_max=5, max_attempts=100, random_state=5)

    nn_model3p300.fit(X_train_scaled, y_train_hot)

    y_train_predp300 = nn_model3p300.predict(X_train_scaled)

    y_train_accuracyp300 = accuracy_score(y_train_hot, y_train_predp300)

    print("y_train_accuracyp300", y_train_accuracyp300)

    y_test_predp300 = nn_model3p300.predict(X_test_scaled)

    y_test_accuracyp300 = accuracy_score(y_test_hot, y_test_predp300)

    print("y_test_accuracyp300", y_test_accuracyp300)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(range(0, len(list(nn_model3.fitness_curve))), nn_model3.fitness_curve[..., 0], '-',
            label='GA pop=200')  # Plot some data on the axes.
    ax.plot(range(0, len(list(nn_model3p300.fitness_curve))), nn_model3p300.fitness_curve[..., 0], '-',
            label='GA pop=300')  # Plot more data on the axes...
    ax.plot(range(0, len(list(nn_model3p100.fitness_curve))), nn_model3p100.fitness_curve[..., 0], '-',
            label='GA pop=100')  # ... and some more.
    # ax.plot(range(0,len(list(nn_model3p100f.fitness_curve))),nn_model3p100f.fitness_curve[...,0],'-',label='Gradient Descent')
    ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    ax.set_title("GA fitness/loss curve on different population")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    fig.savefig("./output6/GANN.jpg")
    ax.cla()

    nn_model3p100f = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                                algorithm='genetic_alg', pop_size=100,
                                                max_iters=3000, bias=True, is_classifier=True,
                                                learning_rate=0.1, early_stopping=False, curve=True,
                                                clip_max=5, max_attempts=100, random_state=5)

    nn_model3p100f.fit(X_train_scaled, y_train_hot)

    # # Calculate the training and testing scores

    # learning curve
    cv_method = ShuffleSplit(n_splits=5, test_size=0.2, random_state=322)
    train_sizes, train_scores, test_scores = learning_curve(nn_model3, X_train_scaled, y_train_hot, cv=cv_method)

    # fit_times_mean = np.mean(fit_times,axis = 1)
    # fit_times_std = np.std(fit_times,axis = 1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.cla()
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
    plt.title("NN Learning Curve using GA")
    fig.savefig("./output6/LC-GANN.jpg")

    return nn_model3p300


def baseline(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot):
    nn_model4 = mlrose_hiive.NeuralNetwork(hidden_nodes=[15], activation='relu',
                                           algorithm='gradient_descent',
                                           max_iters=3000, bias=True, is_classifier=True,
                                           learning_rate=0.0001, early_stopping=True, curve=True,
                                           clip_max=5, max_attempts=100, random_state=3)

    nn_model4.fit(X_train_scaled, y_train_hot)

    y_train_pred = nn_model4.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print("y_train_accuracy", y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model4.predict(X_test_scaled)

    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print("y_test_accuracy", y_test_accuracy)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    # ax.plot(range(0,len(list(nn_model1.fitness_curve))),nn_model1.fitness_curve[...,0],'-',label='RHC')  # Plot some data on the axes.
    # ax.plot(range(0,len(list(nn_model2.fitness_curve))),nn_model2.fitness_curve[...,0],'-',label='SA')  # Plot more data on the axes...
    # ax.plot(range(0,len(list(nn_model3.fitness_curve))),nn_model3.fitness_curve[...,0],'-',label='GA') # ... and some more.
    ax.plot(range(0, len(list(nn_model4.fitness_curve))), -nn_model4.fitness_curve, '-', label='Gradient Descent')
    ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    ax.set_title("fitness/loss curve of Gradient Descent")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    fig.savefig("./output6/BASELINE.jpg")

    return nn_model4


if __name__ == '__main__':
    data = pd.read_csv("input/breast-cancer.csv")

    # Process data and distribute it according to 3:7
    x_train, x_test, y_train, y_test = data_process(data)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.values.reshape(-1, 1)).todense()
    print(y_train_hot.shape, y_test_hot.shape)
    # m1 = RHCNN(x_train, x_test, y_train_hot, y_test_hot)
    # m2 = SANN(x_train, x_test, y_train_hot, y_test_hot)
    # m3 = GANN(x_train, x_test, y_train_hot, y_test_hot)
    m4 = baseline(x_train, x_test, y_train_hot, y_test_hot)
    #
    # fig, ax = plt.subplots()  # Create a figure and an axes.
    # ax.plot(range(0, len(list(m1.fitness_curve))), m1.fitness_curve[..., 0], '-',
    #         label='RHC')  # Plot some data on the axes.
    # ax.plot(range(0, len(list(m2.fitness_curve))), m2.fitness_curve[..., 0], '-',
    #         label='SA')  # Plot more data on the axes...
    # ax.plot(range(0, len(list(m3.fitness_curve))), m3.fitness_curve[..., 0], '-',
    #         label='GA')  # ... and some more.
    # ax.plot(range(0, len(list(m4.fitness_curve))), -m4.fitness_curve, '-', label='Gradient Descent')
    # ax.set_xlabel('Iterations')  # Add an x-label to the axes.
    # ax.set_ylabel('Fitness')  # Add a y-label to the axes.
    # ax.set_title("fitness curves for finding weights algorithms")  # Add a title to the axes.
    # ax.legend()  # Add a legend.
    # fig.savefig("./output6/4.jpg")
    # plt.cla()
    # fig = plt.figure(figsize=(10, 5))
    # # creating the bar plot
    # plt.bar(["RHC", "SA", "GA", "Gradient Descent"], [m1.loss, m2.loss, m3.loss, m4.loss],
    #         width=0.4)
    # plt.ylabel("Loss")
    # plt.title("Loss of the different approaches")
    # plt.savefig("./output6/4-loss.jpg")
