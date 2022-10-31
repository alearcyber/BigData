"""
Aidan Lear
Homework 4
Big Data, 488
"""
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import scipy.io
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix
import itertools


"""
1) Dimensionality Reduction
    - Iris all PC variance graph
    - Indian Pines all PC variance graph
    - Iris Reduced data Visualization with PCA
    - Iris Reduced data Visualization with LDA
    - Indian Pines Reduced Data Visualization with PCA
    - Indian Pines Reduced Data Visualization with LDA
    note: analysis in paper.
"""
random_state = 69

###########################
#Iris Data all PC Variance
###########################
def iris_pca_variance():
    #Load and organize the Iris Dataset
    iris = load_iris()
    class_labels = pd.DataFrame(data=np.array(iris.target).reshape((-1, 1)), columns=['class'])

    #scale the data
    scaler = MinMaxScaler()
    scaler.fit(iris.data.astype(float))
    iris_data = scaler.transform(iris.data)

    #plot_pca_variance(iris_data)

    #perform PCA
    pca = PCA()
    pc = pca.fit_transform(iris_data)


    x_pca_df = pd.DataFrame(data=pc, columns=[f'PC-{i+1}' for i in range(4)])
    print(x_pca_df)

    for i in range(3):
        color = 'rgb'[i]
        indexesToKeep = class_labels['class'] == i
        plt.scatter(x_pca_df.loc[indexesToKeep, 'PC-1'], x_pca_df.loc[indexesToKeep, 'PC-2'], c=color, s=9)

    plt.title('Iris - Best 2 Principle Components', fontsize=20)
    plt.xlabel('PC-1', fontsize=10)
    plt.ylabel('PC-2', fontsize=10)
    plt.grid()
    plt.show()






##################################
#Indian Pines Data all PC Variance
##################################
def pines_pca_variance():
    pines = scipy.io.loadmat(r'indianR.mat')
    pines_data = np.array(pines['X'])

    #There are two ground truths??
    ground_truth_two = scipy.io.loadmat(r'indian_gth.mat')
    ground_truth_two = {i: j for i, j in ground_truth_two.items() if i[0] != '_'}
    ground_truth_in_a_third_variable = pd.DataFrame({i: pd.Series(j[0]) for i, j in ground_truth_two.items()})


    #data normalization
    scaler = MinMaxScaler()
    scaler.fit(pines_data.astype(float))
    pines_data = scaler.transform(pines_data)

    #plot the pca variances
    #plot_pca_variance(pines_data, n_components=10)

    #Transfrom along principle components
    pca = PCA(n_components=10)
    pc = pca.fit_transform(pines_data)


    #transform the basis
    x_pca = np.matmul(pines_data.transpose(), pc)
    x_pca_df = pd.DataFrame(data=x_pca, columns=[f'PC-{i+1}' for i in range(10)])
    x_pca_df = pd.concat([x_pca_df, ground_truth_in_a_third_variable], axis=1)

    #keep only BEST components
    colors = list('rgbymckrgbymckbr')
    markerm = list('ooooooo+++++++**')
    for i in range(16):
        target = i + 1
        color = colors[i]
        marker = markerm[i]
        indexesToKeep = x_pca_df['gth'] == target
        plt.scatter(x_pca_df.loc[indexesToKeep, 'PC-1'], x_pca_df.loc[indexesToKeep, 'PC-2'], c=color, marker=marker, s=9)

    plt.title('Indian Pines - Best 2 Principle Components', fontsize=20)
    plt.xlabel('PC-1', fontsize=10)
    plt.ylabel('PC-2', fontsize=10)
    plt.legend(list(range(1, 17)))
    plt.grid()
    plt.show()


################################
#LDA for the Iris dataset
################################
def iris_lda():
    iris = load_iris()
    class_labels = pd.DataFrame(data=np.array(iris.target).reshape((-1, 1)), columns=['class'])
    lda = LDA()
    pc = lda.fit_transform(iris.data, iris.target)
    x_lda_df = pd.DataFrame(data=pc, columns=['PC-1', 'PC-2'])

    for i in range(3):
        color = 'rgb'[i]
        indexesToKeep = class_labels['class'] == i
        plt.scatter(x_lda_df.loc[indexesToKeep, 'PC-1'], x_lda_df.loc[indexesToKeep, 'PC-2'], c=color, s=9)

    plt.title('Iris - LDA', fontsize=20)
    plt.xlabel('PC-1', fontsize=10)
    plt.ylabel('PC-2', fontsize=10)
    plt.grid()
    plt.show()


##################################
#LDA for the Indian Pines dataset
##################################
def pines_lda():
    #load the dataset
    pines = scipy.io.loadmat(r'indianR.mat')
    pines_data = np.array(pines['X'])
    pines_data = pines_data.transpose() #ATTEMPT TO FIX IT HERE
    bands, samples = pines_data.shape

    #import ground truth from a separate file
    ground_truth_two = scipy.io.loadmat(r'indian_gth.mat')
    ground_truth_two = {i: j for i, j in ground_truth_two.items() if i[0] != '_'}
    gt = pd.DataFrame({i: pd.Series(j[0]) for i, j in ground_truth_two.items()})

    #data normalization
    scaler = MinMaxScaler()
    scaler.fit(pines_data.astype(float))
    pines_data = scaler.transform(pines_data)

    #do the LDA
    lda = LDA(n_components=2)
    pc = lda.fit_transform(pines_data, pines['gth'][0])

    #transform the basis
    #x_pca = np.matmul(pines_data.transpose(), pc)
    x_pca_df = pd.DataFrame(data=pc, columns=['PC-1', 'PC-2'])
    x_pca_df = pd.concat([x_pca_df, gt], axis=1)

    #graph it
    for i in range(16):
        target = i + 1
        c = 'rgbymckrgbymckbr'[i]
        m = 'ooooooo+++++++**'[i]
        indexes_to_keep = x_pca_df['gth'] == target
        plt.scatter(x_pca_df.loc[indexes_to_keep, 'PC-1'], x_pca_df.loc[indexes_to_keep, 'PC-2'], c=c, marker=m, s=9)

    plt.title('Indian Pines - LDA', fontsize=20)
    plt.xlabel('PC-1', fontsize=10)
    plt.ylabel('PC-2', fontsize=10)
    plt.grid()
    plt.show()





###################################################
#Helper Function For Plotting The Variances FOR PCA
###################################################
def plot_pca_variance(data, n_components=None):
    # calculate principle components
    pc = PCA(n_components=n_components)
    pc.fit(data)  # try with SCALED data instead

    # plot explained variance
    plt.bar(range(1, pc.n_components_ + 1), pc.explained_variance_ratio_, align='center', label='Explained Variance')

    # also plot cumulative variance
    cumulative_variance = []
    total = 0
    for i in range(pc.n_components_):
        total += pc.explained_variance_ratio_[i]
        cumulative_variance.append(total)
    plt.step(range(1, pc.n_components_ + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance',
             color='red')

    # clean up and display the plot
    plt.xticks(range(1, pc.n_components_ + 1), range(1, pc.n_components_ + 1))
    for i in range(pc.n_components_):
        text_label = str(round(100 * pc.explained_variance_ratio_[i], 2)) + '%'
        plt.text(i + 1, pc.explained_variance_ratio_[i], text_label, ha='center')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('\'K\'-th Principle Component')
    plt.legend(loc='center right')
    plt.show()





###################################################
# Confusion Matrix For Question Two
###################################################
def confusion_matrix():
    #load in iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    """
    #load indian pines dataset
    pines = scipy.io.loadmat(r'indianR.mat')
    pines_data = np.array(pines['X'])
    pines_data = pines_data.transpose() #ATTEMPT TO FIX IT HERE
    bands, samples = pines_data.shape
    ground_truth_two = scipy.io.loadmat(r'indian_gth.mat')
    ground_truth_two = {i: j for i, j in ground_truth_two.items() if i[0] != '_'}
    gt = pd.DataFrame({i: pd.Series(j[0]) for i, j in ground_truth_two.items()})


    #REMOVE ZEROS
    combined_data = pd.DataFrame(data=pines_data)
    combined_data = pd.concat([combined_data, gt], axis=1)
    combined_data = combined_data[combined_data.gth != 0]
    combined_data = combined_data.drop('gth', axis=1)
    combined_data = combined_data.to_numpy()
    X = combined_data
    y = pines['gth'][0]
    y = [e for e in y if e != 0] # filter zeros from truth as well
    """

    #data normalization
    scaler = MinMaxScaler()
    scaler.fit(X.astype(float))
    X = scaler.transform(X)


    #learning curves
    classifiers = [
        ('SVM - RBF', SVC(kernel='rbf', random_state=random_state)),
        ('SVM - Poly', SVC(kernel='poly', random_state=random_state)),
        ('Gaussian Naive Bayes', GaussianNB()),
    ]
    """
    c = 0
    for label, classifier in classifiers:
        color = colors[c]
        c += 1
        plot_learning_curve(classifier, X, y, steps=num_steps, label=label, color=color, axes=axes)

    axes[1].set_xlabel('N Training Samples')
    axes[1].set_ylabel('Overall Classification Accuracy')
    axes[1].set_title('Model Evaluation - Cross Validation Accuracy')
    axes[1].legend()

    axes[0].set_xlabel('N Training Samples')
    axes[0].set_ylabel('Training/Recall Accuracy')
    axes[0].set_title('Model Evaluation - Training Accuracy')
    axes[0].legend()
    plt.show() 
    """

    #confusion matrix part
    for label, classifier in classifiers:
        plot_per_class_accuracy(classifier, X, y, label)



def plot_learning_curve(classifier, X, y, steps=10, train_size=np.linspace(0.1, 1.0, 10), label='', color='r', axes=None):
    test_size = 0.3
    estimator = Pipeline([('scaler', MinMaxScaler()), ('classifier', classifier)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    train_scores = []
    test_scores = []
    train_sizes = []
    for i in range(0, X_train.shape[0], X_train.shape[0] // steps):
        if i == 0:
            continue
        X_train_i = X_train[:i, :]
        y_train_i = y_train[:i]
        estimator.fit(X_train_i, y_train_i)
        train_scores.append(estimator.score(X_train_i, y_train_i) * 100)
        test_scores.append(estimator.score(X_test, y_test) * 100)
        train_sizes.append(i + 1)

    if X_train.shape[0] % steps !=0:
        estimator.fit(X_train, y_train)
        train_scores.append(estimator.score(X_train, y_train) * 100)
        test_scores.append(estimator.score(X_test, y_test) * 100)
        train_sizes.append(X_train.shape[0])

    if axes is None:
        _, axes = plt.subplot(2)

    train_s = np.linspace(10, 100, num=5)
    axes[0].plot(train_sizes, test_scores, 'o-', color=color, label=label)
    axes[1].plot(train_sizes, train_scores, 'o-', color=color, label=label)

    print(f'\nTraining Accuracy of {label}: {train_scores[-1]}')
    print(f'Testing Accuracy of {label}: {test_scores[-1]}')
    return plt


def plot_per_class_accuracy(classifier, X, y, label, feature_selection=None):
    n_classes = 3 # 3 for iris, 16 for idnian pines
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=3)
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('classifier', classifier)])
    pipeline.fit(X_train, y_train)
    disp = plot_confusion_matrix(pipeline, X_test, y_test, cmap=plt.cm.Blues)
    plt.title(label)
    plt.show()


    #sensitivty and specificity ->         #specificity = TN/(FP + TN)  #sensitivity = TP/(TP+FN)
    print(f'========={label} - Sensitivity/Specificity==========')
    total = 0
    for row, col in itertools.product(list(range(n_classes)), list(range(n_classes))):
        total += disp.confusion_matrix[row][col]

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for c in range(n_classes):
        class_num = c + 1
        self = disp.confusion_matrix[c][c]
        col_count = 0
        for i in range(n_classes):
            col_count += disp.confusion_matrix[i][c]

        row_count = 0
        for i in range(n_classes):
            row_count += disp.confusion_matrix[c][i]

        TP += self
        TN += total - (col_count + row_count - self)
        FN += row_count - self
        FP += col_count - self
        #sensitivity = round(TP/(TP+FN) * 100)
        #specificity = round(TN/(FP+TN) * 100)
        #print(f'class:{class_num},  sensitivity:{sensitivity},  specificity:{specificity}')
    sensitivity = round(TP/(TP+FN) * 100)
    specificity = round(TN/(FP+TN) * 100)
    print(f'sensitivity:{sensitivity},  specificity:{specificity}')



def part_two():
    #######################################
    #import iris, keep raw, apply PCA, LDA
    #######################################
    iris = load_iris()
    X_iris = iris.data
    target_names = iris.target_names
    #scaler = MinMaxScaler()
    #scaler.fit(X_iris.astype(float))

    #raw iris data
    #X_iris = scaler.transform(X_iris)
    y_iris = iris.target

    #PCA on iris data
    pca = PCA(n_components=2)
    pca_iris = pca.fit_transform(X_iris)

    #lda on iris data
    lda = LDA(n_components=2)
    class_labels = pd.DataFrame(data=np.array(iris.target).reshape((-1, 1)), columns=['class'])
    lda_iris = lda.fit_transform(X_iris, y_iris)




    ###############################################
    #import Indian Pines, keep raw, apply PCA, LDA
    ###############################################
    #import Indian Pines
    pines = scipy.io.loadmat(r'indianR.mat')
    pines_data = np.array(pines['X'])
    pines_data = pines_data.transpose()  # TRANSPOSE FIXES IT
    ground_truth_two = scipy.io.loadmat(r'indian_gth.mat')
    ground_truth_two = {i: j for i, j in ground_truth_two.items() if i[0] != '_'}
    gt = pd.DataFrame({i: pd.Series(j[0]) for i, j in ground_truth_two.items()})

    # Remove Zeros from indian Pines dataset
    combined_data = pd.DataFrame(data=pines_data)
    combined_data = pd.concat([combined_data, gt], axis=1)
    combined_data = combined_data[combined_data.gth != 0]
    combined_data = combined_data.drop('gth', axis=1)
    combined_data = combined_data.to_numpy()
    X_pines = combined_data

    #normalize and remove zero from ground truth
    #scaler = MinMaxScaler()   #normalize
    #scaler.fit(combined_data.astype(float))   #normalize
    #X_pines = scaler.transform(combined_data)   #normalize
    y = pines['gth'][0]
    y_pines = [e for e in y if e != 0]  # filter zeros from truth as well

    #apply PCA to Pines dataset
    pca = PCA(n_components=2)
    pca_pines = pca.fit_transform(X_pines)

    #apply LDA to Pines dataset
    lda = LDA(n_components=2)
    lda_pines = lda.fit_transform(X_pines, y_pines)



    #######################################
    # Setup Classifiers and Training Sizes
    #######################################
    classifiers = [
        ('SVM - RBF', SVC(kernel='rbf', random_state=random_state)),
        ('SVM - Poly', SVC(kernel='poly', random_state=random_state)),
        ('Gaussian Naive Bayes', GaussianNB()),
    ]
    training_sizes = [0.10, 0.20, 0.30, 0.40, 0.50]
    colors = {'SVM - RBF': 'red', 'SVM - Poly': 'green', 'Gaussian Naive Bayes': 'blue'}


    ###################################
    # Make Iris Graphs
    ###################################
    # Iris PCA
    make_graph(pca_iris, y_iris, 'Iris PCA K=2', training_sizes=training_sizes, classifiers=classifiers)

    # Iris LDA
    make_graph(lda_iris, y_iris, 'Iris LDA K=2', training_sizes=training_sizes, classifiers=classifiers)

    # Iris Raw
    make_graph(X_iris, y_iris, 'Iris Original Dimensions', training_sizes=training_sizes, classifiers=classifiers)




    #############################
    # Make Indian Pines Graphs
    #############################
    #Pines PCA
    make_graph(pca_pines, y_pines, 'Pines PCA K=2', training_sizes=training_sizes, classifiers=classifiers)

    #Pines LDA
    make_graph(lda_pines, y_pines, 'Pines LDA K=2', training_sizes=training_sizes, classifiers=classifiers)

    #Pines Raw
    make_graph(X_pines, y_pines, 'Pines Original Dimensions', training_sizes=training_sizes, classifiers=classifiers)


################################
# Helper function for graphing
################################

def make_graph(X, y, title, training_sizes, classifiers, steps=10):
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    train_points = {'SVM - RBF': [], 'SVM - Poly': [], 'Gaussian Naive Bayes': []}  # points to be placed on graph
    test_points = {'SVM - RBF': [], 'SVM - Poly': [], 'Gaussian Naive Bayes': []}
    colors = {'SVM - RBF': 'red', 'SVM - Poly': 'green', 'Gaussian Naive Bayes': 'blue'}
    for training_size in training_sizes:
        print(f'\n====train_size - {training_size}====')
        # create test and train split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_size, random_state=random_state)

        for label, classifier in classifiers:
            estimator = Pipeline([('scaler', MinMaxScaler()), ('classifier', classifier)])
            train_scores = []
            test_scores = []
            train_sizes = []
            for i in range(0, X_train.shape[0], X_train.shape[0] // steps):
                if i == 0:
                    continue
                X_train_i = X_train[:i, :]
                y_train_i = y_train[:i]
                if len(y_train_i) == 1:  # this if statement is key for PCA with the Iris Dataset
                    continue
                # print('lower x_train_i:', X_train_i)
                # print('lower y_train_i:', y_train_i)
                estimator.fit(X_train_i, y_train_i)
                train_scores.append(estimator.score(X_train_i, y_train_i) * 100)
                test_scores.append(estimator.score(X_test, y_test) * 100)
                train_sizes.append(i + 1)

            if X_train.shape[0] % steps != 0:
                estimator.fit(X_train, y_train)
                train_scores.append(estimator.score(X_train, y_train) * 100)
                test_scores.append(estimator.score(X_test, y_test) * 100)
                train_sizes.append(X_train.shape[0])

            # place point on train graph
            # axes[0].plot(training_size, train_scores[-1], 'o-', color=colors[label], label=label)
            train_points[label].append(train_scores[-1])

            # place point on
            # axes[1].plot(training_size, test_scores[-1], 'o-', color=colors[label], label=label)
            test_points[label].append(test_scores[-1])

            print(f'\nTraining Accuracy of {label}: {train_scores[-1]}')
            print(f'Testing Accuracy of {label}: {test_scores[-1]}')

    ######################################
    # label and set up graph
    ######################################
    for key in train_points:
        axes[0].plot([x * 100 for x in training_sizes], train_points[key], 'o-', color=colors[key], label=key)
        axes[1].plot([x * 100 for x in training_sizes], test_points[key], 'o-', color=colors[key], label=key)
    axes[0].set_xlabel('Train Size (%)')
    axes[0].set_ylabel('Training/Recall Accuracy')
    axes[0].set_title(f'{title} - Training Accuracy')
    axes[0].legend()
    axes[1].set_xlabel('Test Size (%)')
    axes[1].set_ylabel('Overall Classification Accuracy')
    axes[1].set_title(f'{title} - Cross Validation Accuracy')
    axes[1].legend()
    plt.show()






if __name__ == '__main__':
    """enter"""
    #iris_pca_variance()
    #pines_pca_variance()
    #iris_lda()
    pines_lda()
    #confusion_matrix()
    #part_two_organized()

