import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




"""
Load in and preprocess the Iris dataset
"""
# load iris dataset, convert to pandas dataframe
iris = load_iris()
data = load_iris()
iris_df = pd.DataFrame(data=np.c_[data['data'], iris['target']], columns=data['feature_names'] + ['target'])

# fix the species label
def translate_species(n):
    """ Helper function to translate the species value from a number to a descriptive name"""
    if n == 0:
        return 'setosa'
    elif n == 1:
        return 'versicolor'
    return 'virginica'
iris_df['target'] = iris_df['target'].apply(translate_species)
iris_df = iris_df.rename(columns={'target': 'species'})



def data_visualization():
    """
    ----QUESTION 1----
    Python program for Iris data visualization.
    """

    """
    a) i) Correlation Matrix Heatmap
    """

    # set up the correlation matrix with seaborn
    cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    correlation_matrix = np.corrcoef(iris_df[cols].values.T)
    sns.set(font_scale=1.5)
    sns.heatmap(correlation_matrix, cbar=True, annot=True, square=True, fmt='0.2f',
                annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
    plt.show()


    """
    a) ii) Feature Distribution Analysis
    """
    sns.pairplot(iris_df, hue='species')
    plt.show()




def linear_regression_analysis():
    """
    ----QUESTION 2----
    Linear Regression analysis on the Iris dataset.
    """
    """
    a) Drop the ‘petal length’ feature and train the LR model on:
    i) 30% samples (i.e. train size = 0.3, test size = 0.7)
    """

    # create variables
    iris_nospecies = iris_df.drop('species', axis=1)
    X = iris_nospecies.drop(labels='petal length (cm)', axis=1)
    y = iris_nospecies['petal length (cm)']

    # split data into corresponding sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=111)


    # create and train linear regression
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)

    #Quantatative Analysis
    print('----Train Size = 0.3----')
    print('Slope:', linear_regression.coef_)
    print('Y-Intercept:', linear_regression.intercept_)
    print('Coefficient of Determination:', r2_score(y_test, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test,  y_pred)))
    print('MSE:', mean_squared_error(y_test, y_pred))




    """
    ii) 70% samples (i.e. train size = 0.7, test size = 0.3)
    """
    #create new data split based on different train size
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=111)

    #create and train linear regression on new data split
    linear_regression_70 = LinearRegression()
    linear_regression_70.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)

    #new quantitative analysis
    print('\n----Train Size = 0.7----')
    print('Slope:', linear_regression.coef_)
    print('Y-Intercept:', linear_regression.intercept_)
    print('Coefficient of Determination:', r2_score(y_test, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test,  y_pred)))
    print('MSE:', mean_squared_error(y_test, y_pred))




    """
    b) Petal Length Prediction
    """
    sample = iris_df.loc[103]
    actual_petal_length = sample[2]
    sample = {'sepal length (cm)': [sample[0]],
              'sepal width (cm)': [sample[1]],
              #'petal length (cm)': [sample[2]],
              'petal width (cm)': [sample[3]],
              #'species': sample[4]
              }
    sample = pd.DataFrame(data=sample)

    print('\n----30% prediction----')
    prediction = linear_regression.predict(sample)
    print('Predicted Sepal Length:', prediction[0])
    print('Actual Petal Length:', actual_petal_length)
    print('Prediction RMSE=', round(abs(prediction[0] - actual_petal_length), 3))

    print('\n----70% prediction----')
    prediction_70 = linear_regression_70.predict(sample)
    print('Predicted Sepal Length:', prediction_70[0])
    print('Actual Petal Length:', actual_petal_length)
    print('Prediction RMSE=', round(abs(prediction_70[0] - actual_petal_length), 3))



def main():
    """entry point"""
    #data_visualization() #question 1
    linear_regression_analysis() # question 2



if __name__ == '__main__':
    main()
