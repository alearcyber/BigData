"""
Homework 2
Aidan Lear
Sept 22, 2022
"""
import sklearn.linear_model as sk
import numpy
import matplotlib.pyplot as plt
figure, axis = plt.subplots(1, 3, figsize=(12, 5))
figure.tight_layout(pad=3)
plt.subplots_adjust(left=0.07)


# The data will be stored as a list of 2-tuples where the tuples are x,y coordinates.
# The x coordinate is the Months and the y coordinates is Sales in thousands of dollars.
sales_data = [(3, 100), (5, 250), (7, 330), (9, 590), (12, 660), (15, 780), (18, 890)]

def linear_regression(data):
    """
    Given a list of points [(x1,y2), ... ,(xn, yn)] calculate the slope and y intercept with a linear regression.
    This is my own linear regression model implementation programmed from scratch.
    """
    #allocate space for summations
    x_sum = y_sum = x_square_sum = y_square_sum = xy_sum = 0

    #Perform summations
    for x, y in data:
        x_sum += x
        y_sum += y
        x_square_sum += x**2
        y_square_sum += y**2
        xy_sum += x * y

    #number of data points
    n = len(data)

    #calculate slope with the slope equation
    slope = (xy_sum - (x_sum * y_sum)/n) / (x_square_sum - (x_sum**2/n))

    #calculate y-intercept with the y-intercept equation
    y_intercept = (y_sum/n) - (slope * (x_sum/n))

    #return the results as a 2-tuple. The data is stored as follows (slope, y-intercept)
    return slope, y_intercept




def sklearn_linear_regression(data):
    """
    Given a list of points [(x1,y2), ... ,(xn, yn)] calculate the slope and y intercept with a linear regression.
    This function is simply a wrapper for usage of the linear regression functionality in sklearn.
    """
    #instantiate the linear regression object from sklearn
    regression_model = sk.LinearRegression()

    #Sperate x and y into separate lists
    x_values = numpy.array([x for x, y in data]).reshape((-1, 1))  # unpack x values, ensure the correct array shape
    y_values = [y for x, y in data]  # unpack y values

    #call the fit() function from the LinearRegression object
    regression_model.fit(x_values, y_values)

    #access slope and y-intercept
    slope, y_intercept = regression_model.coef_[0], regression_model.intercept_

    #return the results: the slope and y_intercept
    return slope, y_intercept




def question_one():
    """
    What is the expected product sales for the next year (next 12 months)?
    What are the inferences (α, β values) conveyed through this predictive linear regression model?
    """
    print('--QUESTION ONE--')
    #calculate the prediction equation
    slope, y_intercept = linear_regression(sales_data)
    print(f'The linear regression model creates the following prediction equation:\n\ty = {slope} x + {y_intercept}')


    #expected product sales for the next year
    next_year = 18 + 12  # current month + 12 months (1 year)
    expected_sales = (next_year * slope) + y_intercept  # prediction equation
    what_are_the_expected_sales = f'\nThe expected product sales for next year, in 12 months ' \
                                  f'from now, is {round(expected_sales, 2)} thousand dollars.'
    print(what_are_the_expected_sales)



    #what does then slope convey?
    what_is_conveyed_by_slope = f'\nThe slope in this case estimates that for every additional\n' \
                                f'month the company is in business, The expected product sales\n' \
                                f'will increase by about {round(slope, 2)} thousand dollars.'
    print(what_is_conveyed_by_slope)



    #what does the y-intercept convey?
    what_is_conveyed_by_y_intercept = f'\nThe y-intercept estimates that at 0 months into the business,' \
                                      f'\nthe sales were at {round(y_intercept, 2)} thousand dollars.' \
                                      f'\nThis indicates that the company had to invest {abs(round(y_intercept, 2))}' \
                                      f'\nthousand dollars in overhead costs to begin sales.'
    print(what_is_conveyed_by_y_intercept)





def question_two():
    """
    Company M&M wants to invest in a new product ABC if the current product XYZ has not produced a 1.5 times increase
    in sales over the next year. As a Data Scientist, would you advise company M&M to invest in a new product ABC
    or make changes to the current product XYZ? Provide your reasoning based on facts and figures to substantiate
    your decision-making process.
    """
    print('\n\n--QUESTION TWO--')

    #perform the linear regression
    slope, y_intercept = linear_regression(sales_data)

    #Calculate the expected sales a year from now.
    next_year = 18 + 12  # current month + 12 months (1 year)
    expected_sales = (next_year * slope) + y_intercept  # prediction equation
    current_sales = 890  # taken directly from the data

    # divide sales a year from now to find the multiplicative increase
    multiplicative_increase = round(expected_sales / current_sales, 2)

    #print the results
    print(f'The product is estimated to produce a {multiplicative_increase} times increase in sales.')

    #Give advice. A 1.5 times increase in sales is the goal
    if multiplicative_increase >= 1.5:
        print('The estimated increase in sales is bigger than the goal of 1.5, as such,\n'
              'the company should NOT invest in the new product. The current product is fine.')
    else:
        print('The estimated increase in sales is smaller than the goal of 1.5, as such,\n'
              'the company should invest in the new product.')

    #create a plot with the prediction line and goal sales
    x = numpy.linspace(-2, 40, 100)
    y = slope * x + y_intercept
    axis[0].plot(x, y, linestyle='solid', label='Prediction Equation')
    axis[0].plot(30, 1335, 'ro', label='1.5x increase in a year')
    axis[0].legend()
    axis[0].set(xlabel='Months', ylabel='Sales (in K$)', title='Sales Outlook, For Question 2')





def question_three():
    """
    Compare results obtained using linear regression function from sklearn with your own linear regression model.
    Compare and provide data visualization (scatter plot) and plot the regression line for all cases.
    """
    print('\n\n--QUESTION THREE--')

    #perform linear regression with both the sklearn implementation and my own
    homemade_slope, homemade_intercept = linear_regression(sales_data)
    sklearn_slope, sklearn_intercept = sklearn_linear_regression(sales_data)

    #display the results to the console
    print(f'sklearn prediction equation -> y = {sklearn_slope}x + {sklearn_intercept}')
    print(f'Homemade prediction equation -> y = {homemade_slope}x + {homemade_intercept}')
    print(f'The equations are identical! Wooo!')

    #create the scatter plots
    axis[1].set(xlabel='Months', ylabel='Sales (in K$)', title='Scikit-learn Regression')
    axis[2].set(xlabel='Months', ylabel='Sales (in K$)', title='My Own Regression')
    axis[1].scatter([x for x, y in sales_data], [y for x, y in sales_data])
    axis[2].scatter([x for x, y in sales_data], [y for x, y in sales_data])

    #place the line for Scikit-learn
    sk_x = numpy.linspace(-2, 20, 100)
    sk_y = sklearn_slope * sk_x + sklearn_intercept
    axis[1].plot(sk_x, sk_y, linestyle='solid', label='sklearn equation', color='red')
    axis[1].legend()

    #place the line for my implementation
    my_x = numpy.linspace(-2, 20, 100)
    my_y = homemade_slope * my_x + homemade_intercept
    axis[2].plot(my_x, my_y, linestyle='solid', label='My equation', color='orange')
    axis[2].legend()





def main():
    """entry point"""
    question_one()
    question_two()
    question_three()
    plt.show()


if __name__ == '__main__':
    main()
