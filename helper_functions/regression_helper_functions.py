import pandas as pd
import matplotlib.pyplot as plt
import math


def least_squares(data):
    x_sum = 0
    y_sum = 0
    x_sq_sum = 0
    xy_sum = 0

    # the list of data should have two equal length columns
    assert len(data) == 2
    assert len(data[0]) == len(data[1])

    n = len(data[0])
    # least squares regression calculation
    for i in range(0, n):
        if isinstance(data[0][i],str):
            x = int(data[0][i]) # convert date string to int
        else:
            x = data[0][i] # for GDP vs life-expect data
        y = data[1][i]
        x_sum = x_sum + x
        y_sum = y_sum + y
        x_sq_sum = x_sq_sum + (x**2)
        xy_sum = xy_sum + (x*y)

    m = ((n * xy_sum) - (x_sum * y_sum))
    m = m / ((n * x_sq_sum) - (x_sum ** 2))
    c = (y_sum - m * x_sum) / n

    print("Results of linear regression:")
    print("m =", m, "c =", c)

    return m, c

def get_model_predictions(x_data, m, c):
    linear_preds = []
    for x in x_data:
        # FIXME: Uncomment below line and complete the line of code to get a model prediction from each x value
#         y = 
        # ANSWER
        y = m * x + c
        
        #add the result to the linear_data list
        linear_preds.append(y)
    return(linear_preds)

def make_regression_graph(x_data, y_data, y_pred, axis_labels):
    plt.scatter(x_data, y_data, label="Data")
    plt.plot(x_data, y_pred, label="Line of best fit")
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.grid()
    plt.legend()

    plt.show()
    
# Calculating RMS (root mean square)
def measure_error(data,preds):
    assert len(data)==len(preds)
    err_total = 0
    for i in range(0,len(data)):
        # FIXME: Uncomment the below line and fill in the blank to add up the squared error for each observation
#         err_total = err_total + ________
        err_total = err_total + (data[i] - preds[i])**2 

    err = math.sqrt(err_total / len(data))
    return err
    
def process_life_expectancy_data(filename, country, train_data_range, test_data_range=None):
    min_date_train = train_data_range[0]
    max_date_train = train_data_range[1]
    df = pd.read_csv(filename, index_col="Life expectancy")

    # get the data used to estimate line of best fit (life expectancy for specific country across some date range)
    # we have to convert the dates to strings as pandas treats them that way
    y_data_train = df.loc[country, str(min_date_train):str(max_date_train)]

    # create a list with the numerical range of min_date to max_date
    # we could use the index of life_expectancy but it will be a string
    # we need numerical data
    x_data_train = list(range(min_date_train, max_date_train + 1))

    # calculate line of best fit
    # FIXME: Uncomment the below line of code and fill in the blank
#     m, c = _______([x_data, life_expectancy])
    m, c = least_squares([x_data_train, y_data_train])

    # FIXME: Uncomment the below line of code and fill in the blank
#     model_preds = _______(x_data, m, c)
    y_preds_train = get_model_predictions(x_data_train, m, c)
    
    # FIXME: Uncomment the below line of code and fill in the blank
#     error = _______(life_expectancy, model_preds)
    train_error = measure_error(y_data_train, y_preds_train)    
    print("Train RMSE =", train_error)
    make_regression_graph(x_data_train, y_data_train, y_preds_train, ['Year', 'Life Expectancy'])
    
    # Test RMSE
    if test_data_range is not None:
        min_date_test = test_data_range[0]
        if len(test_data_range)==1:
            max_date_test=min_date_test
        else:
            max_date_test = test_data_range[1]
        x_data_test = list(range(min_date_test, max_date_test + 1))
        y_data_test = df.loc[country, str(min_date_test):str(max_date_test)]
        y_preds_test = get_model_predictions(x_data_test, m, c)
        test_error = measure_error(y_data_test, y_preds_test)    
        print("Test RMSE =", test_error)
        make_regression_graph(x_data_train+x_data_test, pd.concat([y_data_train,y_data_test]), y_preds_train+y_preds_test, ['Year', 'Life Expectancy'])

    return m, c


def read_data(gdp_file, life_expectancy_file, year):
    df_gdp = pd.read_csv(gdp_file, index_col="Country Name")

    gdp = df_gdp.loc[:, year]

    df_life_expt = pd.read_csv(life_expectancy_file,
                               index_col="Life expectancy")

    # get the life expectancy for the specified country/dates
    # we have to convert the dates to strings as pandas treats them that way
    life_expectancy = df_life_expt.loc[:, year]

    data = []
    for country in life_expectancy.index:
        if country in gdp.index:
            # exclude any country where data is unknown
            if (math.isnan(life_expectancy[country]) is False) and \
               (math.isnan(gdp[country]) is False):
                    data.append((country, life_expectancy[country],
                                 gdp[country]))
            else:
                print("Excluding ", country, ",NaN in data (life_exp = ",
                      life_expectancy[country], "gdp=", gdp[country], ")")
        else:
            print(country, "is not in the GDP country data")

    combined = pd.DataFrame.from_records(data, columns=("Country",
                                         "Life Expectancy", "GDP"))
    combined = combined.set_index("Country")
    # we'll need sorted data for graphing properly later on
    combined = combined.sort_values("Life Expectancy")
    return combined

def process_data(gdp_file, life_expectancy_file, year):
    data = read_data(gdp_file, life_expectancy_file, year)

    gdp = data["GDP"].tolist()
    gdp_log = data["GDP"].apply(math.log).tolist()
    life_exp = data["Life Expectancy"].tolist()

    m, c = least_squares([life_exp, gdp_log])

    # list for logarithmic version
    log_data = []
    # list for raw version
    linear_data = []
    for x in life_exp:
        y_log = m * x + c
        log_data.append(y_log)

        y = math.exp(y_log)
        linear_data.append(y)

    # uncomment for log version, further changes needed in make_regression_graph too
    make_regression_graph(life_exp, gdp_log, log_data, ['Life Expectancy', 'log(GDP)'])
    make_regression_graph(life_exp, gdp, linear_data, ['Life Expectancy', 'GDP'])

    train_error = measure_error(linear_data, gdp)
    print("Train RMSE =", train_error)