import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error


# load training data
training_data = np.load('data/training_data.npy')
prices = np.load('data/prices.npy')

# shuffle
training_data, prices = shuffle(training_data, prices, random_state = 0)

# EX 1

def normalize_data(train_data, test_data):

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_data)
    scaler_train = scaler.transform(train_data)
    scaler_test = scaler.transform(test_data)

    return scaler_train, scaler_test

# EX 2

# 3 - fold => 3 samples per fold
num_samples_fold = len(training_data) // 3

# Split train in 3 folds
training_data_1, prices_1 = training_data[:num_samples_fold], prices[:num_samples_fold]
training_data_2, prices_2 = training_data[num_samples_fold: 2 * num_samples_fold], prices[num_samples_fold: 2 * num_samples_fold]
training_data_3, prices_3 = training_data[2 * num_samples_fold:], prices[2 * num_samples_fold:]

def step(train_data, train_labels, test_data, test_labels, model):

    # normalize train_data & test_data
    normalized_train, normalized_test = normalize_data(train_data, test_data)
    # train a model of linear regression
    model_reg = model.fit(normalized_train, train_labels)
    predict = model_reg.predict(normalized_test)
    # MAE & MSE
    mae = mean_absolute_error(test_labels, predict)
    mse = mean_squared_error(test_labels, predict)

    return mae, mse

model = LinearRegression()
# Run 1
mae_1, mse_1 = step(np.concatenate((training_data_1, training_data_3)), np.concatenate((prices_1, prices_3)), training_data_2, prices_2, model)

# Run 2
mae_2, mse_2 = step(np.concatenate((training_data_1, training_data_2)), np.concatenate((prices_1, prices_2)), training_data_3, prices_3, model)

# Run 3
mae_3, mse_3 = step(np.concatenate((training_data_2, training_data_3)), np.concatenate((prices_2, prices_3)), training_data_1, prices_1, model)

# MAE & MSE for linear regression
print("LINEAR REGRESSION VALUES:")
print("MAE 1: ", mae_1)
print("MSE 1: ", mse_1)
print("MAE 2: ", mae_2)
print("MSE 2: ", mse_2)
print("MAE 3: ", mae_3)
print("MSE 3: ", mse_3)

mae = (mae_1 + mae_2 + mae_3) / 3
print("MEAN MAE: ", mae)
mse = (mse_1 + mse_2 + mse_3) / 3
print("MEAN MSE: ", mse, "\n")

# EX 3

for alpha_ in [1, 10, 100, 1000]:
    model = Ridge(alpha = alpha_)

    print("RIDGE REGRESSION WHERE ALPHA IS ", alpha_)

    # Run 1
    mae_1, mse_1 = step(np.concatenate((training_data_1, training_data_3)), np.concatenate((prices_1, prices_3)), training_data_2, prices_2, model)

    # Run 2
    mae_2, mse_2 = step(np.concatenate((training_data_1, training_data_2)), np.concatenate((prices_1, prices_2)), training_data_3, prices_3, model)

    # Run 3
    mae_3, mse_3 = step(np.concatenate((training_data_2, training_data_3)), np.concatenate((prices_2, prices_3)), training_data_1, prices_1, model)

    # MAE & MSE for linear regression
    print("LINEAR REGRESSION VALUES ")
    print("MAE 1: ", mae_1)
    print("MSE 1: ", mse_1)
    print("MAE 2: ", mae_2)
    print("MSE 2: ", mse_2)
    print("MAE 3: ", mae_3)
    print("MSE 3: ", mse_3)

    mae = (mae_1 + mae_2 + mae_3) / 3
    print("MEAN MAE: ", mae)
    mse = (mse_1 + mse_2 + mse_3) / 3
    print("MEAN MSE: ", mse, "\n")

print("BEST PERFORMANCE WHERE ALPHA IS 10")

# EX 4

model = Ridge(10)
scaler = preprocessing.StandardScaler()
scaler.fit(training_data)
normalized_train = scaler.transform(training_data)
model.fit(normalized_train, prices)

print("COEF: ", model.coef_)
print("BIAS: ", model.intercept_, "\n")

features = ["Year",
            "Kilometers Driven",
            "Mileage",
            "Engine",
            "Power",
            "Seats",
            "Owner Type",
            "Fuel Type",
            "Transmission"]

#print("FEATURES ARE ", features)

max_index = np.argmax(np.abs(model.coef_))
most_significant_feature = features[int(max_index)]
print("MOST SIGNIFICANT FEATURE IS ", most_significant_feature)

second_most_significant_feature = features[(max_index + 1)]
print("SECOND MOST SIGNIFICANT FEATURE IS ", second_most_significant_feature)

min_index = np.argmin(np.abs(model.coef_))
least_significant_feature = features[int(min_index)]
print("LEAST SIGNIFICANT FEATURE IS ", least_significant_feature)