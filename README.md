# House-pricing-model

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice


features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']


X = home_data[features]
X.head()


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
rf_model_on_full_data = RandomForestRegressor(random_state=1)


rf_model_on_full_data.fit(X,y)
test_data_path = '../input/test.csv'


test_data = pd.read_csv(test_data_path)


test_X = test_data[features]



test_preds = rf_model_on_full_data.predict(test_X)
rf_model_on_full_data = RandomForestRegressor()
rf_model_on_full_data.fit(X, y)
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)


output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)[train.csv](https://github.com/Sourav-Tripathy/House-pricing-model/files/9542253/train.csv)
[test.csv](https://github.com/Sourav-Tripathy/House-pricing-model/files/9542254/test.csv)
[data_description.txt](https://github.com/Sourav-Tripathy/House-pricing-model/files/9542255/data_description.txt)
[submission.csv](https://github.com/Sourav-Tripathy/House-pricing-model/files/9542256/submission.csv)
