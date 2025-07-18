import os
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import yfinance as yf
from datetime import datetime, timedelta

def scoreDataset(X_train, X_val, y_train, y_val):
    #define the model
    stockModel = RandomForestRegressor(random_state=1,
                                   max_depth=10
                                   )
    stockModel.fit(X_train,y_train)
    stockPredictions = stockModel.predict(X_val)
    stockMae = mean_absolute_error(y_val, stockPredictions)
    maeAverage = 100 * (stockMae/y_val.mean())
    print("Validation MAE for Random Forest Model: {:,.2f}".format(maeAverage), "%")
    latest_data = stockData.iloc[-1]
    tomorrow_features = pd.DataFrame([latest_data[features]])

    # Predict tomorrow's closing price
    tomorrow_pred = stockModel.predict(tomorrow_features)
    print(f"Predicted closing price for tomorrow: ${tomorrow_pred[0]:.2f}")
    findFeatureImportance(stockModel)

def findFeatureImportance(stockModel):
    # After training
    importances = stockModel.feature_importances_
    # Pair each importance with the feature name
    feature_importance = pd.Series(importances, index=features)
    # Sort descending
    feature_importance = feature_importance.sort_values(ascending=False)
    # Print
    print(feature_importance)
ticker = yf.Ticker("CSU.TO")

# Get historical price data (default daily)
stockData = ticker.history(period="1y")  # options: '1d', '5d', '1mo', '1y', '5y', 'max'
stockData["Target"] = stockData["Close"].shift(-1)
stockData.dropna(inplace=True)
stockData["PrevClose"] = stockData["Close"].shift(1)
stockData["Return"] = (stockData["Close"] - stockData["PrevClose"]) / stockData["PrevClose"]
stockData["MA5"] = stockData["Close"].rolling(5).mean()
stockData["Volatility5"] = stockData["Close"].rolling(5).std()


features = ["Open", "High", "Volume", "PrevClose", "Return", "MA5", "Volatility5"]

X = stockData[features]

#split training and test data
splitDate = (datetime.now() - timedelta(days=45)).date()
train = stockData.loc[stockData.index.date < splitDate]
test = stockData.loc[stockData.index.date>= splitDate]

target = 'Target'


X_train, y_train = train[features], train[target]
X_val, y_val = test[features], test[target]
#Create Imputed Data
Imputer = SimpleImputer()
Imputed_X_Train = pd.DataFrame(Imputer.fit_transform(X_train))
Imputed_X_val = pd.DataFrame(Imputer.transform(X_val))

#Re-add Columns

Imputed_X_Train.columns = X_train.columns
Imputed_X_val.columns = X_val.columns
print("Normal Model:")
scoreDataset(X_train, X_val,y_train, y_val)
print("Imputed model")
scoreDataset(Imputed_X_Train, Imputed_X_val ,y_train, y_val)






