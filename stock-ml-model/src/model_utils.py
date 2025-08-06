import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBRegressor, XGBClassifier
from features import encode_new_row


def find_feature_importance(model, features):
    importances = model.feature_importances_
    series = pd.Series(importances, index=features).sort_values(ascending=False)
    print("Feature Importances:")
    for feat, val in series.items():
        print(f"{feat}: {val:.4f}")

def find_correlation(df, features, target):
    subset = df[features + [target]].dropna()
    for feat in features:
        corr = subset[feat].corr(subset[target])
        print(f"Correlation between {feat} and {target}: {corr:.2f}")

def scoreDataset(X_train, X_val, y_train, y_val, task, stockData, features, OHEncoder):
    if task == "regression":
    
        XGBmodel = XGBRegressor(
            random_state=1,
            max_depth=8,
            gamma=5,
            n_estimators=100,
            colsample_bytree=0.7,
            learning_rate=0.1,
            subsample=0.6,
            reg_lambda=2,
            reg_alpha=1,
            min_child_weight=5,
)
#         param_dist = {
#              'n_estimators': [100, 200, 300, 400],
#              'learning_rate': [0.01, 0.05, 0.1, 0.2],
#              'max_depth': [3, 4, 5, 6, 8],
#              'subsample': [0.6, 0.7, 0.8, 1.0],
#              'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
#             'gamma': [0, 1, 5],
#              'min_child_weight': [1, 3, 5, 7],
#              'reg_alpha': [0, 0.1, 1],
#              'reg_lambda': [1, 1.5, 2],
#  }
#         randomSearch = RandomizedSearchCV(
#              XGBmodel,
#              param_distributions=param_dist,
#              n_iter=30,
#              scoring="neg_mean_absolute_error",
#              cv=3,
#              verbose=2,
#              n_jobs=-1)
#         randomSearch.fit(X_train, y_train)
#         print("Best parameters found: ", randomSearch.best_params_)
#         XGBmodel = randomSearch.best_estimator_

        print("Training XGBoost Model...")
        XGBmodel.fit(X_train, y_train)
        stockPredictions = XGBmodel.predict(X_val)
        stockMae = mean_absolute_error(y_val, stockPredictions)
        maeAverage = 100 * (stockMae/y_val.mean())
        print("Validation MAE for XGBoost Model: {:,.2f}".format(maeAverage), "%")
        latest_data = stockData.iloc[-1]
        object_columns = X_train.select_dtypes(include="object").columns.tolist()
        features = X_train.columns.tolist() 
        
        tomorrow_features = encode_new_row(latest_data, OHEncoder, object_columns, features)
        tomorrow_features = tomorrow_features.astype(float) 

        # Predict tomorrow's closing price
        tomorrow_pred = XGBmodel.predict(tomorrow_features)
        print(f"Predicted closing price for tomorrow: ${tomorrow_pred[0]:.2f}")
        find_feature_importance(XGBmodel, features)
        df_for_corr = X_train.copy()
        df_for_corr["Target"] = y_train
        find_correlation(df_for_corr, features, "Target")

    elif task == "classification":
# Random Search for hyperparameter tuning
        XGBmodel =  XGBClassifier(
            subsample=0.6,
            reg_lambda=1,
            reg_alpha=0.1,
            n_estimators=200,
            min_child_weight=3,
            max_depth=8,
            learning_rate=0.2,
            gamma=1,
            colsample_bytree=0.6,
            eval_metric="logloss",
            random_state=1
        )
#         param_dist = {
#             'n_estimators': [100, 200, 300, 400],
#             'learning_rate': [0.01, 0.05, 0.1, 0.2],
#             'max_depth': [3, 4, 5, 6, 8],
#             'subsample': [0.6, 0.7, 0.8, 1.0],
#             'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
#             'gamma': [0, 1, 5],
#             'min_child_weight': [1, 3, 5, 7],
#             'reg_alpha': [0, 0.1, 1],
#             'reg_lambda': [1, 1.5, 2],
# }
#         randomSearch = RandomizedSearchCV(
#             XGBmodel,
#             param_distributions=param_dist,
#             n_iter=30,
#             scoring='accuracy',
#             cv=3,
#             verbose=2,
#             n_jobs=-1)
#         randomSearch.fit(X_train, y_train)
#         print("Best parameters found: ", randomSearch.best_params_)
#         XGBmodel = randomSearch.best_estimator_
        
        XGBmodel.fit(X_train,y_train)
        stockPredictions = XGBmodel.predict(X_val)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        print("Accuracy: {:.2f}".format(accuracy_score(y_val, stockPredictions)))
        print("F1 Score: {:.2f}".format(f1_score(y_val, stockPredictions)))
        print("Precision: {:.2f}".format(precision_score(y_val, stockPredictions)))
        print("Recall: {:.2f}".format(recall_score(y_val, stockPredictions)))

        object_columns = stockData.select_dtypes(include="object").columns.tolist()
        features = X_train.columns.tolist()  
        encoder = OHEncoder  

        latest_data = stockData.iloc[-1]

        tomorrow_features = encode_new_row(latest_data, encoder, object_columns, features)
        tomorrow_features = tomorrow_features.astype(float) 


        tomorrow_pred = XGBmodel.predict(tomorrow_features)
        if tomorrow_pred[0] == 1:
            print("Predicted stock price will increase tomorrow.")
        else:
            print("Predicted stock price will decrease tomorrow.")
        find_feature_importance(XGBmodel, features)
        df_for_corr = X_train.copy()
        df_for_corr["PriceIncrease"] = y_train
        find_correlation(df_for_corr, features, "PriceIncrease")
