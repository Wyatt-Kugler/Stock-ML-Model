import pandas as pd

def encode_new_row(new_row: pd.Series, encoder, object_columns: list, features: list) -> pd.DataFrame:
    numeric_part = new_row.drop(labels=object_columns)
    if len(object_columns) == 0 or encoder is None:
        # No categorical columns to encode, just return numeric part as dataframe with features order
        df = pd.DataFrame([numeric_part.values], columns=numeric_part.index)
        df = df.reindex(columns=features, fill_value=0)
        return df

    object_part = new_row[object_columns].values.reshape(1, -1)
    encoded_part = encoder.transform(object_part)
    encoded_df = pd.DataFrame(encoded_part, columns=encoder.get_feature_names_out(object_columns))
    final_df = pd.concat([pd.DataFrame([numeric_part.values], columns=numeric_part.index), encoded_df], axis=1)
    final_df = final_df.reindex(columns=features, fill_value=0)
    return final_df
def findFeatureImportance(stockModel, features):
    # After training
    importances = stockModel.feature_importances_
    feature_importance = pd.Series(importances, index=features)
    feature_importance = feature_importance.sort_values(ascending=False)
    print("Feature Importances:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")
