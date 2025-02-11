import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit


# Read Data
transactions = pd.read_csv("Data/transactions.csv")
test = pd.read_csv("Data/test.csv")
stores = pd.read_csv("Data/stores.csv")
holidays = pd.read_csv("Data/holidays_events.csv")
print("data has been loaded")
test = test.groupby(['date', 'store_nbr']).sum().reset_index()[["date","store_nbr"]]

def merge_and_format(df, stores, holidays):
    # Format as datetime
    df['date'] = pd.to_datetime(df['date'])
    # Merge with the stores data
    df_merged = pd.merge(df, stores, on='store_nbr')

    # Prep the holidays data
    holidays_column_names = ['date', 'holiday_type', 'locale', 'locale_name', 'holiday_name', 'transferred']
    holidays.columns = holidays_column_names
    # select the holidays that are important
    important_holidays = [
        'Traslado Batalla de Pichincha', 'Viernes Santo', 'Carnaval',"Navidad+1", "Navidad-1", 
        "Navidad-2", "Fundacion de Quito", "Traslado Primer dia del ano", "Puente Primer dia del ano", "Primer dia del ano",
        "Primer dia del ano-1", "Dia de Difuntos", "Traslado Primer Grito de Independencia", "Puente Dia de Difuntos", 
        "Independencia de Cuenca"]
    # Filter the data
    holidays = holidays[holidays['holiday_name'].isin(important_holidays)]
    holidays = holidays[['date', 'holiday_name']]
    holidays['date'] = pd.to_datetime(holidays['date'])
    # display(holidays.head())
    # Merge with the holidays
    df_merged = df_merged.merge(holidays, on='date', how="left")
    df_merged['holiday_name'] = df_merged['holiday_name'].fillna('no_holiday')
    return df_merged

def create_features(df, train=True): 
    # display(df.head())
    # Time Features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter
    df['day_cat'] = df['date'].dt.day_name()
    df["month_cat"] = df["date"].dt.month_name()
    
    # Categorical Features
    df["store_nbr"] = df["store_nbr"].astype(str)
    df['cluster'] = df['cluster'].astype(str)
    df['is_holiday'] = np.where(df['holiday_name'] == 'no_holiday', "not_holiday", "holiday")

    # Store-Holiday Interactions
    df['store_holiday_interaction'] = df['store_nbr'] + "_" + df['is_holiday']
    df['store_day_interaction'] = df['store_nbr'] + "_" + df['day_cat']
    df['store_month_interaction'] = df['store_nbr'] + "_" + df['month_cat']
    df['type_day_interaction'] = df['type'] + "_" + df['day_cat']
    df['holiday_day_interaction'] = df['holiday_name'] + "_" + df['day_cat']
    df["dec"] = np.where(df["month_cat"] == "December", 'dec', 'not_dec')
    df['dec_day_interaction'] = df['day_cat'] + "_" + df['dec']
    # If the test data is passed through here, we need to include some of the training data to create the lags
    if train == False:
        transactions_filter = transactions.loc[transactions['date'] >= '2017-07-17']
        df = pd.concat([transactions_filter, df],axis =0)
        df = df.reset_index(drop = True)
        df["store_nbr"] = df["store_nbr"].astype(str)
        df = df.sort_values(['store_nbr', 'date'])
    else:
    # Rolling features for the training data
        window_size = 7
        df['transactions_rolling_avg'] = df.groupby('store_nbr')['transactions'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    # Lagged features 
    df["trans_lag"] = df.groupby('store_nbr')['transactions'].shift(1)
    df["trans_lag_2"] = df.groupby('store_nbr')['transactions'].shift(2)
    df["trans_lag_3"] = df.groupby('store_nbr')['transactions'].shift(3)
    df["trans_lag_7"] = df.groupby('store_nbr')['transactions'].shift(7)
    df["trans_lag_14"] = df.groupby('store_nbr')['transactions'].shift(14)
    df["trans_lag_30"] = df.groupby('store_nbr')['transactions'].shift(30)
    df = df.dropna(subset=['trans_lag_30'])  # Remove rows with missing lags
    return df

# Identify feature types
num_features = ["trans_lag", "trans_lag_2", "trans_lag_3", "trans_lag_7","trans_lag_14", "trans_lag_30"]
cat_features = ["store_day_interaction", "store_holiday_interaction", "type_day_interaction", "holiday_name", "dec_day_interaction"]

# Define transformations
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformations
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

ridge_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])

# Prepare training data
train = merge_data(transactions, stores, holidays)
train = create_features(train, train=True)

# Prepare test data
test = merge_data(test, stores, holidays)
test = create_features(test, train=False)  # No transactions column

X_train = train[num_features + cat_features]
y_train = train["transactions"]

# Train the model
ridge_model.fit(X_train, y_train)

# # Predict iteratively for test set
# test_dates = sorted(test["date"].unique())  # Ensure chronological order
# predictions = {}

# for date in test_dates:
#     daily_test = test[test["date"] == date].copy()
    
#     # Update lag features based on previous predictions
#     for store in daily_test["store_nbr"].unique():
#         if store in predictions:
#             prev_day_pred = predictions[store][-1]  # Get last prediction
#             daily_test.loc[daily_test["store_nbr"] == store, "trans_lag"] = prev_day_pred
            
#             # Get two-day lag
#             if len(predictions[store]) > 1:
#                 daily_test.loc[daily_test["store_nbr"] == store, "trans_lag_2"] = predictions[store][-2]
#         else:
#             # Default to mean transaction value if no history
#             default_val = train[train["store_nbr"] == store]["transactions"].mean()
#             daily_test.loc[daily_test["store_nbr"] == store, "trans_lag"] = default_val
#             daily_test.loc[daily_test["store_nbr"] == store, "trans_lag_2"] = default_val
    
#     # Select features
#     X_daily = daily_test[num_features + cat_features]
    
#     # Predict transactions
#     daily_test["transactions_pred"] = ridge_model.predict(X_daily)
    
#     # Store predictions for future iterations
#     for store, pred in zip(daily_test["store_nbr"], daily_test["transactions_pred"]):
#         if store not in predictions:
#             predictions[store] = []
#         predictions[store].append(pred)

#     # Save results back to the test DataFrame
#     test.loc[test["date"] == date, "transactions_pred"] = daily_test["transactions_pred"].values
