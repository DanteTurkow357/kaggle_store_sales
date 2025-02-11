import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import datetime as dt

test = pd.read_csv("Data/test.csv")
stores = pd.read_csv("Data/stores.csv")
train = pd.read_csv("Data/train.csv")
transactions = pd.read_csv("Data/transactions.csv")

df = train.merge(stores, on = "store_nbr")
# df['month_cat'] = pd.to_datetime(df['date']).dt.month_name()
# df['day_cat'] = pd.to_datetime(df['date']).dt.day_name()
# df['new_years_day'] = np.where((pd.to_datetime(df['date']).dt.month == 1) & ((pd.to_datetime(df['date'])).dt.day == 1), True, False)
print(df.head())
print(df.shape)

# PRODUCE
produce = df.loc[(df["family"] == "PRODUCE")]
produce = produce.loc[(pd.to_datetime(produce["date"]).dt.year < 2016)]
produce_missing = produce.groupby('date')['sales'].sum().reset_index()
produce_missing_dates = list(produce_missing.loc[produce_missing['sales'] < 2000]["date"])
produce_missing_idx = produce.loc[produce["date"].isin(produce_missing_dates)].index
df = df.drop(produce_missing_idx)

# CELBRATION
celebration = df.loc[(df["family"] == "CELEBRATION")]
celebration = celebration.loc[(pd.to_datetime(celebration["date"]).dt.year < 2016)]
celebration_missing = celebration.groupby('date')['sales'].sum().reset_index()
celebration_missing_dates = max(celebration_missing.loc[celebration_missing['sales'] == 0]["date"])
celebration_missing_idx = celebration.loc[celebration["date"] < '2015-05-31'].index
df = df.drop(celebration_missing_idx)

# BABY CARE
baby_care_missing_idx = df.loc[(df["family"] == "BABY CARE") & (df["date"] < '2015-12-01')].index
df = df.drop(baby_care_missing_idx)

# PET SUPPLIES
pet_supplies = df.loc[(df["family"] == "PET SUPPLIES")]
pet_supplies_missing = pet_supplies.loc[(pd.to_datetime(pet_supplies["date"]).dt.year <= 2015)]
pet_supplies_missing = pet_supplies_missing.groupby('date')['sales'].sum().reset_index()
pet_supplies_missing_dates = list(pet_supplies_missing.loc[pet_supplies_missing['sales'] == 0]["date"])
pet_supplies_missing_idx = pet_supplies.loc[pet_supplies["date"].isin(pet_supplies_missing_dates)].index
pet_supplies = pet_supplies.drop(pet_supplies_missing_idx)
df = df.drop(pet_supplies_missing_idx)

# HOME CARE
home_care = df.loc[(df["family"] == "HOME CARE")]
home_care_missing = home_care.loc[(pd.to_datetime(home_care["date"]).dt.year <= 2015)]
home_care_missing = home_care_missing.groupby('date')['sales'].sum().reset_index()
home_care_missing_dates = list(home_care_missing.loc[home_care_missing['sales'] == 0]["date"])
home_care_missing_idx = home_care.loc[home_care["date"].isin(home_care_missing_dates)].index
home_care = home_care.drop(home_care_missing_idx)
df = df.drop(home_care_missing_idx)

# LADIES WEAR
ladies_wear = df.loc[(df["family"] == "LADIESWEAR")]
ladies_wear_missing = ladies_wear.loc[(pd.to_datetime(ladies_wear["date"]).dt.year <= 2015)]
ladies_wear_missing = ladies_wear_missing.groupby('date')['sales'].sum().reset_index()
ladies_wear_missing_dates = list(ladies_wear_missing.loc[ladies_wear_missing['sales'] == 0]["date"])
ladies_wear_missing_idx = ladies_wear.loc[ladies_wear["date"].isin(ladies_wear_missing_dates)].index
ladies_wear = ladies_wear.drop(ladies_wear_missing_idx)
df = df.drop(ladies_wear_missing_idx)

# MAGEZINES
magazines_idx = df.loc[(df["family"] == "MAGAZINES") & (df["date"] <= '2015-10-01')].index
df = df.drop(magazines_idx)

# MEATS
meats = df.loc[(df["family"] == "MEATS")]
meats_outlier = meats.loc[meats['sales'] == max(meats['sales'])].index
df = df.drop(meats_outlier)

# HOME AND KITCHEN 
home_kitch = df.loc[df["family"].isin(["HOME AND KITCHEN I","HOME AND KITCHEN II"])]
home_kitch = home_kitch.loc[home_kitch['date'] < '2015-01-01']
home_kitch_missing = home_kitch.groupby('date')['sales'].sum().reset_index()
home_kitch_missing_dates = list(home_kitch_missing.loc[home_kitch_missing['sales'] == 0]["date"])
home_kitch_missing_idx = home_kitch.loc[home_kitch['date'].isin(home_kitch_missing_dates)].index
df = df.drop(home_kitch_missing_idx)
print("size before cleaning", train.shape[0])
print("size after cleaning", df.shape[0])
print("rows removed from the data set", train.shape[0] - df.shape[0])
print(df.head())

df.to_csv("Data/train_clean.csv")