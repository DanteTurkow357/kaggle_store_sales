import pandas as pd
import numpy as np
import datetime as dt

train = pd.read_csv("Data/train.csv")
df = train
df['new_years_day'] = np.where((pd.to_datetime(df['date']).dt.month == 1) & ((pd.to_datetime(df['date'])).dt.day == 1), True, False)

# Remove stores that opened after first day in data
store_20_missing_idx = df.loc[(df["store_nbr"] == 20) &  (df["date"] < '2015-02-12')].index
df = df.drop(store_20_missing_idx)
store_21_missing_idx = df.loc[(df["store_nbr"] == 21) & (df["date"] < '2015-07-26')].index
df = df.drop(store_21_missing_idx)
store_22_missing_idx = df.loc[(df["store_nbr"] == 22) & (df["date"] < '2015-10-12')].index
df = df.drop(store_22_missing_idx)
store_29_missing_idx = df.loc[(df["store_nbr"] == 29) & (df["date"] < '2015-03-19')].index
df = df.drop(store_29_missing_idx)
store_36_missing_idx = df.loc[(df["store_nbr"] == 36) & (df["date"] < '2013-05-08')].index
df = df.drop(store_36_missing_idx)
store_42_missing_idx = df.loc[(df["store_nbr"] == 42) & (df["date"] < '2015-08-20')].index
df = df.drop(store_42_missing_idx)
store_52_missing_idx = df.loc[(df["store_nbr"] == 52) & (df["date"] < '2017-04-19')].index
df = df.drop(store_52_missing_idx)
store_53_missing_idx = df.loc[(df["store_nbr"] == 53) & (df["date"] < '2014-05-28')].index
df = df.drop(store_53_missing_idx)


# Remove the random zero gaps for stores 
def remove_missing_val_by_store(df, store_nbr):
    # FINDS THE MISSING VALUES THAT ARE NOT NEW YEARS
    store_missing_idx = df.loc[(df["store_nbr"] == store_nbr) & (df["new_years_day"] == False) & (df["sales"] == 0)].index
    df = df.drop(store_missing_idx)
    return df

df = remove_missing_val_by_store(df, 12)
df = remove_missing_val_by_store(df, 14)
df = remove_missing_val_by_store(df, 18)
df = remove_missing_val_by_store(df, 24)
df = remove_missing_val_by_store(df, 25)
df = remove_missing_val_by_store(df, 30)

# Remove the missing values by Family

baby_care_missing_idx = df.loc[(df["family"] == "BABY CARE") & (df["date"] < '2015-12-01')].index
df = df.drop(baby_care_missing_idx)

beverage_missing_idx = df.loc[(df["family"] == "BEVERAGES") & (df["date"] < "2015-05-31")].index
df = df.drop(beverage_missing_idx)

books_missing_idx = df.loc[(df["family"] == "BOOKS") & (df["date"] < "2016-10-08")].index
df = df.drop(books_missing_idx)

celebration = df.loc[(df["family"] == "CELEBRATION")]
celebration = celebration.loc[(pd.to_datetime(celebration["date"]).dt.year < 2016)]
celebration_missing = celebration.groupby('date')['sales'].sum().reset_index()
celebration_missing_dates = max(celebration_missing.loc[celebration_missing['sales'] == 0]["date"])
celebration_missing_idx = celebration.loc[celebration["date"] < '2015-05-31'].index
df = df.drop(celebration_missing_idx)

home_kitch = df.loc[df["family"].isin(["HOME AND KITCHEN I","HOME AND KITCHEN II"])]
home_kitch = home_kitch.loc[home_kitch['date'] < '2015-01-01']
home_kitch_missing = home_kitch.groupby('date')['sales'].sum().reset_index()
home_kitch_missing_dates = list(home_kitch_missing.loc[home_kitch_missing['sales'] == 0]["date"])
home_kitch_missing_idx = home_kitch.loc[home_kitch['date'].isin(home_kitch_missing_dates)].index
df = df.drop(home_kitch_missing_idx)

players_missing_idx = df.loc[(df["family"] == "PLAYERS AND ELECTRONICS") & (df["new_years_day"] == False) & (df["sales"] == 0)].index
df = df.drop(players_missing_idx)

home_care = df.loc[(df["family"] == "HOME CARE")]
home_care_missing = home_care.loc[(pd.to_datetime(home_care["date"]).dt.year <= 2015)]
home_care_missing = home_care_missing.groupby('date')['sales'].sum().reset_index()
home_care_missing_dates = list(home_care_missing.loc[home_care_missing['sales'] == 0]["date"])
home_care_missing_idx = home_care.loc[home_care["date"].isin(home_care_missing_dates)].index
home_care = home_care.drop(home_care_missing_idx)
df = df.drop(home_care_missing_idx)

ladies_wear = df.loc[(df["family"] == "LADIESWEAR")]
ladies_wear_missing = ladies_wear.loc[(pd.to_datetime(ladies_wear["date"]).dt.year <= 2015)]
ladies_wear_missing = ladies_wear_missing.groupby('date')['sales'].sum().reset_index()
ladies_wear_missing_dates = list(ladies_wear_missing.loc[ladies_wear_missing['sales'] == 0]["date"])
ladies_wear_missing_idx = ladies_wear.loc[ladies_wear["date"].isin(ladies_wear_missing_dates)].index
ladies_wear = ladies_wear.drop(ladies_wear_missing_idx)
df = df.drop(ladies_wear_missing_idx)

magazines_idx = df.loc[(df["family"] == "MAGAZINES") & (df["date"] <= '2015-10-01')].index
df = df.drop(magazines_idx)

meats = df.loc[(df["family"] == "MEATS")]
meats_outlier = meats.loc[meats['sales'] == max(meats['sales'])].index
df = df.drop(meats_outlier)

pet_supplies = df.loc[(df["family"] == "PET SUPPLIES")]
pet_supplies_missing = pet_supplies.loc[(pd.to_datetime(pet_supplies["date"]).dt.year <= 2015)]
pet_supplies_missing = pet_supplies_missing.groupby('date')['sales'].sum().reset_index()

pet_supplies_missing_dates = list(pet_supplies_missing.loc[pet_supplies_missing['sales'] == 0]["date"])
pet_supplies_missing_idx = pet_supplies.loc[pet_supplies["date"].isin(pet_supplies_missing_dates)].index
pet_supplies = pet_supplies.drop(pet_supplies_missing_idx)
df = df.drop(pet_supplies_missing_idx)

produce = df.loc[(df["family"] == "PRODUCE")]
produce = produce.loc[(pd.to_datetime(produce["date"]).dt.year < 2016)]
produce_missing = produce.groupby('date')['sales'].sum().reset_index()
produce_missing_dates = list(produce_missing.loc[produce_missing['sales'] < 2000]["date"])
produce_missing_idx = produce.loc[produce["date"].isin(produce_missing_dates)].index
df = df.drop(produce_missing_idx)