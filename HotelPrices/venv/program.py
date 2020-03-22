__author__ = "compiler"

import pandas as pd
import matplotlib as plt

def compute_room_choice(row):
    if row['reserved_room_type'] == row['assigned_room_type']:
        val = 1
    else:
        val = 0
    return val

df = pd.read_csv("hotel_bookings.csv")

# converting text values to 1, 0 and -1
df.loc[df["deposit_type"] == "No Deposit", 'deposit_type'] = -1
df.loc[df["deposit_type"] == "Non Refund", 'deposit_type'] = 0
df.loc[df["deposit_type"] == "Refundable", 'deposit_type'] = 1

df.loc[df["reservation_status"] == "Check-Out", 'reservation_status'] = 1
df.loc[df["reservation_status"] == "Canceled", 'reservation_status'] = -1
df.loc[df["reservation_status"] == "No-Show", 'reservation_status'] = 0

# Handling missing values
df.loc[df["market_segment"] == "Undefined", 'market_segment'] = "Online TA"     # Since Online TA has the largest count and only 2 were missing
df.loc[df["distribution_channel"] == "Undefined", 'distribution_channel'] = "TA/TO"     # Since  TA/TO has the largest count and only 5 were missing

# Handling date time formats
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

# Simplification //ToDo
df =  df.drop(columns =["agent", "company"])

# One Hot Encoding for columns: ['meal', 'country', 'market_segment', 'distribution_channel', 'customer_type']
df = pd.get_dummies(df, prefix=['meal', 'country', 'market_segment', 'distribution_channel', 'customer_type'], columns=['meal', 'country', 'market_segment', 'distribution_channel', 'customer_type'])

# Handling Room Choice
df['room_choice_granted'] = df.apply(compute_room_choice, axis=1)
df =  df.drop(columns =["reserved_room_type", "assigned_room_type"])


# print(df["reservation_status_date"].value_counts())

# for col in df.columns:
#     print(col)
# print(df_input.columns)