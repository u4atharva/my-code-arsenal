__author__ = "compiler"

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

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
# df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
df = df.drop(columns =["reservation_status_date"])

#Converting Month names to numbers
df['arrival_date_month'] = df['arrival_date_month'].str[:3]
df.loc[df["arrival_date_month"] == "Jan", 'arrival_date_month'] = 1
df.loc[df["arrival_date_month"] == "Feb", 'arrival_date_month'] = 2
df.loc[df["arrival_date_month"] == "Mar", 'arrival_date_month'] = 3
df.loc[df["arrival_date_month"] == "Apr", 'arrival_date_month'] = 4
df.loc[df["arrival_date_month"] == "May", 'arrival_date_month'] = 5
df.loc[df["arrival_date_month"] == "Jun", 'arrival_date_month'] = 6
df.loc[df["arrival_date_month"] == "Jul", 'arrival_date_month'] = 7
df.loc[df["arrival_date_month"] == "Aug", 'arrival_date_month'] = 8
df.loc[df["arrival_date_month"] == "Sep", 'arrival_date_month'] = 9
df.loc[df["arrival_date_month"] == "Oct", 'arrival_date_month'] = 10
df.loc[df["arrival_date_month"] == "Nov", 'arrival_date_month'] = 11
df.loc[df["arrival_date_month"] == "Dec", 'arrival_date_month'] = 12

# Simplification //ToDo
df =  df.drop(columns =["agent", "company"])

# One Hot Encoding for columns: ['meal', 'country', 'market_segment', 'distribution_channel', 'customer_type']
df = pd.get_dummies(df, prefix=['meal', 'country', 'market_segment', 'distribution_channel', 'customer_type', 'hotel'], columns=['meal', 'country', 'market_segment', 'distribution_channel', 'customer_type', 'hotel'])

# Handling Room Choice
df['room_choice_granted'] = df.apply(compute_room_choice, axis=1)
df = df.drop(columns =["reserved_room_type", "assigned_room_type"])



# Sklearn Test-Train Split
RSEED = 50
labels = np.array(df.pop('is_canceled'))
X_train, X_test, y_train, y_test = train_test_split(df,
                                                    labels,
                                                    stratify=labels,
                                                    test_size=0.3,
                                                    random_state=RSEED)  # 70% training and 30% test

#  X_train, X_test, y_train, y_test = train_test_split(df, df["is_canceled"], test_size=0.3, random_state=109) # 70% training and 30% test

features = list(X_train.columns)

# SUPPORT VECTOR MACHINE

# Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Accuracy evaluation
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))



# RANDOM FOREST:
model = RandomForestClassifier(n_estimators=100,
                               random_state=RSEED,
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Accuracy evaluation
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))


# Extract feature importance
fi = pd.DataFrame({'feature': list(X_train.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending=False)

# Display
fi.head()
