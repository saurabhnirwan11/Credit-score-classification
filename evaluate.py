import numpy as np
import pandas as pd
import scipy.stats as stats

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from custom_encoder import GetDummies
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


## Load the model
grid_model_rfc = pickle.load(open("models/grid_model_rfc.pkl", "rb"))
sc = pickle.load(open("models/credit_score_multi_class_sc.pkl", "rb"))
ohe = pickle.load(open("models/credit_score_multi_class_ohe_encoder.pkl", "rb"))
le = pickle.load(open("models/credit_score_multi_class_le.pkl", "rb"))
dummy = pickle.load(open("models/credit_score_multi_class_get_dummy.pkl", "rb"))

df = pd.read_csv("Data/cleaned_test.csv")
for i in df.columns:
    if i in ["Customer_ID", "Month"]:
        df.drop(columns=[i], inplace=True)
X_test = df.copy()

X_test_dummy = dummy.transform(X_test)
X_test_dummy = X_test_dummy.reset_index(drop=True)
cat = X_test_dummy.select_dtypes(include="object").columns.tolist()

X_test_cat = pd.DataFrame(
    ohe.transform(X_test[cat]), columns=ohe.get_feature_names_out()
)
X_test_ohe = X_test_cat.join(X_test_dummy.select_dtypes("number"))

X_test_scaled = sc.transform(X_test_ohe)

y_pred = grid_model_rfc.predict(X_test_scaled)

y_pred = le.inverse_transform(y_pred)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ["Prediction_Credit_Score"]

test_df = pd.read_csv("Data/cleaned_test.csv")
final_df = pd.concat([test_df, y_pred], axis=1)
final_df.to_csv("Result/prediction_on_test_data.csv", index=False)
