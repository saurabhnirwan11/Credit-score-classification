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


# Supervised-Classifier-metrics
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    log_loss,
)
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

# Supervised-cross_validate-GridSearchCV
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("Data/cleaned_train.csv")
for i in df.columns:
    if i in ["Customer_ID", "Month"]:
        df.drop(columns=[i], inplace=True)

X = df.drop(columns=["Credit_Score"])
y = df["Credit_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

dummy = GetDummies()
X_train_dummy = dummy.fit_transform(X_train)
X_test_dummy = dummy.transform(X_test)
X_train_dummy = X_train_dummy.reset_index(drop=True)
X_test_dummy = X_test_dummy.reset_index(drop=True)


cat = X_train_dummy.select_dtypes(include="object").columns.tolist()
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = pd.DataFrame(
    ohe.fit_transform(X_train[cat]), columns=ohe.get_feature_names_out()
)
X_test_cat = pd.DataFrame(
    ohe.transform(X_test[cat]), columns=ohe.get_feature_names_out()
)
X_train_ohe = X_train_cat.join(X_train_dummy.select_dtypes("number"))
X_test_ohe = X_test_cat.join(X_test_dummy.select_dtypes("number"))

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_ohe)
X_test_scaled = sc.transform(X_test_ohe)

grid_model_rfc = RandomForestClassifier(
    class_weight="balanced",
    criterion="gini",
    max_depth=2,
    min_impurity_decrease=0,
    n_estimators=100,
    oob_score=True,
    random_state=1337,
)
grid_model_rfc.fit(X_train_scaled, y_train)
y_pred = grid_model_rfc.predict(X_test_scaled)
report = classification_report(y_test, y_pred, output_dict=True)
report = pd.DataFrame(report).transpose()
report.to_csv("Result/training_report.csv", index=False)

pickle.dump(grid_model_rfc, open("models/grid_model_rfc.pkl", "wb"))
pickle.dump(ohe, open("models/credit_score_multi_class_ohe_encoder.pkl", "wb"))
pickle.dump(le, open("models/credit_score_multi_class_le.pkl", "wb"))
pickle.dump(sc, open("models/credit_score_multi_class_sc.pkl", "wb"))
pickle.dump(dummy, open("models/credit_score_multi_class_get_dummy.pkl", "wb"))
