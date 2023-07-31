import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from custom_encoder import GetDummies # required to load dummy


app = Flask(__name__)
## Load the model
grid_model_rfc = pickle.load(open("models/grid_model_rfc.pkl", "rb"))
sc = pickle.load(open("models/credit_score_multi_class_sc.pkl", "rb"))
ohe = pickle.load(open("models/credit_score_multi_class_ohe_encoder.pkl", "rb"))
le = pickle.load(open("models/credit_score_multi_class_le.pkl", "rb"))
dummy = pickle.load(open("models/credit_score_multi_class_get_dummy.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]

    # prediction on single data point
    X_test = df.copy()
    X_test_dummy = dummy.transform(X_test)
    cat = X_test_dummy.select_dtypes(include="object").columns.tolist()

    X_test_cat = pd.DataFrame(
        ohe.transform(X_test[cat]), columns=ohe.get_feature_names_out()
    )
    X_test_ohe = X_test_cat.join(X_test_dummy.select_dtypes("number"))

    X_test_scaled = sc.transform(X_test_ohe)

    y_pred = grid_model_rfc.predict(X_test_scaled)

    y_pred = le.inverse_transform(y_pred)
    output = y_pred[0]

    return render_template(
        "home.html", prediction_text="The House price prediction is {}".format(output)
    )


if __name__ == "__main__":
    app.run(debug=True)
