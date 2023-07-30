import pickle
from flask import Flask, request, app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
rfc_model = pickle.load(open("models/grid_model_rfc.pkl","rb"))
scalar = pickle.load(open("models/credit_score_multi_class_sc.pkl","rb"))
ohe = pickle.load(open("models/credit_score_multi_class_ohe_encoder.pkl","rb"))
le = pickle.load(open("models/credit_score_multi_class_le.pkl","rb"))
dummy = pickle.load(open("models/credit_score_multi_class_get_dummy.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=rfc_model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)