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

@app.route('/')
def home():
    return render_template('home.html')