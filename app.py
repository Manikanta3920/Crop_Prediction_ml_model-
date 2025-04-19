import numpy as np 
from flask import request,render_template,jsonify,Flask
import pickle


app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["post"])
def predict():
    float_feature=[float(x) for x in request.form.values()]
    features=[np.array(float_feature)]
    prediction=model.predict(features)
    return render_template("index.html", Predicted_text=f"The Predicted Crop is {prediction[0]}")

if __name__=="__main__":
    app.run(debug=True)