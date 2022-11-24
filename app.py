from flask import Flask, jsonify, render_template, request
import joblib
import os
import pickle
import joblib
import numpy as np



app = Flask(__name__)
xg = joblib.load(open('xg.sav','rb'))

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def result():

        Item_Weight=float(request.form['Item_Weight'])
        Item_Fat_Content=float(request.form['Item_Fat_Content'])
        Item_Type=float(request.form['Item_Type'])
        Item_MRP=float(request.form['Item_MRP'])
        Outlet_Establishment_Year = float(request.form['Outlet_Establishment_Year'])
        Outlet_Size=float(request.form['Outlet_Size'])
        Outlet_Type=float(request.form['Outlet_Type'])
        data=np.array([[Item_Weight,Item_Fat_Content,Item_Type,Item_MRP,
              Outlet_Establishment_Year,Outlet_Size,Outlet_Type]])
        
        prediction=xg.predict(data)
        return render_template('home.html', prediction_text= float(prediction))

if __name__ == "__main__":
    app.run(debug=True, port=9457)
