import flask 
from flask import Flask,request,redirect,app,render_template
import pickle
import numpy as np
import pandas as pd

application=Flask(__name__)
app=application

scaler=pickle.load(open(r'C:\Users\Developer\Desktop\end-to-end-classification-project\model\standardscaler.pkl','rb'))
model=pickle.load(open(r'C:\Users\Developer\Desktop\end-to-end-classification-project\model\correctmodel.pkl','rb'))


@app.route('/')

def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict():
    result=""

    if request.method=='POST':
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = int(request.form.get('Age'))
        data=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
        new_data=scaler.transform([data])

       
        prediction=model.predict(new_data)
        print(prediction[0])
        if prediction==1:
            result='Diabetic'
        else:
            result='None-Diabetic'
        return render_template('home.html',result=result)

    else:
        return render_template('home.html')


if __name__=='__main__':
    app.run(debug=True)



