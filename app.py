from typing import MutableMapping
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import pandas as pd
import numpy as np
import sklearn
import matplotlib
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('Predict.pkl', 'rb'))
@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

standard_to = StandardScaler()
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        ServiceSpan = float(request.form['ServiceSpan'])
        QuarterlyPayment = float(request.form['QuarterlyPayment'])
        GrandPayment = float(request.form['GrandPayment'])

        TotalDependents = request.form['TotalDependents']
        if(TotalDependents == 'Yes'):
            Totaldependents_Yes = 1
        
        else:
            Totaldependents_No = 0
        
        Aged = request.form['Aged']
        if(Aged == 'Yes'):
            Yes = 1
            
        elif(Aged == 'No'):
            Aged = 1
        else:
            No = 0
            
        
        sex = request.form['sex']
        if(sex == 'Male'):
            Male = 1
        
        else:
            Male = 0

        MobileService = request.form['MobileService']
        if(MobileService == 'Yes'):
            Mobile_Service_Yes = 1
            
        else:
            Mobile_Service_Yes = 0
            
        Married = request.form['Married']
        if(Married == 'Yes'):
            Married_Status_Yes = 1
         
        else:
            Married_Status_Yes = 0
            

        CyberProtection = request.form['CyberProtection']
        if(CyberProtection == 'Yes'):
            CyberProtection_Yes = 1
            
        else:
            CyberProtection_Yes = 0
            

        HardwareSupport = request.form['HardwareSupport']
        if(HardwareSupport == 'Yes'):
            Hardware_Support_Yes = 1
            
        else:
            Hardware_Support_Yes = 0
            

        TechnicalAssistance = request.form['TechnicalAssistance']
        if(TechnicalAssistance == 'Yes'):
            Technical_Assistance_Yes = 1
            
        else:
            Technical_Assistance_Yes = 0
            

        FilmSubscription = request.form['FilmSubscription']
        if(FilmSubscription == 'Yes'):
            Film_Subscription_Yes = 1
            
        else:
            Film_Subscription_Yes = 0
            
        
        
        GService = request.form['GService']
        if(GService == 'GService_No'):
            GService_No = 1
            GService_Satellite_Broadband= 0
            GService_Wifi_Broadband = 0
                
        elif(GService == 'GService_Satellite_Broadband'):
            GService_No = 0
            GService_Satellite_Broadband= 1
            GService_Wifi_Broadband = 0
        
        else:
            GService_No = 0
            GService_Satellite_Broadband= 0
            GService_Wifi_Broadband = 1

        SettlementProcess = request.form['SettlementProcess']
        if(SettlementProcess == 'SettlementProcess_Card'):
            SettlementProcess_Card = 1
            SettlementProcess_Check = 0
            SettlementProcess_Bank = 0
            SettlementProcess_Electronic = 0
                
        elif(SettlementProcess == 'SettlementProcess_Bank'):
            SettlementProcess_Card= 0
            SettlementProcess_Check = 0
            SettlementProcess_Bank = 1
            SettlementProcess_Electronic = 0
        
        elif(SettlementProcess == 'SettlementProcess_Check'):
            SettlementProcess_Card = 0
            SettlementProcess_Check = 1
            SettlementProcess_Bank = 0
            SettlementProcess_Electronic = 0
        else:
            SettlementProcess_Card = 0
            SettlementProcess_Check = 0
            SettlementProcess_Bank = 0
            SettlementProcess_Electronic = 1

        prediction = model.predict([[sex ,Aged,	Married	,TotalDependents,ServiceSpan,MobileService,CyberProtection,
        HardwareSupport,TechnicalAssistance,FilmSubscription,QuarterlyPayment,GrandPayment,GService_No,
        GService_Satellite_Broadband, GService_Wifi_Broadband ,SettlementProcess_Bank,SettlementProcess_Card,
        SettlementProcess_Check,SettlementProcess_Electronic]])
        if prediction==1:
             return render_template('index.html',prediction_text="The Customer will leave the bank")
        else:
             return render_template('index.html',prediction_text="The Customer will not leave the bank")
                
if __name__=="__main__":
    app.run(debug=True)
