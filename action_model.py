from flask import Flask,render_template

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("frontend1.html")

@app.route('/predict')
def predict():

    df=pd.read_csv("best_crop.csv")
    le=LabelEncoder()
    df["crop_type"]=le.fit_transform(df["crop_type"])
    x=df.drop(["crop_type"],axis=1)
    y=df["crop_type"]
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
    model=xgb.XGBClassifier(n_estimators=1000,max_depth=2,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,reg_alpha=5,reg_lambda=10,eval_metric='merror',early_stopping_rounds=10,random_state=42)
    model.fit(xtrain,ytrain,eval_set=[(xtest,ytest)],verbose=True)
    y_pred=model.predict(xtest)
    import serial
    ser = serial.Serial('/dev/ttyUSB0', 9600) 
    line = ser.readline().decode().strip()
    values = line.split(',')
    Soil_Moisture = int(values[0])
    Temperature = float(values[1])
    Humidity = float(values[2])
    pH = float(values[3])
    Light = int(values[4])
    Rain_digital = int(values[5])
    Rain_analog = int(values[6])
    input_df=pd.DataFrame([[Soil_Moisture,Temperature,Humidity,pH,Light,Rain_digital,Rain_analog]],columns=["Soil_Moisture","Temperature","Humidity","pH","Light","Rain_digital","Rain_analog"])
    prob=model.predict(input_df)[0]
    crop_type=(le.inverse_transform([prob])[0]).lower()
    crop=crop_type
    print("Train accuracy:", model.score(xtrain, ytrain) * 100)
    print("Test accuracy :", model.score(xtest, ytest) * 100)




    df=pd.read_csv("action.csv")
    df['crop_type']=le.fit_transform(df['crop_type'])
    le1=LabelEncoder()
    df['action']=le1.fit_transform(df['action'])
    x=df.drop(['action'],axis=1)
    y=df['action']
    x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2,random_state=0)
    model=xgb.XGBClassifier(n_estimators=50,max_depth=2,learning_rate=0.05,subsample=0.7,colsample_bytree=0.7,reg_alpha=2,reg_lambda=4,eval_metric='merror',random_state=42,early_stopping_rounds=10)
    model.fit(x_train,y_train,eval_set=[(x_test,y_test)],verbose=False)
    y_pred=model.predict(x_test)
    crop_type=le.transform([crop_type])[0]
    input_df=pd.DataFrame([[Soil_Moisture,Temperature,Humidity,pH,Light,Rain_digital,Rain_analog,crop_type]],columns=["Soil_Moisture","Temperature","Humidity","pH","Light","Rain_digital","Rain_analog","crop_type"])
    prob=model.predict(input_df)[0]
    action=le1.inverse_transform([prob])[0]
    print("Train accuracy:", model.score(x_train, y_train) * 100)
    print("Test accuracy :", model.score(x_test, y_test) * 100)



    
    df=pd.read_csv("price.csv")
    le2=LabelEncoder()
    df['crop_type']=le.fit_transform(df['crop_type'])
    df['cost']=le2.fit_transform(df['cost'])
    x=df.drop("cost",axis=1)
    y=df["cost"]
    model=xgb.XGBClassifier(n_estimators=1200,max_depth=3,learning_rate=0.03,subsample=0.9,colsample_bytree=0.9,reg_alpha=0.5,reg_lambda=2,eval_metric='merror',early_stopping_rounds=20,random_state=42)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0,stratify=y)
    model.fit(x_train, y_train,eval_set=[(x_test, y_test)],verbose=True)
    y_pred=model.predict(x_test)
    prob=model.predict(input_df)[0]
    cost=le2.inverse_transform([prob])[0]
    print("Train accuracy:", model.score(x_train, y_train) * 100)
    print("Test accuracy :", model.score(x_test, y_test) * 100)

    return render_template("frontend2.html", 
                           crop=crop, 
                           action=action, 
                           cost=cost,
                           soilMoisture=Soil_Moisture,
                           temperature=Temperature,
                           humidity=Humidity,
                           ph=pH,
                           light=Light,
                           rainDigital=Rain_digital,
                           rainAnalog=Rain_analog)

if __name__ == "__main__":
    app.run(debug=True)