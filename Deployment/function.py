from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import streamlit as st
import joblib

def mlStatsMetrics():
    path = "Deployment/data/Indian Liver Patient Dataset (ILPD).csv"
    df = pd.read_csv(path)
    df.drop("Selector",axis=1,inplace=True)
    labelencoder = LabelEncoder()
    df["Gender"] = labelencoder.fit_transform(df["Gender"])
    df["Gender"] = df["Gender"].astype(int)
    df.drop_duplicates(inplace=True)
    #ngisi missing value
    df["A/G Ratio"].fillna(df["A/G Ratio"].mean(),axis=0,inplace=True)
    X = df.drop("Selector2",axis=1)
    y = df["Selector2"]
    X.fillna(X.mean(),inplace=True)
    scaler = SMOTE(k_neighbors=3, random_state=42)
    X_resampled,y_resampled = scaler.fit_resample(X,y)
    X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2, random_state=42)
    model = GradientBoostingClassifier()
    model.fit(X_train,y_train)
    y_predgbctest = model.predict(X_test)
    y_predgbctrain = model.predict(X_train)
    accgbctrain = round(accuracy_score(y_train,y_predgbctrain)*100,2)
    accgbctest = round(accuracy_score(y_test,y_predgbctest)*100,2)
    st.write("Model Name : 'GradientBoostingClasifier' ")
    st.write("Accuracy Training : , ",accgbctrain,"|| Accuracy Testing : ",accgbctest)

def singlePredict():
    col1,col2 = st.columns(2)
    with col1:
        age = st.number_input("Age")
        Gender = st.number_input("Gender")
        Total_Bilirubin = st.number_input("TB")
        Direct_Bilirubin = st.number_input("DB")
        alkphos = st.number_input("Alkphos")
    with col2:
        sgpt = st.number_input("Sgpt")
        sgot= st.number_input("Sgot")
        tp = st.number_input("TP")
        alb = st.number_input("ALB")
        agratio = st.number_input("A/G Ration")
    
    if st.button("Predict"):
        user_input=[[age,Gender,Total_Bilirubin,Direct_Bilirubin,alkphos,sgpt,sgot,tp,alb,agratio]]
        model = joblib.load("model/modelGBC.joblib")
        prediction = model.predict(user_input)
        if prediction == 0:
            st.error("Liver Disorder")
        elif prediction == 1:
            st.succes("Not Liver Disorder")