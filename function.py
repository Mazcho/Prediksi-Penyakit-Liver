from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import streamlit as st

def mlStatsMetrics():
    df = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv")
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
    y_predgbc = model.predict(X_test)
    accgbc = accuracy_score(y_test,y_predgbc)
    st.write(accgbc)


