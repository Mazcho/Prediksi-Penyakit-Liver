import streamlit as st
import numpy as np
import pandas as pd
import function as ml

st.set_page_config(
    page_title = "Liver Classification"
)
st.title("Liver Disosder Classification")

tab1,tab2,tab3 = st.tabs(["Home","DataSet","Prediction"])
with tab1:
    df_web = pd.read_csv("Deployment/data/Indian Liver Patient Dataset (ILPD).csv")
    st.image("Deployment/aset/header.png")
    st.subheader("What is _Liver Disorder?_")
    st.markdown("Liver disorder refers to any condition or disease that affects the functioning of the liver. The liver is a vital organ responsible for numerous essential functions in the body. Deaths from cirrhosis of the liver continue to increase due to rising rates of alcohol consumption, chronic hepatitis infections, and obesity-related liver disease. Despite the high mortality of this disease, liver disease does not affect all subpopulations equally. Early detection of pathology is a determinant of patient outcomes, yet female patients appear to be marginalized when it comes to early diagnosis of liver pathology. The dataset consists of 584 patient records collected from the North East of Andhra Pradesh, India. The prediction task is to determine whether a patient suffers from liver disease based on information about several biochemical markers, including albumin and other enzymes required for metabolism.")

    st.subheader("What age does liver disease affect?")
    df_filtered = df_web[df_web["Selector2"] == 0]
    value_counts = df_filtered["Age"].value_counts().sort_index(ascending=True)
    st.bar_chart(data=value_counts)

    col1,col2 = st.columns(2)

    with col1:
        st.write("Based on the age of those affected by liver disease, many occur at the age of 60 years, while in young age the highest occurs at the age of 17 years. The graph above shows that age susceptibility is one of the factors that can occur in the liver. Possibilities that occur if affected by liver disorders since adolescence can be caused by excessive alcohol consumption, drugs and others. If in old age it could be due to liver organ dysfunction, or indeed unable to maintain liver health due to unhealthy consumption. From a gender analysis, there are more cases of men than women.")
    with col2:
        df_filtered_age = df_web[df_web["Selector2"] == 0]
        st.bar_chart(data = df_filtered_age["Gender"].value_counts())
with tab2:
    st.image("aset/datasetkaggle.png")
    st.subheader("Where do I get the record?")
    url = "https://www.kaggle.com/datasets/fatemehmehrparvar/liver-disorders/data"
    text_link = "Link Dataset here."
    hyperlink = f"[{text_link}]({url})"
    st.write(f"I got this dataset from one of the public data platforms called Kaggle. This dataset is the latest updated dataset a few days ago. It contains information about the predictor and target variables of liver diseases.{hyperlink}",unsafe_allow_html=True)
    st.dataframe(df_web.drop("Selector",axis=1))
    st.markdown("""Based on the additional information provided about the dataset related to liver disorders, here's a description of the features:

1. **Age**: Age of the patient.
2. **Gender**: Gender of the patient (male or female).
3. **Total Bilirubin**: The total amount of bilirubin in the blood, which can indicate liver function and health. Elevated levels may suggest liver disease.
4. **Direct Bilirubin**: The direct bilirubin level, which is a specific type of bilirubin. Elevated levels may indicate liver or bile duct problems.
5. **Total Proteins**: Total protein levels in the blood, which can be affected by liver function. Abnormal levels may indicate liver disease.
6. **Albumin**: Albumin is a protein produced by the liver. Abnormal levels may indicate liver disease or other health issues.
7. **A/G Ratio**: The albumin/globulin ratio, which can provide information about liver and kidney function. Abnormal levels may indicate liver disease.
8. **SGPT (Serum Glutamate Pyruvate Transaminase)**: Also known as ALT (alanine transaminase), SGPT is an enzyme found in the liver. Elevated levels may indicate liver damage or disease.
9. **SGOT (Serum Glutamic Oxaloacetic Transaminase)**: Also known as AST (aspartate transaminase), SGOT is an enzyme found in the liver and other organs. Elevated levels may indicate liver damage or disease.
10. **Alkphos (Alkaline Phosphatase)**: Alkaline phosphatase is an enzyme found in the liver, bones, and other tissues. Elevated levels may indicate liver or bone disease.

These features are important biochemical markers used to assess liver health and diagnose liver disorders. By analyzing these features, healthcare professionals can identify patterns and trends that may help in diagnosing liver diseases, predicting patient outcomes, and designing appropriate treatment plans. Additionally, the dataset aims to explore differences in liver diseases among patients across different demographics, including gender disparities in predicting liver disease, and to understand how these biochemical markers may vary in effectiveness for male and female patients.""")
with tab3:
    ml.mlStatsMetrics()
    ml.singlePredict()