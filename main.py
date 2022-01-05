import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
import streamlit as st
from streamlit import cli as stcli
import sys

def main():
    st.set_page_config(page_title="Support Vector Regression", page_icon="📈", layout="wide")
    st.title("📈 Support Vector Regression")
    st.write("Data Mining und Visual Analytics")
    #st.write("**Standardparameter: class sklearn.svm.SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, "
    #         "C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)**")
    st.write("Im folgenden wird ein kleiner Datensatz bestehend aus Berufserfahrung in Jahren und dem jährlichen Gehalt geladen. "
             "Anschließend wird mithilfe des Support Vector Regression Algorithmus der Datensatz analysiert und visualisiert.")
    st.write("Über den Filter auf der linken Seite können die verschiedenen Parameter verändert werden." 
             " Bei mobilen Geräten ist der Filter standardmäßig ausgeblendet und lässt sich mit dem Pfeil oben links aktivieren.")
    with st.sidebar.header("train_test_split"):
        split_size = st.sidebar.slider("Aufteilen in Traings- und Testdaten (Standard: 30% Testdaten):", 10, 90, 30, 5)
    with st.sidebar.subheader("Parameter"):
        kernel_select = st.sidebar.selectbox("kernel auswählen", options=["linear", "poly", "rbf"], index=2)
        C_select = st.sidebar.slider("C auswählen:", 1, 100000, 10000, 100)
        epsilon_select = st.sidebar.slider("epsilon auswählen:", 0.1, 40000.0, 100.0, 0.1)
    with st.sidebar.subheader("Vorhersage"):
        pred_salary= st.sidebar.slider("Für Gehaltsvorhersage Berufserfahrung auswählen:", 1.0, 20.0, 1.0, 0.1)
    #einlesen des Dataframes
    df=pd.read_csv(r"https://raw.githubusercontent.com/tobiarnold/Support-Vector-Regression/main/Salary_Data.csv")
    st.dataframe(df.style.format({"YearsExperience": "{:.1f}", "Salary": "{:.0f}"}), width=4000)
    #initalisieren der SVR
    X = df["YearsExperience"].to_numpy().reshape(-1, 1)
    y = df["Salary"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size/100, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    model = SVR(kernel=kernel_select, C = C_select, epsilon = epsilon_select)
    model.fit(X_train, y_train)
    model_train=model.score(X_train, y_train)
    model_test=model.score(X_test, y_test)
    st.write("Score Trainingsdaten:",model_train)
    st.write("Score Testdaten:",model_test)
    model.fit(X, y)
    #Diagramm
    f, ax = plt.subplots(nrows=1, ncols=1,figsize=(10, 5))
    ax.scatter(X,y, color="blue",label="Daten")
    ax.scatter(sc.inverse_transform(X_test), y_test, color="green", label="Testdaten")
    ax.scatter(X[model.support_],  y[model.support_], color="purple", label="Vektoren")
    ax.plot(X, model.fit(X, y).predict(X), color="orange", label="Vorhersage")
    ax.plot(X, model.fit(X, y).predict(X)+epsilon_select, color="black",linestyle="dashed", label="epsilon")
    ax.plot(X, model.fit(X, y).predict(X)-epsilon_select, color="black",linestyle="dashed")
    plt.xlabel("Berufserfahrung in Jahren")
    plt.ylabel("Gehalt")
    ax.legend()
    st.pyplot(f)
    pred_salary=np.array([pred_salary]).reshape(1,-1)
    pred_salary_result=model.predict(pred_salary)
    st.write("Gehaltsvorhersage nach Jahren", pred_salary_result)
    #link = "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html"
    #st.markdown(link, unsafe_allow_html=True)
    
if __name__ == "__main__":
  main()
