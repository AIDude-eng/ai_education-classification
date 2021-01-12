import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("AI in Education JKU Cool Lab")

st.write("""
### Mit dem Pfeil oben links kannst du eine Seitenansicht aufrufen. Nun kannst du in der Seitenansicht links ein Dataset und einen Classifier auswählen. Zusätzlich kannst du für jeden Classifier gewisse Parameter einstellen.
""")

st.sidebar.write("## Auswahlmenü")
dataset_name = st.sidebar.selectbox("Wähle ein Dataset aus", ("Iris", "Breast Cancer", "Wine Dataset"))

classifier_name = st.sidebar.selectbox("Wähle einen Classifier aus", ("KNN", "SVM", "Random Forest", "MLPClassifier"))

st.sidebar.write("Wähle Werte für die Parameter des ausgewählten Classifiers aus:")

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()

    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()

    elif dataset_name == "Wine Dataset":
        data = datasets.load_wine()

    X = data.data
    y = data.target

    return data, X, y


data, X, y = get_dataset(dataset_name)


def add_parameter_ui(clf_name):

    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        kernel = st.sidebar.selectbox("kernel", ("rbf", "linear", "poly"))
        params["C"] = C
        params["kernel"] = kernel
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        criterion = st.sidebar.selectbox("criterion", ("gini", "entropy"))
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
    elif clf_name == "MLPClassifier":
        learning_rate_init = st.sidebar.slider("learning rate", 0.01, 1.00)
        max_iter = st.sidebar.slider("max_iter", 1, 300)
        params["learning_rate_init"] = learning_rate_init
        params["max_iter"] = max_iter

    return params


params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"], kernel=params["kernel"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], criterion=params["criterion"], random_state=1234)
    elif clf_name == "MLPClassifier":
        clf = MLPClassifier(learning_rate_init=params["learning_rate_init"], max_iter=params["max_iter"])

    return clf


def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['Klasse'] = pd.Series(sklearn_dataset.target)
    return df


clf = get_classifier(classifier_name, params)

# Description of dataset
st.write("## Beschreibung des ausgewählten Datasets")
st.write(f"Dataset:  {dataset_name}")
st.write(f"Größe: {X.shape[0]} Einträge mit jeweils {X.shape[1]} variablen")
st.write(f"Anzahl verschiedener Klassen: {len(np.unique(y))}")
df = sklearn_to_df(data)
st.table(df.head())
# Add further description of dataset so kids have a understanding what this is
# TODO


# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

y_pred_all = clf.predict(X)

# Description of classifier
st.write("## Beschreibung des ausgewählten Classifiers")
st.write(f"Classifier:  {classifier_name}")
st.write(f"Genauigkeit = {acc*100:.2f}%")
# Descripe very general what each classifier does
# TODO

# Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig1 = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()

st.write(f"## Abbildung des {dataset_name} Datasets")
st.write("### Jeder Punkt ist ein Eintrag im Dataset und die Farbe bestimmt, zu welcher Klasse dieser Eintrag gehört.")
st.pyplot(fig1)

fig2 = plt.figure()
plt.scatter(x1, x2, c=y_pred_all, alpha=0.8, cmap="viridis")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()

st.write(f"## Klassifizierung")
st.write("### Die Punkte sind an der selben Position, jedoch bestimmt nun der Classifier welche Farbe, also welche Klasse die Einträge sind.")
st.pyplot(fig2)

