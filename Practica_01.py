# ======================================================
# Trabajando con árboles de decisión y técnicas de ensemble learning
# Dataset: Car Evaluation
# Nombre: Miguel Angel C.
# ======================================================

# ===============================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ===============================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# ===============================
# 2. VARIABLES GLOBALES
# ===============================
df = None
df_encoded = None

X = y = None
X_train = X_test = y_train = y_test = None

encoders = {}
class_names = None

models = {}
accuracy_results = {}

# ===============================
# 3. FUNCIONES
# ===============================

def cargar_dataset():
    global df
    df = pd.read_csv("Laboratorio_dataset_car.csv", sep=";")

    print("\n=== Dataset cargado correctamente ===\n")
    print(df.head())
    print("\nInformación general:\n")
    print(df.info())
    print("\nInstancias:", len(df))
    print("\nValores nulos:\n", df.isnull().sum())


def analisis_exploratorio():
    if df is None:
        print("Primero debes cargar el dataset")
        return

    print("\n=== Análisis Exploratorio de Datos ===")

    # Distribución de clases
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="class")
    plt.title("Distribución de la variable objetivo")
    plt.show()

    # Codificación temporal para correlación
    df_temp = df.copy()
    for col in df_temp.columns:
        df_temp[col] = LabelEncoder().fit_transform(df_temp[col])

    plt.figure(figsize=(8,6))
    sns.heatmap(df_temp.corr(), annot=True, cmap="coolwarm")
    plt.title("Heatmap de correlación")
    plt.show()

    # Histogramas
    df_temp.hist(figsize=(10,8))
    plt.suptitle("Histogramas (codificación ordinal)")
    plt.show()


def preprocesamiento():
    global df_encoded, X, y, X_train, X_test, y_train, y_test
    global encoders, class_names

    if df is None:
        print("Primero debes cargar el dataset")
        return

    df_encoded = df.copy()
    encoders.clear()

    for col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    X = df_encoded.drop("class", axis=1)
    y = df_encoded["class"]

    class_names = encoders["class"].classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Preprocesamiento finalizado")


def entrenar_modelos():
    global models

    if X_train is None:
        print("Ejecuta el preprocesamiento primero")
        return

    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)

    tree.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    models = {
        "Decision Tree": tree,
        "Random Forest": rf
    }

    print("Modelos entrenados correctamente")



def evaluar_modelos():
    if not models:
        print("No hay modelos entrenados")
        return

    accuracy_results.clear()

    for name, model in models.items():
        print(f"\n========== {name} ==========")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracy_results[name] = acc

        print("Accuracy:", acc)
        print("\nClassification Report:\n")

        print(classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0
        ))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues"
        )
        plt.title(f"Matriz de confusión - {name}")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.show()


def curva_roc():
    if not models:
        print("No hay modelos entrenados")
        return

    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    plt.figure(figsize=(8,6))

    for name, model in models.items():
        y_score = model.predict_proba(X_test)

        fpr, tpr = {}, {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(
                y_test_bin[:, i], y_score[:, i]
            )

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes
        auc_macro = auc(all_fpr, mean_tpr)

        plt.plot(all_fpr, mean_tpr, label=f"{name} (AUC={auc_macro:.2f})")

    plt.plot([0,1], [0,1], 'k--')
    plt.title("Curva ROC Multiclase (Macro-average)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


def comparar_accuracy():
    if not accuracy_results:
        print("Primero evalúa los modelos")
        return

    plt.figure(figsize=(6,4))
    sns.barplot(
        x=list(accuracy_results.keys()),
        y=list(accuracy_results.values())
    )
    plt.title("Comparación de Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.show()


# ===============================
# 4. MENÚ INTERACTIVO
# ===============================

def menu():
    while True:
        print("""
==================== MENÚ ====================
1. Cargar Dataset
2. Análisis Exploratorio (EDA)
3. Preprocesamiento
4. Entrenar Modelos
5. Evaluar Modelos
6. Curva ROC Multiclase
7. Comparar Accuracy
0. Salir
=============================================
""")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            cargar_dataset()
        elif opcion == "2":
            analisis_exploratorio()
        elif opcion == "3":
            preprocesamiento()
        elif opcion == "4":
            entrenar_modelos()
        elif opcion == "5":
            evaluar_modelos()
        elif opcion == "6":
            curva_roc()
        elif opcion == "7":
            comparar_accuracy()
        elif opcion == "0":
            print("\n👋 Ejecución finalizada\n")
            break
        else:
            print("Opción inválida")


# ===============================
# 5. EJECUCIÓN
# ===============================
if __name__ == "__main__":
    menu()

