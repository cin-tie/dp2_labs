from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")

st.title("Лабораторная работа № 4")
st.markdown(
    """
    ### Классификация: kNN, деревья решений, SVM и MLP

    В работе рассматриваются две задачи: бинарная классификация Smoking и
    многоклассовая классификация Drug200. В разделе симулятора можно выбрать
    датасет, алгоритм и основные гиперпараметры. После изменения настроек модель
    переобучается на фиксированной обучающей выборке, а метрики и матрица ошибок
    пересчитываются на одной и той же тестовой выборке.

    """
)
st.divider()

BASE_DIR = Path(__file__).resolve().parent.parent
LAB4_DIR = BASE_DIR / "labs" / "lab4"
DATA_DIR = LAB4_DIR / "part1" / "datasets"

with st.sidebar:
    st.header("Навигация")
    section = st.selectbox(
        "Часть работы",
        [
            "Часть 1. Данные и постановка задачи",
            "Часть 2. Моделирование и оптимизация",
            "Часть 3. Оценка качества",
            "Часть 4. Интерактивный классификатор",
        ],
    )

if section.startswith("Часть 1"):
    from utils.notebook import render_notebook

    render_notebook(LAB4_DIR / "part1" / "lab4_part1_datasets.ipynb")
    st.stop()

if section.startswith("Часть 2"):
    from utils.notebook import render_notebook

    render_notebook(LAB4_DIR / "part2" / "lab4_part2_modeling.ipynb")
    st.stop()

if section.startswith("Часть 3"):
    from utils.notebook import render_notebook

    render_notebook(LAB4_DIR / "part3" / "lab4_part3_evaluation.ipynb")
    st.stop()


@st.cache_data
def load_datasets():
    smoking = pd.read_csv(DATA_DIR / "smoking.csv")
    drug = pd.read_csv(DATA_DIR / "drug200.csv")
    return smoking, drug


def get_split(dataset_key):
    smoking, drug = load_datasets()
    if dataset_key == "Smoking":
        df = smoking.copy()
        if len(df) > 8000:
            df, _ = train_test_split(
                df, train_size=8000, random_state=42, stratify=df["smoking"]
            )
        df = df.dropna(subset=["smoking"]).drop(columns=["ID"], errors="ignore")
        target = "smoking"
    else:
        df = drug.copy().dropna(subset=["Drug"])
        target = "Drug"

    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def make_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )


def build_estimator(model_name, params):
    if model_name == "kNN":
        return KNeighborsClassifier(
            n_neighbors=int(params["n_neighbors"]),
            weights=params["weights"],
            p=int(params["p"]),
        )
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            criterion=params["criterion"],
            random_state=42,
        )
    if model_name == "SVM":
        return SVC(
            C=float(params["C"]),
            kernel=params["kernel"],
            gamma=params["gamma"],
        )
    return MLPClassifier(
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        activation=params["activation"],
        alpha=float(params["alpha"]),
        max_iter=600,
        early_stopping=False,
        random_state=42,
    )


@st.cache_resource(show_spinner="Обучаем классификатор и рассчитываем метрики…")
def train_and_evaluate(dataset_key, model_name, params_json):
    params = json.loads(params_json)
    X_train, X_test, y_train, y_test = get_split(dataset_key)
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", build_estimator(model_name, params)),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    labels = sorted(pd.Series(y_train).unique().tolist(), key=str)

    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, predictions, labels=labels)
    cm_norm = confusion_matrix(y_test, predictions, labels=labels, normalize="true")
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Recall macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "F1 macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "Precision weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall weighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1 weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }
    return pipeline, metrics, report, cm, cm_norm, labels, y_test, predictions


st.subheader("Интерактивный классификатор")
st.write(
    "Выберите набор данных и модель. Основные параметры можно менять в боковой "
    "панели ниже; после изменения Streamlit пересчитает результат. Одинаковое "
    "разбиение train/test используется во всех сравнениях."
)

dataset_key = st.selectbox("Набор данных", ["Smoking", "Drug200"])
model_name = st.selectbox("Классификатор", ["kNN", "Decision Tree", "SVM", "MLP"])

st.markdown("#### Настройки гиперпараметров")
params = {}

if model_name == "kNN":
    params["n_neighbors"] = st.slider("Количество соседей (k)", 3, 15, 5, 2)
    params["weights"] = st.selectbox("Веса соседей", ["uniform", "distance"])
    params["p"] = st.selectbox(
        "Метрика Минковского: p=1 — Manhattan, p=2 — Euclidean", [1, 2], index=1
    )
elif model_name == "Decision Tree":
    depth_option = st.selectbox(
        "Максимальная глубина дерева",
        ["Без ограничения", "3", "5", "8", "12", "16", "20"],
    )
    params["max_depth"] = (
        None if depth_option == "Без ограничения" else int(depth_option)
    )
    params["min_samples_leaf"] = st.slider("Минимум объектов в листе", 1, 8, 1)
    params["criterion"] = st.selectbox(
        "Критерий разбиения", ["gini", "entropy", "log_loss"]
    )
elif model_name == "SVM":
    params["C"] = st.select_slider(
        "Параметр регуляризации C", options=[0.1, 1.0, 10.0, 100.0], value=1.0
    )
    params["kernel"] = st.selectbox("Ядро SVM", ["rbf", "linear"])
    params["gamma"] = st.selectbox("Параметр gamma", ["scale", "auto"])
else:
    architecture = st.selectbox(
        "Архитектура скрытых слоёв",
        ["32", "64", "100", "64 + 32"],
    )
    params["hidden_layer_sizes"] = {
        "32": [32],
        "64": [64],
        "100": [100],
        "64 + 32": [64, 32],
    }[architecture]
    params["activation"] = st.selectbox("Функция активации", ["relu", "tanh"])
    params["alpha"] = st.select_slider(
        "Сила L2-регуляризации alpha",
        options=[0.0001, 0.001, 0.01],
        value=0.0001,
    )

st.caption(
    "Примечание: подбор гиперпараметров с кросс-валидацией выполняется в части 2. "
    "Этот интерфейс нужен для интерактивного эксперимента с выбранными параметрами."
)

params_json = json.dumps(params, sort_keys=True)
try:
    pipeline, metrics, report, cm, cm_norm, labels, y_test, predictions = (
        train_and_evaluate(dataset_key, model_name, params_json)
    )

    st.markdown("#### Метрики на тестовой выборке")
    metric_cols = st.columns(4)
    for col, metric_name in zip(
        metric_cols, ["Accuracy", "Precision macro", "Recall macro", "F1 macro"]
    ):
        col.metric(metric_name, f"{metrics[metric_name]:.4f}")

    st.dataframe(
        pd.DataFrame([metrics]).T.rename(columns={0: "Значение"}).round(4),
        use_container_width=True,
    )

    st.markdown("#### Матрица ошибок")
    tab_counts, tab_normalized = st.tabs(
        ["Абсолютные значения", "Нормированная матрица"]
    )
    with tab_counts:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_title(f"{dataset_key} — {model_name}")
        st.pyplot(fig)
        plt.close(fig)
    with tab_normalized:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_title("Доля предсказаний внутри каждого истинного класса")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Classification report по классам")
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.round(4), use_container_width=True)

    st.markdown("#### Вывод о текущем эксперименте")
    st.write(
        f"Для набора **{dataset_key}** модель **{model_name}** с параметрами "
        f"`{params}` получила Accuracy = **{metrics['Accuracy']:.4f}** и "
        f"Macro-F1 = **{metrics['F1 macro']:.4f}**. Смотрите на показатели каждого "
        "класса в отчёте и на внедиагональные значения матрицы ошибок: они показывают, "
        "какие категории модель путает."
    )
except Exception as exc:
    st.error(f"Не удалось обучить модель: {exc}")
    st.info(
        "Проверьте зависимости из requirements.txt и наличие CSV в "
        "labs/lab4/part1/datasets/."
    )
