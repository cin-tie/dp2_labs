import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.notebook import render_notebook

BASE_DIR = Path(__file__).resolve().parent.parent
LAB4_DIR = BASE_DIR / "labs" / "lab4"
DATASETS_DIR = LAB4_DIR / "part1" / "datasets"
PART2_RESULTS = LAB4_DIR / "part2" / "results"
PART3_RESULTS = LAB4_DIR / "part3" / "results"

RANDOM_STATE = 42  # как в ноутбуках: одинаковое разбиение train/test
MODEL_NAMES = {"knn": "kNN", "tree": "Decision Tree", "svm": "SVM", "mlp": "MLP"}

st.title("Лабораторная работа №4")
st.markdown("### Сравнительный анализ алгоритмов классификации")

st.markdown(
    """
    **Цель работы:** Практическое освоение методов машинного обучения для решения
    задач классификации, изучение метрик оценки качества классификаторов,
    исследование влияния гиперпараметров на точность моделей, а также применение
    интерактивной визуализации.
    """
)

st.divider()

PARTS = {
    "Часть 1. Данные и постановка задачи": ("part1", "lab4_part1_datasets.ipynb"),
    "Часть 2. Моделирование и оптимизация": ("part2", "lab4_part2_models.ipynb"),
    "Часть 3. Оценка качества": ("part3", "lab4_part3_evaluation.ipynb"),
    "Часть 4. Интерактивное приложение": None,
}

DATASET_LABELS = {
    "smoking": "Body signal of smoking (бинарная)",
    "drug": "Drug classification (5 классов)",
}
TARGET_NAMES = {"smoking": ["не курит (0)", "курит (1)"]}

C_OPTIONS = [0.01, 0.1, 1, 10, 100, 1000]
GAMMA_OPTIONS = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
ALPHA_OPTIONS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
LR_OPTIONS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
NEURON_OPTIONS = [8, 16, 32, 64, 128, 256]


# ---------------------------------------------------------------------------
# Данные и модели. Тот же код, что в ноутбуках частей 1-2 (одинаковые выборки).
# ---------------------------------------------------------------------------
def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    transformers = [("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                      ("scaler", StandardScaler())]), numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]), categorical_cols))
    return ColumnTransformer(transformers)


def load_dataset(key):
    if key == "smoking":
        df = pd.read_csv(DATASETS_DIR / "smoking.csv").drop(columns=["ID", "oral"])
        df = df.drop_duplicates().reset_index(drop=True)
        return df.drop(columns="smoking"), df["smoking"].astype(int)
    df = pd.read_csv(DATASETS_DIR / "drug200.csv")
    return df.drop(columns="Drug"), df["Drug"].astype(str)


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def subsample_stratified(X, y, n):
    if n >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=RANDOM_STATE)
    return X_sub, y_sub


def make_pipeline(key, name, X_train, **params):
    if name == "knn":
        clf = KNeighborsClassifier(**params)
    elif name == "tree":
        clf = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    elif name == "svm":
        clf = SVC(random_state=RANDOM_STATE, **params)
    else:
        small = key == "drug"  # 160 объектов: early_stopping выключен, нужно больше эпох
        defaults = dict(max_iter=2000 if small else 300, early_stopping=not small,
                        n_iter_no_change=10, random_state=RANDOM_STATE)
        defaults.update(params)
        clf = MLPClassifier(**defaults)
    return Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    res = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    if len(model.classes_) == 2:
        score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        res["ROC-AUC"] = roc_auc_score(y_test, score)
    return res


def load_best_params():
    """Лучшие параметры из части 2 (results/best_params.json)."""
    path = PART2_RESULTS / "best_params.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    for models in data.values():
        for params in models.values():
            if "hidden_layer_sizes" in params:
                params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    return data


# ---------------------------------------------------------------------------
# Данные и обучение (кэшируются, чтобы повторные клики по тем же настройкам
# не пересчитывались)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_split(key):
    X, y = load_dataset(key)
    return split_data(X, y)


@st.cache_data(show_spinner=False)
def get_reference(key):
    """Таблица оптимизированных моделей из части 3 (если ноутбук уже выполнен)."""
    path = PART3_RESULTS / f"comparison_{key}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_resource(show_spinner=False)
def fit_and_evaluate(key, name, params_items, n_train):
    X_train, X_test, y_train, y_test = get_split(key)
    X_fit, y_fit = subsample_stratified(X_train, y_train, n_train)

    pipe = make_pipeline(key, name, X_train, **dict(params_items))
    t0 = time.perf_counter()
    pipe.fit(X_fit, y_fit)
    fit_time = time.perf_counter() - t0

    y_pred = pipe.predict(X_test)
    labels = list(pipe.classes_)
    return {
        "metrics": compute_metrics(pipe, X_test, y_test),
        "cm": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
        "report": classification_report(
            y_test, y_pred, labels=labels,
            target_names=TARGET_NAMES.get(key, labels),
            output_dict=True, zero_division=0,
        ),
        "fit_time": fit_time,
        "n_train": len(X_fit),
        "n_test": len(X_test),
    }


# ---------------------------------------------------------------------------
# Виджеты гиперпараметров
# ---------------------------------------------------------------------------
def _state_key(key, name, param, default=None):
    """Ключ виджета; при первом обращении записывает в session_state значение по умолчанию."""
    widget_key = f"{key}_{name}_{param}"
    if default is not None or param == "unlimited":
        st.session_state.setdefault(widget_key, default)
    return widget_key


def hyperparameter_widgets(key, name):
    """Рисует элементы управления для выбранной модели и возвращает словарь параметров.

    Значения по умолчанию кладём в session_state до создания виджета (а не через
    аргумент value), чтобы кнопка «Подставить лучшие параметры» могла их менять
    без предупреждений Streamlit.
    """
    k = lambda p, default=None: _state_key(key, name, p, default)  # noqa: E731

    if name == "knn":
        return {
            "n_neighbors": st.slider("Число соседей (k)", 1, 50, key=k("n_neighbors", 11)),
            "weights": st.selectbox("Веса соседей", ["uniform", "distance"], key=k("weights", "uniform")),
            "p": st.radio("Метрика расстояния", [1, 2], key=k("p", 2),
                          format_func=lambda v: "Манхэттенская (p=1)" if v == 1 else "Евклидова (p=2)",
                          horizontal=True),
        }

    if name == "tree":
        unlimited = st.checkbox("Глубина без ограничения", key=k("unlimited", False))
        depth = None
        if not unlimited:
            depth = st.slider("Максимальная глубина", 1, 30, key=k("max_depth", 7))
        return {
            "criterion": st.selectbox("Критерий разбиения", ["gini", "entropy"], key=k("criterion", "gini")),
            "max_depth": depth,
            "min_samples_leaf": st.slider("Мин. объектов в листе", 1, 50, key=k("min_samples_leaf", 5)),
        }

    if name == "svm":
        kernel = st.selectbox("Ядро", ["rbf", "linear", "poly"], key=k("kernel", "rbf"))
        params = {
            "kernel": kernel,
            "C": st.select_slider("Параметр регуляризации C", C_OPTIONS, key=k("C", 1)),
        }
        if kernel in ("rbf", "poly"):
            params["gamma"] = st.selectbox("gamma", GAMMA_OPTIONS, key=k("gamma", "scale"))
        if kernel == "poly":
            params["degree"] = st.slider("Степень полинома", 2, 5, key=k("degree", 3))
        return params

    # MLP
    n_layers = st.slider("Число скрытых слоёв", 1, 3, key=k("n_layers", 2))
    defaults = [64, 32, 16]
    sizes = []
    for i, col in enumerate(st.columns(n_layers)):
        with col:
            sizes.append(st.select_slider(f"Слой {i + 1}", NEURON_OPTIONS, key=k(f"layer{i + 1}", defaults[i])))
    return {
        "hidden_layer_sizes": tuple(sizes),
        "activation": st.selectbox("Функция активации", ["relu", "tanh", "logistic"], key=k("activation", "relu")),
        "alpha": st.select_slider("L2-регуляризация (alpha)", ALPHA_OPTIONS, key=k("alpha", 1e-3)),
        "learning_rate_init": st.select_slider("Скорость обучения", LR_OPTIONS, key=k("learning_rate_init", 1e-3)),
        # drug: всего 160 объектов, early_stopping выключен, поэтому сети нужно больше эпох
        "max_iter": st.slider("Макс. число эпох", 50, 2000, step=50, key=k("max_iter", 2000 if key == "drug" else 300)),
    }


def apply_best_params(key, name):
    """Callback кнопки: записывает лучшие параметры из GridSearch в состояния виджетов."""
    best = load_best_params().get(key, {}).get(name)
    if not best:
        return
    k = lambda p: f"{key}_{name}_{p}"  # noqa: E731

    def put(widget, value, options=None):
        if options is None or value in options:
            st.session_state[k(widget)] = value

    if name == "knn":
        put("n_neighbors", best["n_neighbors"], range(1, 51))
        put("weights", best["weights"])
        put("p", best["p"])
    elif name == "tree":
        st.session_state[k("unlimited")] = best.get("max_depth") is None
        if best.get("max_depth") is not None:
            put("max_depth", best["max_depth"], range(1, 31))
        put("criterion", best["criterion"])
        put("min_samples_leaf", best["min_samples_leaf"], range(1, 51))
    elif name == "svm":
        put("kernel", best["kernel"])
        put("C", best["C"], C_OPTIONS)
        if "gamma" in best:
            put("gamma", best["gamma"], GAMMA_OPTIONS)
        if "degree" in best:
            put("degree", best["degree"], range(2, 6))
    else:
        sizes = best["hidden_layer_sizes"]
        put("n_layers", len(sizes), range(1, 4))
        for i, s in enumerate(sizes):
            put(f"layer{i + 1}", s, NEURON_OPTIONS)
        put("activation", best["activation"])
        put("alpha", best["alpha"], ALPHA_OPTIONS)
        put("learning_rate_init", best["learning_rate_init"], LR_OPTIONS)


# ---------------------------------------------------------------------------
# Графики и таблицы
# ---------------------------------------------------------------------------
def plot_confusion(cm, labels, key, normalize):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    if normalize:
        data = cm / cm.sum(axis=1, keepdims=True)
        annot = np.array([[f"{cm[i, j]}\n({data[i, j]:.0%})" for j in range(len(labels))] for i in range(len(labels))])
    else:
        data = cm
        annot = cm
    ticks = TARGET_NAMES.get(key, labels)
    sns.heatmap(data, annot=annot, fmt="" if normalize else "d", cmap="Blues", cbar=False,
                xticklabels=ticks, yticklabels=ticks, linewidths=0.5, linecolor="white", ax=ax, annot_kws={"size": 11})
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинный класс")
    fig.tight_layout()
    return fig


def report_to_frame(report):
    df = pd.DataFrame(report).T
    if "accuracy" in df.index:  # у строки accuracy sklearn дублирует число во всех столбцах
        df.loc["accuracy", "support"] = df.loc["macro avg", "support"]
    df.index.name = "Класс"
    return df.rename(columns={"precision": "Precision", "recall": "Recall",
                              "f1-score": "F1-score", "support": "Support"})


def params_to_text(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


# ---------------------------------------------------------------------------
# Часть 4
# ---------------------------------------------------------------------------
def render_part4():
    st.subheader("Интерактивное сравнение классификаторов")
    st.caption(
        "Выберите датасет и классификатор в боковой панели, меняйте гиперпараметры — "
        "модель переобучается на обучающей выборке, а матрица неточностей и метрики "
        "на тестовой выборке обновляются сразу."
    )

    with st.sidebar:
        st.divider()
        st.subheader("Модель")
        key = st.selectbox("Датасет", list(DATASET_LABELS), format_func=DATASET_LABELS.get, key="lab4_dataset")
        name = st.selectbox("Классификатор", list(MODEL_NAMES), format_func=MODEL_NAMES.get, key="lab4_model")

        best_all = load_best_params()
        if best_all.get(key, {}).get(name):
            st.button("Подставить лучшие параметры (GridSearch)", on_click=apply_best_params,
                      args=(key, name), width="stretch")
        else:
            st.caption("Лучшие параметры появятся после выполнения ноутбука части 2.")

        st.subheader("Гиперпараметры")
        params = hyperparameter_widgets(key, name)

        X_train, X_test, _, _ = get_split(key)
        n_full = len(X_train)
        if key == "smoking":
            options = [n for n in (2000, 5000, 10000, 20000) if n < n_full] + [n_full]
            n_train = st.select_slider("Размер обучающей выборки", options, value=n_full, key="lab4_ntrain")
            if name == "svm" and n_train > 15000:
                st.warning("SVM на таком объёме обучается десятки секунд.")
        else:
            n_train = n_full

        st.subheader("Отображение")
        normalize = st.toggle("Матрица в долях (%) по строкам", value=True)
        auto = st.toggle("Обновлять автоматически", value=True,
                         help="Выключите для тяжёлых моделей и запускайте обучение кнопкой.")
        run_clicked = auto or st.button("Обучить модель", type="primary", width="stretch")

    config_key = (key, name, tuple(sorted(params.items(), key=lambda kv: kv[0])), n_train)

    if run_clicked or "lab4_last" not in st.session_state:
        with st.spinner("Обучение модели..."):
            result = fit_and_evaluate(key, name, config_key[2], n_train)
        st.session_state["lab4_last"] = (config_key, result)
    config_key, result = st.session_state["lab4_last"]
    key, name, params_items, n_train = config_key
    params = dict(params_items)

    if "lab4_history" not in st.session_state:
        st.session_state["lab4_history"] = {}
    st.session_state["lab4_history"][config_key] = {
        "Датасет": key, "Модель": MODEL_NAMES[name], "Параметры": params_to_text(params),
        "Объектов в train": result["n_train"], **result["metrics"], "Время обучения, с": result["fit_time"],
    }

    # --- шапка
    classes = result["labels"]
    st.markdown(
        f"**{DATASET_LABELS[key]}** · классификатор **{MODEL_NAMES[name]}** · "
        f"train: {result['n_train']} · test: {result['n_test']} · классов: {len(classes)} · "
        f"обучение: {result['fit_time']:.2f} с"
    )
    st.code(params_to_text(params), language="text")

    # --- метрики с дельтой относительно оптимизированной модели из части 3
    ref_table = get_reference(key)
    ref = ref_table.loc[MODEL_NAMES[name]] if ref_table is not None and MODEL_NAMES[name] in ref_table.index else None
    metric_names = [m for m in ("Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "ROC-AUC")
                    if m in result["metrics"]]
    cols = st.columns(len(metric_names))
    for col, m in zip(cols, metric_names):
        value = result["metrics"][m]
        delta = f"{value - ref[m]:+.3f}" if ref is not None and m in ref and pd.notna(ref[m]) else None
        col.metric(m, f"{value:.3f}", delta=delta,
                   help="Δ относительно оптимизированной модели этого типа из части 3" if delta else None)

    # --- матрица неточностей + отчёт
    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("**Матрица неточностей (test)**")
        fig = plot_confusion(result["cm"], classes, key, normalize)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Классификационный отчёт (test)**")
        st.dataframe(
            report_to_frame(result["report"]).style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "Support": "{:.0f}"}, na_rep=""
            ),
            width="stretch",
        )

    # --- таблицы сравнения
    st.markdown("#### Сравнение")
    tab_history, tab_ref = st.tabs(["История экспериментов", "Оптимизированные модели (часть 3)"])

    with tab_history:
        history = pd.DataFrame(list(st.session_state["lab4_history"].values()))
        history = history[history["Датасет"] == key].drop(columns="Датасет")
        metric_cols = [col for col in history.columns if col in
                       ("Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)", "ROC-AUC")]
        st.caption("Каждая новая комбинация настроек добавляется в таблицу. Лучшие значения подсвечены.")
        st.dataframe(
            history.style.format({col: "{:.4f}" for col in metric_cols} | {"Время обучения, с": "{:.2f}"}, na_rep="")
            .highlight_max(subset=metric_cols, color="#c6efce"),
            width="stretch", hide_index=True,
        )
        if st.button("Очистить историю"):
            st.session_state["lab4_history"] = {}
            st.rerun()

    with tab_ref:
        if ref_table is None:
            st.info("Файл с результатами части 3 не найден. Выполните ноутбук части 3.")
        else:
            st.caption("Результаты оптимизированных моделей на тесте (GridSearchCV / RandomizedSearchCV).")
            st.dataframe(
                ref_table.style.format("{:.4f}", na_rep="").highlight_max(axis=0, color="#c6efce"),
                width="stretch",
            )


# ---------------------------------------------------------------------------
# Навигация
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Навигация")
    part = st.selectbox("Часть работы", list(PARTS))

if PARTS[part] is None:
    render_part4()
else:
    folder, notebook = PARTS[part]
    notebook_path = LAB4_DIR / folder / notebook
    if notebook_path.exists():
        render_notebook(notebook_path)
    else:
        st.warning("Notebook не найден")
