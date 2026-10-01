import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from utils.notebook import render_notebook

st.title("Лабораторная работа №3")
st.markdown("### Линейная регрессия: парная и множественная")

st.markdown(
    """
     **Цель работы:** Изучение математических основ и практическое применение
    моделей парной и множественной линейной регрессии с использованием
    библиотеки scikit-learn, оценка качества моделей с помощью метрик
    эффективности, а также проведение экспериментов с различными
    комбинациями признаков.
    """
)


st.divider()

BASE_DIR = Path(__file__).resolve().parent.parent
LAB3_DIR = BASE_DIR / "labs" / "lab3"
MODELS_DIR = LAB3_DIR / "part3" / "models"
DATASETS_DIR = LAB3_DIR / "part1" / "datasets"

with st.sidebar:
    st.header("Навигация")

    part = st.selectbox(
        "Часть работы",
        ["Часть 2. Моделирование", "Часть 3. Симулятор предсказаний"],
    )


@st.cache_resource
def load_bundle(path: Path):
    return joblib.load(path)


@st.cache_data
def load_cars() -> pd.DataFrame:
    return pd.read_csv(DATASETS_DIR / "CarPrice_Assignment.csv")


def show_metrics(metrics: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (test)", f"{metrics['MAE']:,.0f}")
    c2.metric("MSE (test)", f"{metrics['MSE']:,.0f}")
    c3.metric("R² (test)", f"{metrics['R2']:.3f}")


def salary_simulator() -> None:
    path = MODELS_DIR / "salary_model.joblib"
    if not path.exists():
        st.warning("Модель не найдена. Запустите notebook лабораторной работы.")
        return

    bundle = load_bundle(path)
    model = bundle["model"]
    data = bundle["data"]
    lo, hi = bundle["ranges"]["YearsExperience"]

    years = st.slider("Стаж, лет", 0.0, 20.0, float(round((lo + hi) / 2, 1)), 0.1)
    pred = float(model.predict(pd.DataFrame({"YearsExperience": [years]}))[0])

    st.metric("Прогноз зарплаты", f"{pred:,.0f}")
    st.caption(
        f"Salary = {model.coef_[0]:,.2f} * YearsExperience + {model.intercept_:,.2f}"
    )
    if years < lo or years > hi:
        st.info(
            f"Значение вне диапазона обучающих данных ({lo}–{hi} лет): это экстраполяция."
        )

    xs = pd.DataFrame({"YearsExperience": [0.0, 20.0]})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(data["YearsExperience"], data["Salary"], alpha=0.7, label="Данные")
    ax.plot(
        xs["YearsExperience"], model.predict(xs), color="red", label="Линия регрессии"
    )
    ax.scatter([years], [pred], color="green", s=140, zorder=5, label="Ваш прогноз")
    ax.set_xlabel("Стаж, лет")
    ax.set_ylabel("Зарплата")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    show_metrics(bundle["metrics"])


def car_simulator() -> None:
    path = MODELS_DIR / "car_models.joblib"
    if not path.exists():
        st.warning("Модели не найдены. Запустите notebook лабораторной работы.")
        return

    bundles = load_bundle(path)
    name = st.selectbox("Модель (эксперимент)", list(bundles.keys()), index=1)
    bundle = bundles[name]
    model, features = bundle["model"], bundle["features"]

    st.markdown("**Параметры автомобиля**")
    values = {}
    cols = st.columns(2)
    for i, f in enumerate(features):
        lo, hi = bundle["ranges"][f]
        mean = bundle["means"][f]
        is_int = (
            float(lo).is_integer()
            and float(hi).is_integer()
            and f not in ("boreratio", "stroke", "compressionratio")
        )
        with cols[i % 2]:
            if is_int:
                values[f] = st.slider(f, int(lo), int(hi), int(round(mean)))
            else:
                step = 0.01 if hi - lo < 10 else 0.1
                values[f] = st.slider(
                    f, float(lo), float(hi), float(round(mean, 2)), step
                )

    row = pd.DataFrame([values], columns=features)
    pred = float(model.predict(row)[0])
    st.metric("Прогноз цены", f"{pred:,.0f}")

    show_f = (
        st.selectbox("Признак для графика", features)
        if len(features) > 1
        else features[0]
    )
    lo, hi = bundle["ranges"][show_f]
    grid = pd.concat([row] * 100, ignore_index=True)
    grid[show_f] = [lo + (hi - lo) * k / 99 for k in range(100)]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    cars = load_cars()
    ax.scatter(cars[show_f], cars["price"], alpha=0.4, color="blue", label="Данные")

    ax.plot(grid[show_f], model.predict(grid), color="red", label="Прогноз модели")
    ax.scatter(
        [values[show_f]], [pred], color="green", s=140, zorder=5, label="Ваш прогноз"
    )
    ax.set_xlabel(show_f)
    ax.set_ylabel("price")
    ax.grid(True)
    ax.legend()
    if len(features) > 1:
        ax.set_title("Остальные признаки зафиксированы на выбранных значениях")
    st.pyplot(fig)
    plt.close(fig)

    show_metrics(bundle["metrics"])
    with st.expander("Коэффициенты модели"):
        coefs = pd.Series(model.coef_, index=features, name="Коэффициент")
        st.write(f"Intercept: {model.intercept_:,.2f}")
        st.dataframe(coefs)


if part == "Часть 2. Моделирование":
    notebook_path = LAB3_DIR / "part2" / "linear_regression_lab3.ipynb"
    if notebook_path.exists():
        render_notebook(notebook_path)
    else:
        st.warning("Notebook не найден")

else:
    st.subheader("Интерактивный симулятор предсказаний")
    tab_salary, tab_cars = st.tabs(["Зарплата (парная)", "Цена авто (множественная)"])
    with tab_salary:
        salary_simulator()
    with tab_cars:
        car_simulator()
