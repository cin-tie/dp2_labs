import streamlit as st
from pathlib import Path
from utils.notebook import render_notebook

st.title("Лабораторная работа №2")
st.markdown("### Разведочный анализ данных")

st.markdown(
    """
    **Цель работы:** Изучение основных методов разведочного 
    анализа данных, выявление скрытых закономерностей, поиск 
    аномалий, визуализация распределений признаков и подготовка 
    данных к последующему моделированию.
    """
)

st.divider()

BASE_DIR = Path(__file__).resolve().parent.parent
LAB2_DIR = BASE_DIR / "labs" / "lab2"

with st.sidebar:
    st.header("Навигация")

    part = st.selectbox(
        "Часть работы",
        ["Часть 2. Разведочный анализ"],
    )

if part == "Часть 2. Разведочный анализ":
    notebook_path = LAB2_DIR / "part2" / "dp2_lab2_part2.ipynb"
    if notebook_path.exists():
        render_notebook(notebook_path)
    else:
        st.warning("Notebook не найден")