import streamlit as st
from pathlib import Path
from utils.notebook import render_notebook

st.title("Лабораторная работа №1")
st.markdown("### Настройка среды разработки и управление виртуальными окружениями")

st.markdown(
    """
    **Цель работы:** Освоение навыков развертывания изолированных сред разработки,
    управления пакетами через графический интерфейс (GUI) и терминал в Anaconda
    и Google Colaboratory, а также первичный аудит аппаратных ресурсов.
    """
)

st.divider()

BASE_DIR = Path(__file__).resolve().parent.parent
LAB1_DIR = BASE_DIR / "labs" / "lab1"

with st.sidebar:
    st.header("Навигация")
    
    part = st.selectbox(
        "Часть работы",
        ["Часть 1. Anaconda", "Часть 2. Google Colab", "Часть 3. Датасеты"]
    )
    
    if part == "Часть 3. Датасеты":
        dataset = st.selectbox(
            "Датасет",
            ["faults.csv", "WineQT.csv"]
        )

if part == "Часть 1. Anaconda":
    notebook_path = LAB1_DIR / "part1" / "lab1_anaconda.ipynb"
    if notebook_path.exists():
        render_notebook(notebook_path)
    else:
        st.warning("Notebook не найден")

elif part == "Часть 2. Google Colab":
    notebook_path = LAB1_DIR / "part2" / "lab1_colab.ipynb"
    if notebook_path.exists():
        render_notebook(notebook_path)
    else:
        st.warning("Notebook не найден")

else:
    if dataset == "faults.csv":
        notebook_path = LAB1_DIR / "part3" / "lab1_dataset1.ipynb"
    else:
        notebook_path = LAB1_DIR / "part3" / "lab1_dataset2.ipynb"
    
    if notebook_path.exists():
        render_notebook(notebook_path)
    else:
        st.warning("Notebook не найден")