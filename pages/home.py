from pathlib import Path
import streamlit


BASE_DIR = Path(__file__).resolve().parent.parent
LABS_DIR = BASE_DIR / "labs"


streamlit.title("Лабораторные работы ДП2")

streamlit.markdown(
    """
    Коллекция лабораторных работ по ДП2

    Собрано:

    - исходные Jupyter Notebook
    - результаты выполнения
    - датасеты
    - графики
    - описание работ
    """
)

streamlit.divider()

streamlit.subheader("Лабораторные работы")


labs = [
    path
    for path in LABS_DIR.iterdir()
    if path.is_dir() and path.name.startswith("lab")
]

count_labs = len(labs)


datasets = [
    file
    for file in LABS_DIR.glob("lab*/part*/datasets/*.csv")
    if file.is_file()
]

count_datasets = len(datasets)


col1, col2 = streamlit.columns(2)

with col1:
    streamlit.metric(
        "Лабораторные",
        count_labs,
    )

with col2:
    streamlit.metric(
        "Датасеты",
        count_datasets,
    )
