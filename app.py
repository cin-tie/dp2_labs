import streamlit as st

st.set_page_config(
    page_title="Лабораторные работы ДП2",
    layout="wide",
)

pages = {
    "Главная": "pages/home.py",
    "Лабораторная работа №1": "pages/lab1.py",
}

page_list = []
for title, path in pages.items():
    page_list.append(st.Page(path, title=title, default=(title == "Главная")))

page = st.navigation(page_list)
page.run()