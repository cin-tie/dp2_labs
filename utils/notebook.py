import base64
from pathlib import Path
import nbformat
import streamlit

def render_notebook(path: str | Path) -> None:
    path = Path(path)

    # Notebook path
    if not path.exists():
        streamlit.error(f"Notebook не найден")
        return

    # Чтение файла
    try:
        with path.open("r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
    except Exception as e:
        streamlit.error(f"Ошибка: {e}")
        return

    # Обработка ячеек
    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", "")
        
        if isinstance(source, list):
            source = "".join(source)

        # Markdown
        if cell_type == "markdown":
            streamlit.markdown(source)

        # Code
        elif cell_type == "code":
            with streamlit.expander("Code", expanded=True):
                streamlit.code(source, language="python")

            # Code output
            for output in cell.get("outputs", []):
                output_type = output.get("output_type", "")

                # Text
                if output_type == "stream":
                    text = output.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    if text.strip():
                        with streamlit.container(border=True):
                            streamlit.text(text)

                # Error
                elif output_type == "error":
                    traceback = output.get("traceback", [])
                    if isinstance(traceback, list):
                        traceback = "\n".join(traceback)
                    with streamlit.container(border=True):
                        streamlit.error(traceback)

                # Graphic
                elif output_type in ("execute_result", "display_data"):
                    data = output.get("data", {})

                    # PNG
                    if "image/png" in data:
                        img = data["image/png"]
                        if isinstance(img, list):
                            img = "".join(img)
                        img = img.replace("\n", "")
                        try:
                            streamlit.image(base64.b64decode(img), width="stretch")
                        except:
                            pass

                    # JPEG
                    elif "image/jpeg" in data:
                        img = data["image/jpeg"]
                        if isinstance(img, list):
                            img = "".join(img)
                        img = img.replace("\n", "")
                        try:
                            streamlit.image(base64.b64decode(img), width="stretch")
                        except:
                            pass

                    # HTML
                    elif "text/html" in data:
                        html = data["text/html"]
                        if isinstance(html, list):
                            html = "".join(html)
                        streamlit.html(html)

                    # Text
                    elif "text/plain" in data:
                        text = data["text/plain"]
                        if isinstance(text, list):
                            text = "".join(text)
                        if text.strip():
                            with streamlit.container(border=True):
                                streamlit.text(text)