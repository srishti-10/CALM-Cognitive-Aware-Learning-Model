import streamlit as st
from typing import Optional

def get_doc_content(uploaded_file) -> Optional[str]:
    """
    Extracts and returns the text content from an uploaded .txt or .docx file.
    Returns None if the file is not supported or not provided.
    """
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".docx"):
        try:
            from docx import Document
        except ImportError:
            st.error("python-docx is required for .docx file support. Please install it.")
            return None
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        st.warning("Unsupported file type. Please upload a .txt or .docx file.")
        return None 

def is_new_file_uploaded(uploaded_file):
    # You can use file name and size as simple heuristic
    if uploaded_file is None:
        return False
    prev_file_info = st.session_state.get("uploaded_file_info")
    current_file_info = (uploaded_file.name, uploaded_file.size)
    if prev_file_info != current_file_info:
        st.session_state["uploaded_file_info"] = current_file_info
        return True
    return False