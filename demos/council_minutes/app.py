import streamlit as st
from st_pages import Page, show_pages, add_page_title  # allow multipages

# ================================ Widgets =============================== 
st.title( 'Civil discourse' )

show_pages(
    [
        Page(Path( utils.parent_dir, "main.py"), "Civil discourse", "🏠", "Civil discourse"),
        Page(Path( utils.parent_dir, "Burnaby.py"), "Burnaby"),       
    ]
)
