import streamlit as st
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path

# ================================ Widgets =============================== 
st.title( 'Civil discourse' )

show_pages(
    [
        Page(Path( "main.py"), "Civil discourse", "🏠", "Civil discourse"),
        Page(Path( "Burnaby.py"), "Burnaby"),       
    ]
)
