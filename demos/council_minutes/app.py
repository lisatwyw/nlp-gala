import streamlit as st
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path

# ================================ Widgets =============================== 
st.title( 'Civil discourse' )

parent_dir = str( Path(__file__).parents[0] )
gparent_dir= str( Path(__file__).parents[1] )
st.write(gparent_dir)

show_pages(
    [
        Page(Path( "main.py"), "Civil discourse", "🏠", "Civil discourse"),
        Page(Path( "Burnaby.py"), "Burnaby"),       
    ]
)
