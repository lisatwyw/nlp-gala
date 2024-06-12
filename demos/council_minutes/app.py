import streamlit as st
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ================================ Widgets =============================== 
st.title( 'Civil discourse' )

parent_dir = str( Path(__file__).parents[0] )
gparent_dir= str( Path(__file__).parents[1] )
st.write(gparent_dir)

print( gparent_dir )

show_pages(
    [
        Page(Path( current_dir, "main.py"), "Civil discourse", "🏠", "Civil discourse"),
        Page(Path( current_dir, "Burnaby.py"), "Burnaby"),       
    ]
)
