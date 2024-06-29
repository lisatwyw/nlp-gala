import streamlit as st
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print( current_dir )
print( parent_dir )

parent_dir = str( Path(__file__).parents[0] )
gparent_dir= str( Path(__file__).parents[1] )
print( parent_dir )
print( gparent_dir )

sys.path.append(current_dir)
sys.path.append(parent_dir)

# ================================ Widgets =============================== 
show_pages(
    [
        Page(Path( current_dir, "app.py"), "Who is Who", "🏠", "Who is Who"),
        Page(Path( current_dir, "data.py"), "Sample data" ),
        
    ]
)


f = parent_dir + '/readme.md'
mkd = Path( f ).read_text()
st.markdown( mkd )
