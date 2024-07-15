import streamlit as st
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path
import pandas as pd
import os, sys
import plotly.express as px

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


st.set_page_config(layout="wide")

# ================================ Widgets =============================== 
show_pages(
    [
        Page(Path( current_dir, "app.py"), "Home", "🏠", "home"),
        Page(Path( current_dir, "data1.py"), "Popular baby names in US",  ),
        Page(Path( current_dir, "data2.py"), "Crabtree",  ),
        Page(Path( current_dir, "data3.py"), "WVS", ),
        Page(Path( current_dir, "data4.py"), "SPK-R1", ),
        Page(Path( current_dir, "data5.py"), "USA2010", ),        
        Page(Path( current_dir, "data6.py"), "BCM", ),        
    ]
)

if 1:
    f = parent_dir + '/readme.md'
    mkd = Path( f ).read_text()
    st.markdown( mkd )

