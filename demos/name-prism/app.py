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
        Page(Path( current_dir, "data2.py"), "Data from Crabtree et al.",  ),
        Page(Path( current_dir, "data3.py"), "Data dict WVS", ),
        
    ]
)

if 1:
    f = parent_dir + '/readme.md'
    mkd = Path( f ).read_text()
    st.markdown( mkd )

    mkd = '''
    | Ethnicity             | N (%)|
    |:--|:--|
    |First Nations	|    6301 (1.60)|
    |Metis	        |    6384 (1.62)|
    |Inuit	                    |197 (0.05)|
    |White (European descent)	|301,563 (76.46)|
    |Chinese	|19,071 (4.84)|
    |South Asian (eg East Indian, Pakistani, Sri Lankan)|	9892 (2.51)|
    |Black (eg African or Caribbean)|	1807 (0.45)|
    |Filipino|	4389 (1.11)|
    |Latin American / Hispanic|	4722 (1.20)|
    |Southeast Asian (eg Vietnamese, Cambodian, Malaysian, Laotian)	|2136 (0.54)|
    |Arab	|1005 (0.25)|
    |West Asian (eg Iranian, Afghan)|	2072 (0.53)|
    |Korean|	1324 (0.34)|
    |Japanese|	2726 (0.69)|
    |Other|	12,768 (3.24)|
    |Prefer not to answer	|6682 (0.69)|
    '''
    st.markdown( mkd )
