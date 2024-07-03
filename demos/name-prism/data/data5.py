Hugging Face's logo
Hugging Face
Search models, datasets, users...
Models
Datasets
Spaces
Posts
Docs
Solutions
Pricing



Spaces:

lisatwyw
/
name-prism


like
0

Logs
App
Files
Community
1
Settings
name-prism
/
app.py

lisatwyw's picture
lisatwyw
Update app.py
67a6bd4
VERIFIED
4 days ago
raw
history
blame
edit
delete
No virus

1.01 kB
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
st.title('Hello world!')

tabs = st.tabs( ['first name', 'middle name', 'last name' ] )

@st.cache_data
def load_pbs():
    df_f = pd.read_csv('data/first_nameRaceProbs.csv')
    df_m = pd.read_csv('data/middle_nameRaceProbs.csv')
    df_l = pd.read_csv('data/last_nameRaceProbs.csv')
    return df_f, df_m, df_l

df_f, df_m, df_l = load_pbs()

with tabs[0]:
    st.dataframe(df_f)
with tabs[1]:
    st.dataframe(df_m)
with tabs[2]:
    st.dataframe(df_l)


