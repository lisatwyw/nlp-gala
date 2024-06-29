import streamlit as st
import pandas as pd
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path
from glob import glob 

import plotly.express as px

import os, sys
import numpy as np

parent_dir = str( Path(__file__).parents[0] )
gparent_dir= str( Path(__file__).parents[1] )
data_dir = gparent_dir +  '/data/' 


st.set_page_config(layout="wide")
st.write(__file__[:-3])

city_name = os.path.basename(__file__)[:-3].capitalize()

# ============================== read data ==============================
filepath = gparent_dir + '/data/Popular_Baby_Names.csv'

dat = pd.read_csv( Path(filepath ))

st.dataframe( dat ) 
