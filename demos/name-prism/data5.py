import streamlit as st
import pandas as pd
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path
from glob import glob 

import plotly.express as px

import os, sys
import numpy as np

import sys, os
from streamlit_dynamic_filters import DynamicFilters



parent_dir = str( Path(__file__).parents[0] )
gparent_dir= str( Path(__file__).parents[1] )
data_dir = parent_dir +  '/data/' 

st.set_page_config(layout="wide")
st.write(gparent_dir)

# ============================== read data ==============================
try:
    filepath = Path( data_dir, 'USA_rare_lastnames_2010.txt')
    df = pd.read_csv( filepath, sep ='\t' )    
except Exception as e:
    st.text(e)

st.dataframe( df )
