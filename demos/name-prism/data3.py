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
    df = pd.read_excel( Path( data_dir, 'F00003844-WVS_Time_Series_List_of_Variables_and_equivalences_1981_2022_v3_1.xlsx'  ) )
    st.text( filepath )
except:
    pass
st.dataframe( df )
