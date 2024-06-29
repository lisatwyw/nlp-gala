import streamlit as st
import pandas as pd
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path
from glob import glob 

import plotly.express as px

import os, sys
import numpy as np

import sys, os
from st_pages import Page, show_pages, add_page_title  # allow multipages
from streamlit_dynamic_filters import DynamicFilters

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
) 

def color_red_column(col):
    return ['color: red' for _ in col]

def color_backgroubd_red_column(col):
    return ['background: red' for _ in col]






parent_dir = str( Path(__file__).parents[0] )
gparent_dir= str( Path(__file__).parents[1] )
data_dir = parent_dir +  '/data/' 


st.set_page_config(layout="wide")
st.write(gparent_dir)

# ============================== read data ==============================
filepath = parent_dir + '/data/Popular_Baby_Names.csv'

df = pd.read_csv( Path(filepath ))

mkd = '''
## Ethnicities in this dataset
- Data source: https://catalog.data.gov/dataset/popular-baby-names
- Size of dataset:
'''

st.markdown( mkd )
st.write( df.shape )
st.text( df.columns )

tabs= st.tabs( ['Filter', 'Summary'] )

with tabs[1]:
    fig = px.histogram( df, x = 'w.asian',  title=f'{1}' )
    fig.update_layout(hovermode="x unified")
    
    #fig.update_xaxes(showspikes=True, spikemode="across")
    #fig.update_yaxes(showspikes=True, spikemode="across")
    
    st.plotly_chart( fig )


with tabs[0]:
    try:
        S = df.columns 
        dynamic_filters = DynamicFilters(df,
                                         filters=S, 
                                         )
        with st.sidebar:
            st.write("Apply filters in any order")
        dynamic_filters.display_filters(location='sidebar')
        dynamic_filters.display_df()   
        new_df = dynamic_filters.filter_df()
    except Exception as e:
        st.text( e )
    
    st.write( 'Summary (of filtered subset):')
    st.write( new_df.Ethnicity.describe()  )
    try:
      st.write( new_df['Child\'s First Name'].describe() )
    except:
      pass

