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

# _August+13+2021_17.15
filepath = parent_dir + '/data/govt-10-immigrants.csv'    
try:
    df = pd.read_csv( Path(filepath ))
    st.text( filepath )
except:
    pass

# _August+13+2021_17.15
filepath = parent_dir + '/data/govt-10-omnibus.csv'

try:
    df2 = pd.read_csv( Path(filepath ))
    st.text( filepath )
except:
    pass


filepath = parent_dir + '/data/study123.csv'
st.text( filepath )
df = pd.read_csv( Path(filepath), on_bad_lines='skip' )
# df = pd.read_excel( Path(filepath), index_col=0 )


mkd = '''
## Ethnicities in this dataset
- Data source: Crabtree et al.
- Size of dataset:
'''

st.markdown( mkd )
st.write( df.shape )

st.dataframe( df )

st.text(df.columns  ) 

tabs= st.tabs( ['Filter', 'Summary'] )


with tabs[1]:
    fig = px.histogram( df, x = 'speciic.race', y='education',  title=f'Race vs education' )
    fig.update_layout(hovermode="x unified")    
    st.plotly_chart( fig )


with tabs[0]:
    try:
        S = df.columns 
        st.text(S) 
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
    
    
    try:
      st.write( 'Summary (of filtered subset):')
      st.write( new_df.Ethnicity.describe()  )
      st.write( new_df['Child\'s First Name'].describe() )
    except:
      pass
