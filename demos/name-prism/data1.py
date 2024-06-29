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
    fig = px.histogram( df, x = 'Ethnicity',  title=f'{1}' )
    fig.update_layout(hovermode="x unified")
    
    st.plotly_chart( fig )


with tabs[0]:
    try:
        S = df.columns 
        dynamic_filters1 = DynamicFilters(df,
                                         filters=S, 
                                         )
        dynamic_filters1.display_filters(location='sidebar',)  # or columns, num_columns=2 sidebar, or None
        dynamic_filters1.display_df()   
        
        new_df1 = dynamic_filters1.filter_df()
        st.write( 'Summary (of filtered subset):')
        st.write( new_df1.Ethnicity.describe()  )
        st.write( new_df1['Child\'s First Name'].describe() )
    except Exception as e:
        st.text( e )
    
     

