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
@st.cache_data
def get():
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
    return df
    
df2 = get()


mkd = '''
## Ethnicities in this dataset
- Data source: Crabtree et al.
- Size of dataset:
'''
st.markdown( mkd )
st.write( df2.shape )
st.text( df2.columns  ) 




tabs= st.tabs( [ 'Summary','Data'] )

with tabs[0]:
    for c in df2.columns:
        try:
            if c != 'id':
                fig = px.histogram( df2, x = c, title=c.capitalize() )
                fig.update_layout(hovermode="x unified")    
                st.plotly_chart( fig )
        except:
            pass
     

with tabs[1]:
    st.header('All')
    st.dataframe( df2[ ['name','first','last','res.race', 'res.male', 'res.age', 'res.edu', 'res.income'] ] )
    
    st.header('Indigenous only')    
    df2 = df2[ df2['res.race'] == 'Indigenous' ] 
    st.dataframe( df2 )

    if 0:
        try:
            S = ['res.male', 'res.age', 'res.edu', 'res.income' ]
            dynamic_filters2 = DynamicFilters(df2,
                                             filters=S, 
                                             )
            dynamic_filters2.display_filters()
            dynamic_filters2.display_df()   
            new_df2 = dynamic_filters2.filter_df()
            st.write( 'Summary (of filtered subset):')
            st.write( new_df2['res.age'].describe()  )
            st.write( new_df2['res.edu'].describe() )
        except:
            pass
 
