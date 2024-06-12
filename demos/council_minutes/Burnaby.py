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
filepath = gparent_dir + '/council_minutes/output/bby.csv'
dat = pd.read_csv( Path(filepath ))

# ============================== widgets ==============================
st.header( 'Prelim. results' )
fig = px.bar( dat, x = 'date', hover_data ='alc_contexts', y='alc_counts', title=f'Alcohol - Number of mentions in {city_name}' )
fig.update_layout(hovermode="x unified")

fig.update_xaxes(showspikes=True, spikemode="across")
fig.update_yaxes(showspikes=True, spikemode="across")

st.plotly_chart( fig)

mkd = '''
Hover over the counts to see each reference of the word ```alcohol``` 
'''
st.markdown( mkd )


st.header( 'Dataframe created by the backend' )
st.dataframe(dat)
