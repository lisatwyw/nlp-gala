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


fig = px.histogram( df, x = 'Ethnicity',  title=f'{1}' )
fig.update_layout(hovermode="x unified")

#fig.update_xaxes(showspikes=True, spikemode="across")
#fig.update_yaxes(showspikes=True, spikemode="across")

st.plotly_chart( fig )

st.dataframe( df ) 

st.write( df['Child''s First Name'].describe() )
st.write( df.Ethnicity.describe()  )
