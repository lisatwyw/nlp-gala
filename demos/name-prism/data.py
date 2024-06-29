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


st.write( df.Ethnicity.describe()  )
try:
  st.write( df['Child\'s First Name'].describe() )
except:
  pass



import streamlit as st
import polars as pol
import plotly.express as px
import sys, os
from st_pages import Page, show_pages, add_page_title  # allow multipages
import utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from streamlit_dynamic_filters import DynamicFilters

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


# ================================ config ================================ 
st.set_page_config(layout="wide") 


# ================================  add paths ================================ 
sys.path.append(utils.gparent_dir)
sys.path.append(utils.parent_dir)

data_dir = utils.gparent_dir + '/data/'

# ================================ read data =============================== 

rated_tweets = utils.get_rated_tweets()
# st.text( utils.regions); # EUR, CAN, USA

st.header('European tweets')
D1 = rated_tweets[ 'eur']
D1=D1[ D1['sentiment_confidence'] > 0.66 ]


def color_red_column(col):
    return ['color: red' for _ in col]

def color_backgroubd_red_column(col):
    return ['background: red' for _ in col]


try:
    S = df.columns 
    dynamic_filters = DynamicFilters(df,
                                     filters=S, 
                                     )
    with st.sidebar:
        st.write("Apply filters in any order")
    dynamic_filters.display_filters(location='sidebar')
    dynamic_filters.display_df()    
except Exception as e:
    st.text( e )

 
