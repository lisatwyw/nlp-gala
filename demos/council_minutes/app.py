import streamlit as st
import pandas as pd
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path
from glob import glob 

import numpy as np

parent_dir = str( Path(__file__).parents[0] )
gparent_dir= str( Path(__file__).parents[1] )
data_dir = gparent_dir +  '/data/' 

dat = pd.read_csv( gparent_dir + 'output/bby.csv')
st.dataframe(dat)


