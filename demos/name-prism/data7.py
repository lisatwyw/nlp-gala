import os, sys
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters
import streamlit.components.v1 as components  # needed for html
from st_pages import Page, show_pages, add_page_title  # allow multipages

import numpy as np
import pandas as pd

from pathlib import Path
from glob import glob 

import plotly.express as px

parent_dir = str( Path(__file__).parents[0] )
gparent_dir= str( Path(__file__).parents[1] )
data_dir = parent_dir +  '/data/' + 'n-database' 

st.set_page_config(layout="wide")
# st.write(gparent_dir)

# ============================== read data ==============================

with open( Path( data_dir, 'majority_of_lengths_of_lastname.html') ,'r') as f: 
    html_data = f.read()

# ================================  widget ================================ 

st.title("n-database: lastnames and nationality")
st.text('Do some countries have longer lastnames than the rest?')
components.html(html_data, scrolling=True, height=800 ) 

st.header("Classification labels")

mkd='''


| Cohort ID | White* | Black | Hispanic | Asian	| Other  | Unknown |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| validated_names | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:  |  |
| ```rethnicity``` |  :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark:  |  |  |
| BCM | Not visible minority | | |  | | Prefer not tell  | 
| SPK | | | | | |
| ... | | | | | |

Subcategories to be confirmed:
- Black 
    - African-American, ...
- Indian, Inuk, Inuit, Indigenous Peoples, Chipewyan, ... 
- *Caucasian 
    - Nordic, Jew, Jewish, ...   
- Hispanic
    - Latin, ...

## Evaluation metrics

- F1-macro, PPV, FPV

'''

st.markdown(mkd)
