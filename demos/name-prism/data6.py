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
data_dir = parent_dir +  '/data/' 

st.set_page_config(layout="wide")
# st.write(gparent_dir)

# ============================== read data ==============================

with open( Path( data_dir, 'vm_age_sex_distr.html') ,'r') as f: 
    html_data = f.read()

# ================================  widget ================================ 

st.title("BCM: about the data")
st.header("Age and sex distribution in visible minority")

components.html(html_data, scrolling=True, height=900 ) 
st.text('To examine a group more closely, click on its label.')

st.header("Classification labels")

mkd='''


| Cohort ID | White* | Black | Hispanic | Asian	| Other  |
| :-- | :-- | :-- | :-- | :-- | :-- |
| validated_names | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:  | 
| rethnicity |  :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark:  |  | 
| BCM | Not visible minority | | |  | Prefer not tell  |
| SPK | | | | |
| ... | | | | |

Other:
- Indigenous Peoples, African-American, Chipewyan 
- *Caucasian/ Nordic
- Indian, Inuk, Inuit, Jew, Jewish, Latin, etc.

## Evaluation metrics

- F1-macro, PPV, FPV

'''

st.markdown(mkd)
