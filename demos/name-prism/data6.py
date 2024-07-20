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

components.html(html_data, scrolling=True, height=800 ) 
st.text('To examine a group more closely, click on its label.')



st.header("Notes")
mkd='''

## About BCM
> Purpose: Several non-pharmaceutical interventions, such as physical distancing, handwashing, self-isolation, and school and business closures, were implemented in British Columbia (BC) following the first laboratory-confirmed case of COVID-19 on 26 January 2020, to minimise in-person contacts that could spread infections. The BC COVID-19 Population Mixing Patterns Survey (BC-Mix) was established as a surveillance system to measure behaviour and contact patterns in BC over time to inform the timing of the easing/re-imposition of control measures. In this paper, we describe the BC-Mix survey design and the demographic characteristics of respondents.
>
> Participants: The ongoing repeated online survey was launched in September 2020. Participants are mainly recruited through social media platforms (including Instagram, Facebook, YouTube, WhatsApp). A follow-up survey is sent to participants 2–4 weeks after completing the baseline survey. Survey responses are weighted to BC’s population by age, sex, geography and ethnicity to obtain generalisable estimates. Additional indices such as the Material and Social Deprivation Index, residential instability, economic dependency, and others are generated using census and location data.
>
> Findings to date: As of 26 July 2021, over 61 000 baseline survey responses were received of which 41 375 were eligible for analysis. Of the eligible participants, about 60% consented to follow-up and about 27% provided their personal health numbers for linkage with healthcare databases. Approximately 83.5% of respondents were female, 58.7% were 55 years or older, 87.5% identified as white and 45.9% had at least a university degree. After weighting, approximately 50% were female, 39% were 55 years or older, 65% identified as white and 50% had at least a university degree.


## Classification labels

| Cohort ID | White* | Black | Hispanic | Asian	| Other  | Unknown |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| validated_names | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:  |  |
| ```rethnicity``` |  :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark:  |  |  |
| BCM | Not visible minority | | |  | | Prefer not tell  | 
| Crabtree et al. |  :heavy_check_mark:| :heavy_check_mark: |  Latin | AAPI** | Other, Indigenous |
| SPK | | | | | |
| ... | | | | | |

Subcategories to be confirmed:
- ```Asian American and Pacific Islander``` 
    - "strangely broad" [source](https://www.npr.org/transcripts/1126642816)
    - ``linguistic, religious, socioeconomic diversity'' [ibid]

- Black 
    - African-American, ...
- Indian, Inuk, Inuit, Indigenous Peoples, Chipewyan, ... 
- *Caucasian 
    - Nordic, Jew, Jewish, ...   
- Hispanic
    - Latin, ...


## Evaluation metrics

- F1-macro, PPV, FPV, AUC

'''

st.markdown(mkd)
