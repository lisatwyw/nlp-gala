import streamlit as st
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path
import pandas as pd
import os, sys
import plotly.express as px

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print( current_dir )
print( parent_dir )

parent_dir = str( Path(__file__).parents[0] )
gparent_dir= str( Path(__file__).parents[1] )
print( parent_dir )
print( gparent_dir )

sys.path.append(current_dir)
sys.path.append(parent_dir)


st.set_page_config(layout="wide")

# ================================ Widgets =============================== 
show_pages(
    [
        Page(Path( current_dir, "app.py"), "Home", "🏠", "home"),
        Page(Path( current_dir, "data1.py"), "Popular baby names in US",  ),
        Page(Path( current_dir, "data2.py"), "Data from Crabtree et al.",  ),
        Page(Path( current_dir, "data3.py"), "Data dict WVS", ),
        
    ]
)

if 1:
    f = parent_dir + '/readme.md'
    mkd = Path( f ).read_text()
    st.markdown( mkd )

    st.markdown( '# A possible distribution')
    df = pd.DataFrame( dict( Ethnicity = ['White', 'Chinese','Other',  'South Asian', 'First Nations', 
                                          'Metis' , 'Latin America/Hispanic', 'Filipino', 'Southeast Asian',
                                          'Japanese', 'West Asian', 'Black', 'Korean', 'Arab', 'Inuit' ], 
                            N = [301563,19071,12768,9892,6301,6384,4722,4389,2136,2726,2072,1807,1324,1005,197],
                            Percentage=[76.46,4.84,3.24,2.51,1.6,1.62,1.2,1.11,.54,.69,.53,0.45,.34,.25,.05,] ))
    st.dataframe( df )
     
