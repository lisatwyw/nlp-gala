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

# ================================ Widgets =============================== 
show_pages(
    [
        Page(Path( current_dir, "app.py"), "Home", "🏠", "home"),
        Page(Path( current_dir, "data1.py"), "Popular baby names in US",  ),
        Page(Path( current_dir, "data2.py"), "Data from Crabtree et al.",  ),
        Page(Path( current_dir, "data3.py"), "Data dict WVS", ),
        
    ]
)


if 0:
    f = parent_dir + '/readme.md'
    mkd = Path( f ).read_text()
    st.markdown( mkd )

filepath = parent_dir + '/data/Popular_Baby_Names.csv'
df = pd.read_csv( Path(filepath))

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

