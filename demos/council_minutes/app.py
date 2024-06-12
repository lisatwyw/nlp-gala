import streamlit as st
import pandas as pd

dat = pd.read_csv('output/bby.csv')
st.dataframe(dat)


