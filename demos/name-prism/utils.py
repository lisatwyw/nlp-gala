import streamlit as st
import pandas as pd
from st_pages import Page, show_pages, add_page_title  # allow multipages
from pathlib import Path
from glob import glob 

import plotly.express as px

import os, sys
import numpy as np

import sys, os
from st_pages import Page, show_pages, add_page_title  # allow multipages
from streamlit_dynamic_filters import DynamicFilters

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
) 

def color_red_column(col):
    return ['color: red' for _ in col]

def color_backgroubd_red_column(col):
    return ['background: red' for _ in col]




