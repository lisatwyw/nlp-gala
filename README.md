# nlp-gala

## Option 1: Servers

Replace ```yourusername``` with your's
- https://sfu.syzygy.ca/jupyter/user/yourusername/lab
- https://ubc.syzygy.ca/jupyter/user/yourusername/lab

## Option 2: Digital Research Alliance 

### Login links

| Login | Max memory (MB) | Max run time | GRU(s)? |
| :-- | :-- | :-- | :-- |
| https://jupyterhub.cedar.computecanada.ca/ |  63000 | | |
| https://jupyterhub.beluga.computecanada.ca/ | 47750 | | | 
| https://jupyterhub.narval.computecanada.ca/ | 80000 | 8 | 1x A700 |  

### Allocations	
- 004 -- without resource application - narval-storage → 1 TB Project Storage
- 003 -- without resource application - beluga-storage → 1 TB Project Storage
- 002 -- without resource application - graham-storage → 1 TB Project Storage
- 001 -- without resource application - cedar-storage → 1 TB Project Storage

### Getting started on clusters

Once logged in:
```
module load python/
python3 -m venv tf2.15
source ~/tf2.15/bin/activate

pip3 install tensorflow torch
pip3 install polars pandas # database

pip3 install lifelines scikit-survival # survival data analyses

pip3 transformers # NLP: sentence-transformer
pip3 install -U space[cuda122] spacytextblob nltk vader # NLP

pip3 install seaborn matplotlib  plotly
pip3 install pandas
```

When in ipython or python:
```
import sys, os
if 'narval' in os.environ['CC_CLUSTER']:
    os.chdir('/lustre06/project/6088123/myusername/opensource/physionet.org/files/mimic-iv-note/2.2/note') # replace myusername with yours 
elif 'beluga' in os.environ['CC_CLUSTER']:
    os.chdir('/home/myusername/datasets/mimic-iv-note/2.2/note') # replace myusername with yours
```
