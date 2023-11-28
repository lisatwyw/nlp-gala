import nltk

nltk.download('punkt')

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 


# LEALLA model
import torch
from transformers import BertModel, BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-base")
model = BertModel.from_pretrained("setu4993/LEALLA-base")
model = model.eval()
