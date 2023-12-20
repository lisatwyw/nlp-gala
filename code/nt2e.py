#!/usr/bin/env python # 
"""\
Usage: nt2e.py 
"""
__author__ = "Lisa"
__copyright__ = "Copyright 2023, R4G"
__credits__ = ["Lisa"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Lisa"
__status__ = "Test"

# project specific
import neiss 

import torch
import random
import numpy as np

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import subprocess, sys, os
from time import time
from datetime import date
from pathlib import Path
import multiprocessing as mp

# database
import pandas as pd
import polars as pol

# i/o
import json, pickle

# word embedding
import tensorflow as tf
# emb=4
import tensorflow_hub as hub
# emb=1,2,3
try:
    from sentence_transformers import SentenceTransformer
except:
    pass  
# emb=6
from torch.multiprocessing import Pool, Process, set_start_method    
from transformers import BertModel, BertTokenizerFast

import umap, optuna  

# model dev and eval
import scipy
import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, log_loss

# SDA
from sksurv.metrics import concordance_index_censored, brier_score
from sksurv.util import Surv    

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    #print('random, torch, tf, os packages seeded')

seed_everything(1119)

      

data_dir, inter_dir, res_dir, proj_dir = neiss.neiss_setup()
os.chdir( proj_dir )
try:
    os.mkdir( res_dir )
except:
    pass    




def rid_typos( r ):
    r=r.replace('TWO','2').replace('TWPO','2').replace('TW D','2 D')
    r=r.replace('FIOUR DAYS AGO', 'FOUR DAYS AGO').replace('6X DAYS','6 DAYS').replace('4 FDAYS','4 DAYS')
    r=r.replace('FOOUR DAYS','4 DAYS').replace('SIX','6').replace('SEVEN','7')
    r=r.replace('THREE','3').replace('ONE','1').replace('FOUR','4').replace('FIVE','5').replace(')','')
    r=r.replace('TEN','10').replace('NINE','9').replace('24 HOURS AGO','1 DAY AGO').replace('EIGHT','8')
    r=r.replace('E DAYS AGO','8').replace('A DAY','1 DAY')
    r=r.replace('ELEVEN','11').replace('TWELVE','12').replace('HRS', 'HOURS')
    r=r.replace('HR AGO', 'HOUR AGO').replace(' F DAYS AGO', ' FEW DAYS AGO')
    r=r.replace('SEVERALDAYS', 'SEVERAL DAYS').replace('2 AND A HALF','2.5').replace('2 AND HALF','2.5').replace('5 DADAYS','5 DAYS')
    r=r.replace('2HOURS','2 HOURS').replace('1HOUR','1 HOUR').replace('AN HOUR','1 HOUR').replace('FERW HOUR','FEW HOUR')
    r=r.replace('2,DAYS', '2 DAYS').replace('SEV DAYS', 'SEVERAL DAYS')
    r=r.replace('COUPLEOF','COUPLE OF').replace('A DAY','1 DAY').replace('HALF HOUR','1 HOUR') # round to 1 hour
    r=r.replace('   ',' ').replace('  ',' ').replace('DAYSA GO', 'DAYS AGO').replace('witho ','with')
    r=r.replace('LAST MIGHT AND', 'LAST NIGHT AND').replace('AT NH','AT NURSING HOME').replace(' BAKC ', ' BACK ')
    r=r.replace('DXZ','DX').replace('10NIS', 'TENNIS').replace('N/S INJURY', 'NOT SIGNIFICANT INJURY')
    r=r.replace('***','*').replace('**','*').replace('>>>','>').replace('>>','>').replace('...','.').replace('..','.')
    r=r.replace('&','and').replace("@", "at").replace('+', ' ').replace('--','. ')
    r=r.replace('VERTABRA','VERTEBRA').replace('+LOCDX','LOC DX').replace(' ONTPO ', ' ONTO ').replace(' STAT ES ', ' STATES ')
    return r


def strip_basic_info( r ):
    for a in ['YOF', 'YO FEMALE', 'Y/O FEMALE', 'YO F', 'YF', 'Y O F', 'YOM', ' YWF',
              'YO MALE', 'YO M', 'Y O M', 'YM', 'Y/O WM',
              'Y/O MALE' , 'Y/O M', 'OLD FE', 'OLD MALE ', 'FEMALE', 'MALE']:
        try:
            r = r.split(a[:10])[1]
            break
        except:
            pass
    if r.find('DX')>-1:
        parts=r.split('DX')
    elif r.find('>')>-1:
        parts=r.split('>')
    else:
        parts=r.split('*')
    try:
        dx = parts[1]
    except:
        dx = '' # assumed not narrated
    return parts[0], dx

def lemmatizer(narrative:str) -> str:
    doc = nlp(narrative)
    p = doc._.blob.polarity
    s = doc._.blob.subjectivity
    return " ".join([token.lemma_ for token in doc ]), p, s


def _clean_narrative( text0 ):    # Step 1) rid of typos
    text = rid_typos( text0 )

    # Step 2) strip off demo graphic info + diagnosis
    text, dx = strip_basic_info( text )
    dx = dx.replace('*',' ').replace('LBP','low back pain')

    # Step 3)
    text += ' ' # add char space to end so that step 5 will not ignore words immediately followed by period

    # Step 4) lowercase and add char spaces so not interprettted as spelling errors
    text = text.lower().replace(',',' ').replace('.',' .')

    # Ack: https://www.drivendata.org/competitions/217/cdc-fall-narratives/community-code/50/
    #
    # Changes need: special words are capitalized "CLOSED" "LEFT" to prevent lemmatization
    #
    # Step 5) map abbrevations back to English words;
    for term, replacement in abbr_terms.items():
        text = text.replace( f' {term} ', f' {replacement} ' )

    text, p, s = lemmatizer( text )

    return text, dx, p, s

abbr_terms = {
      "abd": "abdomen", # "af": "accidental fall",
      "afib": "atrial fibrillation",
      "aki": "acute kidney injury",
      "am": "morning", #"a.m.": "morning",
      "ams": "ALTERED mental status",
      "bac": "BLOOD ALCOHOL CONTENT",
      "bal": "BLOOD ALCOHOL LEVEL",
      "biba": "brought in by ambulance",
      "c/o": "complains of",
      "chi": "CLOSED head injury",
      "clsd hd": "CLOSED head",
      "clsd": "CLOSED",
      "cpk": "creatine phosphokinase",
      "cva": "cerebral vascular accident", # stroke
      "dx": "diagnosis",
      "ecf": "EXTENDED CARE FACILITY",
      "elf": "EXTENDED CARE FACILITY",
      "er": "emergency room",
      "ed": "emergency room",
      "etoh": "ETHYL ALCOHOL",
      "eval": "evaluate", # "fd": "fall detected",
      "fxs": "fractures",  # "glf": "ground level fall",
      "fx": "fracture",
      "h/o": "history of", # "htn": "hypertension",
      "hx": "history of",
      "inj": "injury",
      "inr": "INR",  # "international normalized ratio": special type of measurement; keep abbv to retain clinical meaning ==> capitalize
      "intox": "intoxication",
      "lac": "lacerations",
      "loc": "loss of consciousness", # capitalize so that "left" will be ignored by lemmatizer, rather than convert to "leave" (present tense of "left")
      "mech": "mechanical",
      "mult": "multiple",
      "n.h.": "NURSING HOME",
      "nh": "NURSING HOME",
      "p/w": "presents with",
      "pm": "afternoon",
      "pt": "patient", # "p.m.": "afternoon",
      'prev': "previous",
      "pta": "prior to arrival",
      "pts": "patient's", # "px": "physical examination", # not "procedure", "r": "right", "l": "left",
      "r/o": "rules out",
      "rt": "right",
      "s'dandf": "slipped and fell",
      "s'd&f": "slipped and fell",
      "t'd&f": "tripped and fell", "t'dandf": "tripped and fell",
      "tr": "trauma",
      "s/p": "after",
      "rt.": "right",
      "lt.": "LEFT",
      "lt": "LEFT",
      "sah": "subarachnoid hemorrhage",
      "sdh": "acute subdural hematoma",
      "sts": "sit-to-stand",
      "uti": "UTI", # "urinary tract infection",
      "unwit'd": "unwitnessed",
      "w/": "with",
      "w": "with",
      "w/o": "without",
      "wks": "weeks"
    }

ncores = 8 # mp.cpu_count()
print(ncores)

with Path( data_dir + "variable_mapping.json").open("r") as f:
    mapping = json.load(f, parse_int=True)
for c in mapping.keys():
    mapping[c] = {int(k): v for k, v in mapping[c].items()}
try:
    if ('decoded_df' in globals() )==False:
        print('Loading cleaned and msked narratives from disk')
        decoded_df = pol.read_csv( f'{inter_dir}/decoded_df_cleaned.csv' )
        with open( f'{inter_dir}/dev_tst_split.pkl', 'rb') as handle:
            d=pickle.load( handle )
            trn_case_nums = d['trn_case_nums']
            tst_case_nums = d['tst_case_nums']        
except:
    if ('decoded_df' in globals() )==False:
        print('Loading decoded_df from disk to clean narratives')
        decoded_df, org_columns, trn_case_nums, tst_case_nums = neiss.get_data( data_dir )     
    DEMO=0
    if DEMO:  # test on small sample
        sample = decoded_df.iloc[3::3000,:].copy()
        with mp.Pool(mp.cpu_count()) as pool:
            sample_narr_cleaned = pool.map( _clean_narrative, sample['narrative'] )
        for a,b in zip(sample['narrative'], sample_narr_cleaned):
            print(a, '\n', b[0], '\n\tDX:', b[1], '\n\tpolarity:', b[2], '\n\tsubjectivity:', b[3])
    else:
        # from spellchecker import SpellChecker; spellchecker = SpellChecker()
        nlp = spacy.load( 'en_core_web_lg' )
        nlp.add_pipe('spacytextblob')

        def get_cleaned_narratives( dff ):
            with mp.Pool( ncores ) as pool:
                res =  pool.map( _clean_narrative, dff['narrative'] )
            A = [ r[0] for r in res ]
            B = [ r[1] for r in res ]
            P = [ r[2] for r in res ]
            S = [ r[3] for r in res ]
            dff['narrative_cleaned']= A
            dff['narrative_dx'] = B
            dff['narrative_pol'] = P
            dff['narrative_sub'] = S
            return dff

        #sample = get_cleaned_narratives( sample )
        #print(sample['narrative'], '\n\n', sample['narrative_cleaned'] )

        cpcs_nums = decoded_df.cpsc_case_number
        starttime=time()
        decoded_df = get_cleaned_narratives( decoded_df )
        print( ( time() - starttime)/60 )

        decoded_df.to_csv( inter_dir + '/decoded_df_cleaned.csv' )

        with open(  inter_dir + '/dev_tst_split.pkl', 'wb') as hd:
            pickle.dump( dict( trn_case_nums = trn_case_nums, tst_case_nums=tst_case_nums, cpcs_nums=cpcs_nums), hd )
            
            
            
            
def get_cohort():                      
    terms=[]
    for n in range(1,120 ):
        s,t = f'{n} minutes ago|{n} min ago|{n} min. ago', (n/60.0)
        terms.append( (s,t) )

    for n in range(1,20):
        s,t = f'{n} days ago|{n} d ago|{n} dy ago|{n}night|{n} nights ago|{n}day', (n*24)
        terms.append( (s,t) )

    s,t = r'an hour|1 hour|a hour', 1
    terms.append( (s,t) )

    s,t = r'couple h|coup. h', 3
    terms.append( (s,t) )
    s,t = r'few h', 4
    terms.append( (s,t) )
    s,t = r'sev h|sev. h|several h', 5
    terms.append( (s,t) )

    s,t = r'this even', 6
    terms.append( (s,t) )

    s,t = r'this afternoon', 8
    terms.append( (s,t) )

    for n in range(1,24):
        s,t = f'around {n}:|around {n} a|around {n} p', 8
        terms.append( (s,t) )

    for n in range(1,24):
        s,t = f'{n} hours ago|{n} h ago|{n} hrs ago', n
        terms.append( (s,t) )

    s,t = r'today|tdy|morning|this am|this pm', 12
    terms.append( (s,t) )

    s,t = r'around midnight|last ngiht|last ev|yesterday n', 16
    terms.append( (s,t) )

    s,t = r'previous day|last noc|last might|last nite|last even|last night|day before|yesterday', 24
    terms.append( (s,t) )

    s,t = r'couple d|coup. d|3 nites ago|3 nights ago', 3*24
    terms.append( (s,t) )
    s,t = r'few d|few n', 4*24
    terms.append( (s,t) )
    s,t = r'sev d|sev. d|several d', 5*24
    terms.append( (s,t) )

    s,t = r'last week|last wk|week ago', 7*24
    terms.append( (s,t) )

    s,t = r'couple w|coup. w', 3*24*7
    terms.append( (s,t) )
    s,t = r'few w', 4*7*24
    terms.append( (s,t) )
    s,t = r'sev w|sev. w|several w', 5*24*7
    terms.append( (s,t) )

    s,t = r'last mth|last month', 24*30
    terms.append( (s,t) )

    s,t = r'couple m|coup. m', 3*24*30
    terms.append( (s,t) )
    s,t = r'few m', 4*30*24
    terms.append( (s,t) )
    s,t = r'sev w|sev. m|several m|mos ago', 5*24*30
    terms.append( (s,t) )

    def get_subset( sstr, time ):
        sstr = r'%s'% sstr
        d = decoded_df.filter( pol.col('narrative_cleaned').str.contains( sstr )  )
        d = d.with_columns( pol.col("narrative_cleaned").str.replace(sstr, "###").alias('narrative_cleaned_masked') )
        d = d.with_columns( pol.lit( time*1.0 ).alias('time2event') )
        return d

    for c,t in enumerate( terms ):
        d = get_subset( t[0], t[1] )
        if c == 0:
            Big = d
        else:
            Big = pol.concat( [Big, d] )
        if d.shape[0]>0:
            print( t[0], t[1], d.shape)

    print( Big.shape[0]/ decoded_df.shape[0] , '(sample size as a percentage of entire cohort)')

    for s in Big.sample(10).iter_rows():
        print( '\n\n', s[3], '\n', s[-2], '\n',s[-1] )

    sset = Big.filter( pol.col('time2event') <1 )
    for s in sset.sample(10).iter_rows():
        print( '\n\n', s[3], '\n', s[-2], '\n',s[-1] )
    return Big

Big = get_cohort()

    

def get_pipe():
    tokenizer = BertTokenizerFast.from_pretrained(names[emb])
    model =  BertModel.from_pretrained(names[emb])
    model = model.eval()
    return tokenizer, model

def compute( inp ):
    tokenizer, model = get_pipe()
    english_inputs = tokenizer(inp, return_tensors="pt", padding=True)
    with torch.no_grad():
        english_outputs = model(**english_inputs)
    r = np.array( english_outputs.pooler_output )
    return r

names, embeddings,Embeddings = {}, {}, {}
narratives = list( Big['narrative_cleaned_masked'] )
cpsc_case_nums = Big[:,2]
names[6] = "setu4993/LEALLA-small"
names[7] = "setu4993/LEALLA-big"
names[8] = "setu4993/LEALLA-large"

def embed(model, model_type, sentences):
    """
    wrapper function for generating message embeddings
    """
    if model_type == 'use':
        embeddings = model(sentences)
    elif model_type == 'sentence transformer':
        embeddings = model.encode(sentences)
    return embeddings

n = Big.shape[0]

EMB=[1,2,3,4,6]
RDIMS = [4]

T = ['trn1', 'trn2', 'val', 'tst']

def where_inter(a,b):
    return np.arange(a.shape[0])[~np.in1d(a,b)].tolist()

trn_inds=where_inter(cpsc_case_nums, trn_case_nums )
tst_inds=where_inter(cpsc_case_nums, tst_case_nums )
Inds = {}
Inds['trn1']=trn_inds[0::3]
Inds['trn2']=trn_inds[1::3]
Inds['val']=trn_inds[2::3]
Inds['tst']=tst_inds

if 0:
    os.environ['TOKENIZERS_PARALLELISM'] = True 
    for emb in [6]: 
        starttime = time()
        if emb==1:
            model = SentenceTransformer('all-mpnet-base-v2')
        elif emb==2:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        elif emb==3:    
            model = SentenceTransformer('paraphrase-mpnet-base-v2')
        elif emb==4:
            try:
                model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')              
            except:
                model = hub.load( f'{inter_dir}/063d866c06683311b44b4992fd46003be952409c') 
            res = embed(model, 'use', narratives ).numpy()
        if emb<4:
            with mp.Pool( ncores ) as pool:            
                res= pool.map(model.encode, narratives )
            res = np.array( res )
        elif emb==6:     
            multi_pool = Pool(processes=ncores)
            predictions = multi_pool.map(compute, narratives)
            multi_pool.close()
            multi_pool.join()
            res = np.array(predictions).squeeze()    
        embeddings[emb]= res
        n,ndims = res.shape                         
        print( ndims, 'dimensions' )
        with open( f'{inter_dir}/WordEmb{emb}_n{n}_d{ndims}.pkl', 'wb') as handle:
            pickle.dump( {"embeddings": res, 'cpsc_case_nums':cpsc_case_nums} , handle)
        print( emb, -(starttime - time())/60/60 , 'hours' )

if 1:
    d = pd.read_pickle( inter_dir+f'WordEmb1_n{n}_d768.pkl' )
    embeddings[1]=d['embeddings']
    d = pd.read_pickle( inter_dir+f'WordEmb2_n{n}_d384.pkl' )
    embeddings[2]=d['embeddings']
    d = pd.read_pickle( inter_dir+f'WordEmb3_n{n}_d768.pkl' )
    embeddings[3]=d['embeddings'] 
    d = pd.read_pickle( inter_dir+f'WordEmb4_n{n}_d512.pkl' )
    embeddings[4]=d['embeddings']        
    d = pd.read_pickle( inter_dir+f'WordEmb6_n{n}_d128.pkl' )
    embeddings[6]=d['embeddings']     
    cpsc_case_nums = d['cpsc_case_nums']

    
    for t in T:
        for e in EMB:
            Embeddings[e, t]= embeddings[e][ Inds[t], : ]

        

if ('we_reduced' in globals())==False:
    we_reduced, reducers = {}, {}


# convert string (categorical) attributes to numeric
mapping_inv={}
for k in mapping.keys():
    mapping_inv[k]  = {v: k for k, v in mapping[k].items()}
att = []
for k in ['location', 'race', 'sex', 'body_part', 'product_1', 'product_2', 'drug', 'alcohol' ]:
    k2 = k+'_numeric'
    Big = Big.with_columns(
      pol.col( k )
      .map_dict( mapping_inv[k]).alias( k2 )  )
    att.append( k2)
print( Big.sample(11).to_pandas().transpose() )

att += ['month']

surv_pols = {}
for t in T:
    surv_pols[t] = Big[ Inds[t], : ]
    surv_pols[t]= surv_pols[t].with_columns([
        pol.col("narrative").str.lengths().alias("narrative_len")
    ])
    print(t, surv_pols[t].shape )
    if 0:
        fig=px.histogram( surv_pols[t].to_pandas(),'narrative_len')
        fig.show()


def get_surv( dff, DEFN = 2 ):
    ev,time,surv_inter,surv_str={},{},{},{}
    if DEFN==1:
        thres=3 # treated
    elif DEFN==2:
        thres=5 # hosp/ died
    k = 'time2event'
    for t in T:
        df=dff[t].to_pandas()
        surv_inter[t]=pd.DataFrame( {'label_lower_bound': df[k] ,
                                     'label_upper_bound': df[k] } )
        if DEFN==3:
            q=np.where(  df['severity'] < thres )[0] # unseen + observed
            ev[t] = ( df['severity'] >= thres ).values
        else:
            q=np.where(  df['severity'] < thres )[0] # unseen + observed
            ev[t] = ( df['severity'] >= thres ).values

        surv_inter[t].iloc[q, 1] = np.inf
        print(t, np.sum(ev[t])/ len(ev[t]), 'n=',len(ev[t]) )
        time[t] = ( df[k]  ).values
        surv_str[t]=Surv.from_arrays(ev[t], time[t])
    return ev, time, surv_inter, surv_str



for t in T:
    print( np.where( surv_pols[t][:,-2] <=0 ) )
for t in T:
    fig= plt.figure()
    fig = px.histogram( surv_pols[t][:,-2].to_pandas() );
    fig.show()

# ====================== optimization ======================



def run_xgb_optuna( T, emb, X, surv_inter, output_file ):
    ds = {}
    base_params = {'verbosity': 0,
                  'objective': 'survival:aft',
                  'eval_metric': 'aft-nloglik',
                  'tree_method': 'hist'}  # Hyperparameters common to all trials

    samp_choices = ['uniform']
    for t in T:
        ds[t] = xgb.DMatrix( X[t])
        print(t)
        # see details https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html
        ds[t].set_float_info('label_lower_bound', surv_inter[t]['label_lower_bound'] )
        ds[t].set_float_info('label_upper_bound', surv_inter[t]['label_upper_bound'] )

    if gpus:
        base_params.update( {'tree_method': 'gpu_hist', 'device':'cuda', } )
        samp_choices = ['gradient_based','uniform']

    def tuner(trial):
        params = {'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
                  'aft_loss_distribution': trial.suggest_categorical('aft_loss_distribution',
                                                                      ['normal', 'logistic', 'extreme']),
                  'aft_loss_distribution_scale': trial.suggest_float('aft_loss_distribution_scale', 0.1, 10.0),
                  'max_depth': trial.suggest_int('max_depth', 3, 10),
                  'booster': trial.suggest_categorical('booster',['gbtree','dart',]),
                  'scale_pos_weight': trial.suggest_float('scale_pos_weight', pos_ratio*0.1, 10*pos_ratio ),  # L1 reg on weights
                  'alpha': trial.suggest_float('alpha', 1e-8, 10 ),  # L1 reg on weights
                  'lambda': trial.suggest_float('lambda', 1e-8, 10 ),  # L2 reg on weights
                  'eta': trial.suggest_float('eta', 0, 1.0),  # step size
                  'sampling_method': trial.suggest_categorical('sampling_method', samp_choices ),
                  'subsample': trial.suggest_float('subsample', 0.01, 1 ),
                  'gamma': trial.suggest_float('gamma', 1e-8, 10)  # larger, more conservative; min loss reduction required to make leaf
        }
        params.update(base_params)
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'valid-aft-nloglik')
        bst = xgb.train(params, ds['trn1'], num_boost_round=10000,
                        evals=[(ds['trn1'], 'train'), (ds['trn2'], 'valid')],  # <---- data matrices
                        early_stopping_rounds=50, verbose_eval=False, callbacks=[pruning_callback])
        if bst.best_iteration >= 25:
            return bst.best_score
        else:
            return np.inf  # Reject models with < 25 trees

    # Run hyperparameter search
    study = optuna.create_study(direction='minimize')
    study.optimize( tuner, n_trials= n_trials )
    print('Completed hyperparameter tuning with best aft-nloglik = {}.'.format(study.best_trial.value))
    params = {}
    params.update(base_params)
    params.update(study.best_trial.params)

    print('Re-running the best trial... params = {}'.format(params))
    bst = xgb.train(params, ds['trn1'], num_boost_round=10000, verbose_eval=False,
                    evals=[(ds['trn1'], 'train'), (ds['trn2'], 'valid')],
                    early_stopping_rounds=50)

    # Explore hyperparameter search space
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()

    for t in T:
        res[t]= pd.DataFrame({'Label (lower bound)': surv_inter[t]['label_lower_bound'],
                    'Label (upper bound)': surv_inter[t]['label_upper_bound'],
                    'Predicted label': bst.predict(ds[t]) } )
        sp=scipy.stats.spearmanr( res[t].iloc[:,-2], res[t].iloc[:,-1] )
        c=concordance_index_censored( event_time = time2event[t], event_indicator = event_indicator[t] , estimate=1/res[t].iloc[:,-1] )

        print(t.upper(), f'| R2:{sp[0]:.3f}; p:{sp[1]:.4} | C:{c[0]*100:.2f} ', end='|' )

        labels = ['3h','6h','9h','12h','15h','18h','24h','48h','72h', '1wk','2wks','1mon']
        for d,h in enumerate([3,6,9,12,15,18,24,49,73, 7*24+1, 7*24*2+1, 7*24*4+1 ]):
            bs = brier_score( surv_str['trn1'], surv_str[t], estimate=1/res[t].iloc[:,-1], times=[h] )
            print( end=f'{labels[d]}:{bs[1][0]:.3f} |' )
        print()
    today = date.today()
    bst.save_model( output_file )
    return res    
