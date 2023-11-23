import torch, random
import pandas as pd
import polars as pol
import numpy as np
import subprocess, sys, os
import tensorflow as tf
from time import time

def environ():
  try:
    INTERACTIVE = os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive'
    package_dir = '../working/mypackages'
  except:
    if 'ipykernel' in os.environ['MPLBACKEND']:
      INTERACTIVE = 2
      try:
        os.mkdir('/content/drive/MyDrive/R4G-2023-11-20/mypackages')
      except:
        pass
      package_dir = '/content/drive/MyDrive/R4G-2023-11-20/mypackages'
      folder = '/content/drive/MyDrive/R4G-2023-11-20/'
    return folder, package_dir, INTERACTIVE

folder, package_dir, INTERACTIVE = environ()
gpus = tf.config.list_physical_devices('GPU')

import multiprocessing as mp
ncpus = mp.cpu_count()

def add_path( package_dir ):
  sys.path.append( package_dir )

def install_packages( cmds, INTERACTIVE=False, package_dir=None ):
    if package_dir is not None:
        add_path( package_dir )
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    try:
        if package_dir is not None:
            sys.mkdir( package_dir )
    except:
        pass

    for cmd in cmds:
        if package_dir is not None:
            cmd +=' --target=' + package_dir
        print( 'Running:\n\t', cmd )
        cmd = cmd.split(' ')
        subprocess.run(cmd, shell=False)

    if package_dir is not None:
        sys.path.append( package_dir )
    return gpus

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
print( '\n\n- torch, tf, os, sys, subprocess loaded \n- np, pol, pd, time, loaded' )
print( '\n\nngpus = install_packages() \n\nseed_everything(1119)' )
