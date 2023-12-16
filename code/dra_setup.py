

try:
  INTERACTIVE = os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive'
except:
  if 'narval' in os.environ['CC_CLUSTER']:
    loc = ''
  elif 'beluga' in os.environ['CC_CLUSTER']:
    loc = ''

