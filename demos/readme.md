
# I/O

- Read as a list, one element per line 
```
with open( filename ) as fd:
  lines = fd.readlines()
```

- Read one char on each line
```
fd = open(filename, encoding='utf8')
lines = fd.read()  
fd.close()
```

Find indices to all instances of `{` in a string

```
inds = reduce(lambda x, y: x + [y.start()], re.finditer('{', astring ), [])
```

