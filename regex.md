# Regular expressions ```re```


## Phone numbers
```
(|\d |\()\d\d\d(|-|.)\d\d\d(-||.)\d\d\d\d
```

Will match phone numbers, e.g. these strings:

- ✅718-555-3810
- ✅9175552849
- ✅1 212 555 3821
- ✅(917)5551298
- ✅212.555.8731

## ```?```: 0 or 1 instance of the preceeding character 

```
\d crazy sharks?
```
will match:

```1 crazy shark``` and ```3 crazy sharks```

## 

```
\w{1} \w+ \d* \w+!
```
will NOT match: ```I love snakes!``` because there's only 1 whte space between t ```\w+``` and ```\w+!```

## Negation

```
[^cdh]are
```

will not match ```care```, '''dare``` but will match ```mare```
