# Regular expressions ```re```

Examples of regular expressions

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

## Anchor tags

```
^T
```
- Must start with capitalized ```T```

```
$?
```
- Must end with ```?```

The anchor tags ^ and $ will match text at the start of a string and at the end of a string, respectively.


## Range

```
h{1,2}t 
```
- Will match ```hot``` but not ```hooot```


```
[r-t]at
```
- Will match ```rat```, ```sat```, ```tat```, but not ```bat```


## Negation

```
[^cdh]are
```

- Will not match ```care```, '''dare``` but will match ```mare```

## Metacharacter

- ```+``` will match the preceding character 1 or more times 
- ```*``` will match the preceding character 0 or more times

```o+hh``` will match ```oooooohh``` but not ```hh```

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
- Will NOT match: ```I love snakes!``` because there's only 1 whte space between t ```\w+``` and ```\w+!```


