
# Catalog


## 

- Replace ```<username>``` with your username
```
wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/emer-complaint-gout/1.0/
```

```
import pandas as pd
df= pd.read_csv('opensource/physionet.org/files/emer-complaint-gout/1.0/GOUT-CC-2019-CORPUS-REDACTED.csv' )
```

```
In [6]: df.head(10)

                                     Chief Complaint Predict Consensus
0  "been feeling bad" last 2 weeks & switched BP ...       N         -
1  "can't walk", reports onset at <<TIME>>. orien...       Y         N
2  "dehydration" Chest hurts, hips hurt, cramps P...       Y         Y
3  "gout flare up" L arm swelling x 1 week. denie...       Y         Y
4  "heart racing,"dyspnea, and orthopnea that has...       N         -
5  "I started breathing hard"  hx- htn, gout, anx...       N         N
6  "I think I have a gout flare up" L wrist pain ...       Y         Y
7  "I want to see if I have an infection" pt vagu...       Y         N
8  "My gout done flared up on me", c/o R ankle, L...       Y         Y
9  "my gout is hurting me"- reports bilateral foo...       Y         Y

```
