
# Catalog

## EMER Complain Gout

- Replace ```<username>``` with your username
```
wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/emer-complaint-gout/1.0/
```

```
import pandas as pd
df= pd.read_csv('physionet.org/files/emer-complaint-gout/1.0/GOUT-CC-2019-CORPUS-REDACTED.csv' )
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

## ICU Infection ZiGong Fourth

```
df= pd.read_csv('icu-infection-zigong-fourth/DataTables/dtNursingChart.csv' )
```

```
df.head(50)
Out[15]:
    Unnamed: 0  INP_NO temperature  heart_rate  ...                                       NURSING_DESC  Endotracheal_intubation_Depth  Extubation     ChartTime
0            1  963835        36.3        80.0  ...                                                NaN                            NaN       False -21976.916667
1            2  963835         NaN        82.0  ...  病人于09：15急诊以“昏迷待诊”收入我科，平车推入，唇甲发绀，呼吸20次/分，血氧饱和度9...                            NaN       False -21976.916667
2            3  963835         NaN        89.0  ...                                          头孢替唑皮试阴性。                            NaN       False -21976.916667
3            4  963835         NaN       122.0  ...                                                NaN                            NaN       False -21976.916667
4            5  963835         NaN        95.0  ...            病人血压值报告医生后，予NS45ml加硝酸甘油25mg静脉微泵注入2ml/h。                            NaN       False -21976.916667
5            6  963835        36.5        93.0  ...                                                NaN                            NaN       False -21976.916667
6            7  963835         NaN        98.0  ...                                                NaN                            NaN       False -21976.916667
7            8  963835         NaN        78.0  ...                                                NaN                            NaN       False -21976.916667
8            9  963835         NaN        87.0  ...                                                NaN                            NaN       False -21976.916667
9           10  963835        36.3        96.0  ...  心电监护示：节律整齐。予以硝酸甘油静脉微泵注入2ml/h。病人右手中指、环指及右下肢缺如，右...                            NaN       False -21976.916667
```


