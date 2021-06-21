import pandas as pd 
from glob import glob

csv_folder = "../listening-history/csv/"
min_msPlayed = 10000


contents = glob(f"{csv_folder}*.csv")
contents.sort()
for filename in contents:
  print (filename)
  df = pd.read_csv(filename,delimiter=";")  
  numLines = df.shape[0]
  df["endTime"] = pd.to_datetime(df["endTime"], format="%Y-%m-%d %M:%S")
  df = df.sort_values(by=["endTime"])
  df_filtered = df[df["msPlayed"] > min_msPlayed]
  print(f"File: {filename} - Size: {df_filtered.shape[0]}/{numLines}")
"""
Dig Down (Acoustic Gospel Version)
Muse
2020-03-22 11:51
36165
0Tjw5aLMwzCki7ADdLwddL
46
237186
2018-11-09
spotify:track:0Tjw5aLMwzCki7ADdLwddL
0.817
0.522
0.475
0.0291
3.29e-06
0.172
0.202
-7.249
4
0
1
237187
spotify:track:0Tjw5aLMwzCki7ADdLwddL
0Tjw5aLMwzCki7ADdLwddL
"""
