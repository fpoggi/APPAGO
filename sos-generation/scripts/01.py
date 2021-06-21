import pandas as pd 
from glob import glob
import sys

csv_folder = "../listening-history/csv/"
min_msPlayed = 10000

# Section 3.1
def computeNTNA_NTKA(df):
  #artists = df[df["date"] < "2020-03-23"]
  h = df.shape[0]
  
  sum_dNTNAi = 0
  sum_dNTKAi = 0
  for date in pd.unique(df["date"]):
    #print("\n\n\n" + date)
    df_day = df[df["date"] == date]
    # Errore - non va bene
    #artists = pd.unique(df[df["date"] < date]["artistName"])
    #dNTNAi = df_day[~df_day["artistName"].isin(artists)].shape[0]

    known_tracks = pd.unique(df[df["date"] < date]["TrackID"])
    dNTNAi = df_day[~df_day["TrackID"].isin(known_tracks)].shape[0]
    sum_dNTNAi += dNTNAi

    known_tracks = pd.unique(df[df["date"] < date]["TrackID"])
    known_artists = pd.unique(df[df["date"] < date]["artistName"])
    new_tracks = df_day[~df_day["TrackID"].isin(known_tracks)]
    dNTKAi = new_tracks[~new_tracks["artistName"].isin(known_artists)].shape[0]
    sum_dNTKAi += dNTKAi

  NTNA = 100 * sum_dNTNAi / h
  NTKA = 100 * sum_dNTKAi / h
  return {"NTNA": NTNA, "NTKA": NTKA}


# Section 3.2
def songs_byHour(df, h):
  if type(h) == str and len(h) != 2:
    print("ERROR in songs_byHour()")
    sys.exit()
  if type(h) == int:
    h = "{:02d}".format(h)
  print(h)
  temp = df[df["time-hour"] == h]
  print(temp)




#contents = glob(f"{csv_folder}*.csv")
contents = glob(csv_folder + "*.csv")

contents.sort()
for filename in contents:
  print (filename)
  df = pd.read_csv(filename,delimiter=";", encoding="UTF-8") #"ISO-8859-1")  
  numLines = df.shape[0]
  df["endTime"] = pd.to_datetime(df["endTime"], format="%Y-%m-%d %H:%M")
  df = df.sort_values(by=["endTime"])
  df_filtered = df[df["msPlayed"] > min_msPlayed]
  #print(f"File: {filename} - Size: {df_filtered.shape[0]}/{numLines}")
  print("File: " + filename + " - Size: " + str(df_filtered.shape[0]) + "/" + str(numLines))
  #print(df_filtered.head(50))
  
  #df_filtered["date"] = df_filtered["endTime"].apply(lambda x: x.date())
  temp = df_filtered["endTime"].apply(lambda x: str(x.date()))
  df_filtered.insert(loc=3,column="date",value=temp)
  temp = df_filtered["endTime"].apply(lambda x: str(x).split(" ")[1]) #str(x.hour) + ":" + str(x.minute) )
  df_filtered.insert(loc=4,column="time",value=temp)
  temp = df_filtered["endTime"].apply(lambda x: str(x).split(" ")[1].split(":")[0]) #str(x.hour) + ":" + str(x.minute) )
  df_filtered.insert(loc=5,column="time-hour",value=temp)
  #df_filtered["time-hour"]
  
  #print(df_filtered[["date","trackName"]].head(10))
  #print(df_filtered.groupby(["date","time-hour"]).count())

  songs_byHour(df_filtered, "09")
  sys.exit()
  
  ntna_ntka = computeNTNA_NTKA(df_filtered)
  print(ntna_ntka)


  
  #for index, row in df_filtered.iterrows():
  #  print(row["endTime"])
  #  print(row["endTime"].date())
  #  print(row["endTime"].hour)
  
  #break


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
