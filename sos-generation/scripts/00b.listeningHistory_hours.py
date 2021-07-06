import csv
import sys
from datetime import datetime
import math
import glob
import os
from glob import glob
import ntpath
import pandas as pd
import os

import mylib

# Params
#listeningHistoryFolder = "../listening-history/csv/"
listeningHistoryFolder = "../data/input/listening-history-csv/"
#songsetFolder_1 = "input/songset_1/"
#songsetFolder_2 = "input/songset_2/"
#resFolder = "output/" + mylib.getTimeStr() + "/"
#resValuesFilename = "results.tsv"
noRepeatedSongs = False
### NEW ###
#songsetFolder = "../songset/"
min_msPlayed = 30000
min_songs_hour = 10

output_listeningHistoryFolder = "../data/process/listeningHistory-hours/"

def doExperiment(lisHistFile, songsetFile):
  df_lisHist = pd.read_csv(lisHistFile,delimiter=";", encoding="UTF-8") #"ISO-8859-1")
  print(df_lisHist.head(1))
  sys.exit()
  

contents_lh = glob(f"{listeningHistoryFolder}*.csv")
contents_lh.sort()
for listeningHistoryFile in contents_lh:
  #df = pd.read_csv(listeningHistoryFile,delimiter=";", encoding="UTF-8")
  #print(df.head())
  df = mylib.loadData(listeningHistoryFile, min_msPlayed, delimiter="\t")
  df.sort_values(by=["date","time","msPlayed"], inplace=True)
  
  # Aggiungo una colonna con l'orario giusto
  #df["MY_PLAYED_AT"] = [pd.Timestamp('2017-01-01T12') for i in range(0,df.shape[0])]
  df["played_at"] = [" " for i in range(0,df.shape[0])]
  df["datetime"] = [" " for i in range(0,df.shape[0])]
  for index, row in df.iterrows():
    # vado allo scadere del minuto e sottraggo i secondi
    ##df.at[index, "MY_PLAYED_AT"] = row["endTime"] + pd.Timedelta(seconds=(59-row["msPlayed"]//1000))
    #df.at[index, "MY_PLAYED_AT"] = row["endTime"] + pd.Timedelta(seconds=59) - pd.Timedelta(milliseconds=row["msPlayed"])
    played_at_timestamp = row["endTime"] + pd.Timedelta(seconds=59) - pd.Timedelta(milliseconds=row["msPlayed"])
    
    # Problema: non stampa secondi e millisecondi se sono tutti 0
    #timeiso = played_at_timestamp.isoformat()
    #df.at[index, "played_at"] = timeiso[:-3] + "Z"
    #df.at[index, "datetime"] = timeiso[11:19].replace("-","/")
    
    timeiso = (played_at_timestamp.to_pydatetime()).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    print(timeiso)
    df.at[index, "played_at"] = timeiso[:-4] + "Z"
    df.at[index, "datetime"] = timeiso[11:19]
    
    
  print(df.groupby(["time-hour"]).count())
  for hour in range(0,24):
    hour_str = "{:02d}".format(hour)
    df_hour = df[df["time-hour"] == hour_str]
    # Faccio copia, rimuovo duplicati: se #righe < min_songs_hour => skip (perché non ho songset corrispondente ottenuto con clustering
    #if df_hour.shape[0] >= min_songs_hour:
    df_copy = df_hour.copy(deep=True)
    df_copy.drop_duplicates(subset ="TrackID", keep = "first", inplace = True)
    if df_copy.shape[0] >= min_songs_hour:
      print(hour)
      #df_copy = df_hour.copy(deep=True)
      df_final = pd.DataFrame()
      #track_id	track_name	duration_ms	album_name	album_type	artist_name	track_popularity	explicit	artist_genres	album_genres	acousticness	danceability	energy	instrumentalness	key	liveness	loudness	mode	speechness	tempo	time_signature	valence	played_at	date	datetime
      df_final["track_id"] = df_hour["TrackID"]
      df_final["track_name"] = df_hour["trackName"]
      df_final["duration_ms"] = ""
      df_final["album_name"] = ""
      df_final["album_type"] = ""
      df_final["artist_name"] = df_hour["artistName"]
      df_final["track_popularity"] = df_hour["Popularity"]
      df_final["explicit"] = ""
      df_final["artist_genres"] = ""
      df_final["album_genres"] = ""
      df_final["acousticness"] = df_hour["Acousticness"]
      df_final["danceability"] = df_hour["Danceability"]
      df_final["energy"] = df_hour["Energy"]
      df_final["instrumentalness"] = df_hour["Instrumentalness"]
      df_final["key"] = df_hour["Key"]
      df_final["liveness"] = df_hour["Liveness"]
      df_final["loudness"] = df_hour["Loudeness"]
      df_final["mode"] = df_hour["Mode"]
      df_final["speechness"] = df_hour["Speechiness"]
      df_final["tempo"] = df_hour["Tempo"]
      df_final["time_signature"] = df_hour["Time_signature"]
      df_final["valence"] = df_hour["Valence"]
      df_final["played_at"] = df_hour["played_at"]
      df_final["date"] = df_hour["date"]
      df_final["datetime"] = df_hour["datetime"]
      
      baseFilename = ntpath.basename(listeningHistoryFile).replace(".csv","")
      oufput_folder = f"{output_listeningHistoryFolder}/{baseFilename}/"
      
      if not os.path.exists(oufput_folder):
        os.makedirs(oufput_folder)
      df_final.to_csv(f"{oufput_folder}/csv_{hour_str}.tsv", sep="\t", index=False)

"""
  #df.groupby(["date","time"]).count().to_csv("pippo.csv")
  dates = df["date"].unique()
  for date in dates:
    #print(date)
    df_date = df[df["date"] == date]
    times = df_date["time"].unique()
    for time in times:
      df_times = df_date[df_date["time"] == time]
      if df_times.shape[0] > 1:
        print(df_times[["trackName","endTime","msPlayed"]])
        #df_times["pippo"] = df_times.shape[0]
        for index, row in df_times.iterrows():
          msPlayed = row["msPlayed"]
          #msPlayed = df.loc[index]["msPlayed"]
          #print(df.loc[index]["endTime"], msPlayed)
          #print(msPlayed//1000)
          # vado allo scadere del minuto e sottraggo i secondi
          df.loc[index]["MY_PLAYED_AT"] = (row["endTime"] + pd.Timedelta(seconds=(59-msPlayed//1000)))
          
    #sys.exit()
  df.to_csv(listeningHistoryFile.replace(".csv","_POGGI.csv"), sep=";", index=False)
  sys.exit()
  
  #songsetFile = listeningHistoryFile
  baseFilename = ntpath.basename(listeningHistoryFile).replace(".csv","")
  songsetFolder = f"{songsetFolder}/{baseFilename}/"
  contents_ss = glob(f"{songsetFolder}/*.tsv")
  contents_ss.sort()
  #print(songsetFolder)
  for songsetFile in contents_ss:
    print(songsetFile)
    doExperiment(listeningHistoryFile, songsetFile)
"""
