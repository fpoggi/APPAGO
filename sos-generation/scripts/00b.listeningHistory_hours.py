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
listeningHistoryFolder = "../listening-history/csv/"
#songsetFolder_1 = "input/songset_1/"
#songsetFolder_2 = "input/songset_2/"
#resFolder = "output/" + mylib.getTimeStr() + "/"
#resValuesFilename = "results.tsv"
noRepeatedSongs = False
### NEW ###
songsetFolder = "../songset/"
min_msPlayed = 30000
min_songs_hour = 10

output_listeningHistoryFolder = "../data/input/listeningHistory/"

def doExperiment(lisHistFile, songsetFile):
  df_lisHist = pd.read_csv(lisHistFile,delimiter=";", encoding="UTF-8") #"ISO-8859-1")
  print(df_lisHist.head(1))
  sys.exit()
  

contents_lh = glob(f"{listeningHistoryFolder}Elisa_Delbue.csv")
contents_lh.sort()
for listeningHistoryFile in contents_lh:
  #df = pd.read_csv(listeningHistoryFile,delimiter=";", encoding="UTF-8")
  #print(df.head())
  df = mylib.loadData(listeningHistoryFile, min_msPlayed)
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
    #print(row["MY_PLAYED_AT"], row["MY_PLAYED_AT"].isoformat())
    timeiso = played_at_timestamp.isoformat()
    df.at[index, "played_at"] = timeiso[:-3] + "T"
    #df.at[index, "date"] = timeiso[:10].replace("-","/")
    df.at[index, "datetime"] = timeiso[11:19].replace("-","/")
    #print(timeiso, datetime)

  print(df.groupby(["time-hour"]).count())
  for hour in range(0,24):
    hour_str = "{:02d}".format(hour)
    df_hour = df[df["time-hour"] == hour_str]
    if df_hour.shape[0] >= min_songs_hour:
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
      df_final.to_csv(f"{oufput_folder}/csv_{hour}.csv", sep=";", index=False)
  
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
    """
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
###################################################
# EXPERIMENTS START HERE # EXPERIMENTS START HERE #
###################################################
res = list()

for hour in range(0,24):
  listeningHistoryFile = listeningHistoryFolder + "csv_" + str(hour) + ".csv"
  playlistSorted = getLongestPlaylist(listeningHistoryFile)
  
  ### DYNAMIC ALGORITHM - DYN-1,...,DYN-4 ###
  songSetFiles = getSongSetFiles(hour)
  for songsetType in songSetFiles.keys():
    songSetFile = songSetFiles[songsetType]
    songset = getSongSet(songSetFile)
    
    optimal_playlist = mylib.getBestPlaylistInSongSet(playlistSorted,songset,noRepeatedSongs)
    # check
    d = mylib.computePD(playlistSorted, optimal_playlist["playlist"])
    if (abs(optimal_playlist["weight"]-d) > 0.00001):
      print ("ERROR: the playlist pattern distance computed by getBestPlaylistInSongSet() is not correct.")
    print (songSetFile, optimal_playlist["weight"])
    
    temp = {"acronym": getSongSetAcronym(songsetType), "hour": "{:02d}".format(hour), "listening history length": len(playlistSorted), "songset type": songsetType + "_" + "{:02d}".format(hour), "songset file": songSetFile, "songset length": len(songset), "playlist type": "Dynamic programming algorithm", "playlist length": len(optimal_playlist["playlist"]), "PPD value": optimal_playlist["weight"], "computed playlist": optimal_playlist["playlist"]}
    res.append(temp)

  ### Playlist generated with Spotify's Recommender ###
  ### REC-1
  recommended_playlist = mylib.recommenderSingleSong(playlistSorted,temp["playlist length"],True)
  distance = mylib.computePD(playlistSorted, recommended_playlist)
  temp = {"acronym": "REC-1", "hour": "{:02d}".format(hour), "listening history length": len(playlistSorted), "songset type": "Recommender_1song_features_" + "{:02d}".format(hour), "songset file": "", "songset length": 0, "playlist type": "Spotify Recommender (1 song with features)", "playlist length": len(recommended_playlist), "PPD value": distance, "computed playlist": recommended_playlist}
  res.append(temp)
  
  #recommended_playlist = mylib.recommenderSingleSong(playlistSorted,temp["playlist length"],False)
  #distance = mylib.computePD(playlistSorted, recommended_playlist)
  #temp = {"hour": "{:02d}".format(hour), "listening history length": len(playlistSorted), "songset type": "Recommender_1song_" + "{:02d}".format(hour), "songset file": "", "songset length": 0, "playlist type": "Spotify Recommender (1 song NO features)", "playlist length": len(recommended_playlist), "PPD value": distance, "computed playlist": recommended_playlist}
  #res.append(temp)
  
  ### REC-2
  recommended_playlist_prevnext = mylib.recommenderPrevNextSong(playlistSorted,temp["playlist length"],True)
  distance = mylib.computePD(playlistSorted, recommended_playlist_prevnext)
  temp = {"acronym": "REC-2", "hour": "{:02d}".format(hour), "listening history length": len(playlistSorted), "songset type": "Recommender_prev-next_features_" + "{:02d}".format(hour), "songset file": "", "songset length": 0, "playlist type": "Spotify Recommender (prev-next with features)", "playlist length": len(recommended_playlist_prevnext), "PPD value": distance, "computed playlist": recommended_playlist_prevnext}
  res.append(temp)
  
  #recommended_playlist_prevnext = mylib.recommenderPrevNextSong(playlistSorted,temp["playlist length"],False)
  #distance = mylib.computePD(playlistSorted, recommended_playlist_prevnext)
  #temp = {"hour": "{:02d}".format(hour), "listening history length": len(playlistSorted), "songset type": "Recommender_prev-next_" + "{:02d}".format(hour), "songset file": "", "songset length": 0, "playlist type": "Spotify Recommender (prev-next NO features)", "playlist length": len(recommended_playlist_prevnext), "PPD value": distance, "computed playlist": recommended_playlist_prevnext}
  #res.append(temp)
  
  ### DYN-1
  songSetFileKM = songSetFiles["KM"]
  songsetKM = getSongSet(songSetFileKM)
  poolSize = len(songsetKM)
  
  recommended_playlist_pool = mylib.recommenderPoolSongs(playlistSorted,poolSize)
  optimal_playlist = mylib.getBestPlaylistInSongSet(playlistSorted,recommended_playlist_pool[0:poolSize],noRepeatedSongs)
  distance = mylib.computePD(playlistSorted, optimal_playlist["playlist"])
  if (abs(optimal_playlist["weight"]-distance) > 0.00001):
    print ("ERROR: the playlist pattern distance computed by getBestPlaylistInSongSet() is not correct.")
  print ("Pool " + str(hour), ", Length: " + str(len(recommended_playlist_pool)), optimal_playlist["weight"])
  temp = {"acronym": "HYB-1", "hour": "{:02d}".format(hour), "listening history length": len(playlistSorted), "songset type": "Recommender_pool_" + "{:02d}".format(hour), "songset file": "", "songset length": len(recommended_playlist_pool), "playlist type": "Spotify Recommender (pool + dynamic algorithm) - pool 1x", "playlist length": len(optimal_playlist["playlist"]), "PPD value": distance, "computed playlist": optimal_playlist["playlist"]}
  res.append(temp)

  ### DYN-2
  recommended_playlist_pool = mylib.recommenderPoolSongs(playlistSorted,2*poolSize)
  optimal_playlist = mylib.getBestPlaylistInSongSet(playlistSorted,recommended_playlist_pool[0:poolSize],noRepeatedSongs)
  distance = mylib.computePD(playlistSorted, optimal_playlist["playlist"])
  if (abs(optimal_playlist["weight"]-distance) > 0.00001):
    print ("ERROR: the playlist pattern distance computed by getBestPlaylistInSongSet() is not correct.")
  print ("Pool " + str(hour), ", Length: " + str(len(recommended_playlist_pool)), optimal_playlist["weight"])
  temp = {"acronym": "HYB-2", "hour": "{:02d}".format(hour), "listening history length": len(playlistSorted), "songset type": "Recommender_pool_" + "{:02d}".format(hour), "songset file": "", "songset length": len(recommended_playlist_pool), "playlist type": "Spotify Recommender (pool + dynamic algorithm) - pool 2x", "playlist length": len(optimal_playlist["playlist"]), "PPD value": distance, "computed playlist": optimal_playlist["playlist"]}
  res.append(temp)
  
  ### DYN-4
  recommended_playlist_pool = mylib.recommenderPoolSongs(playlistSorted,4*poolSize)
  optimal_playlist = mylib.getBestPlaylistInSongSet(playlistSorted,recommended_playlist_pool[0:poolSize],noRepeatedSongs)
  distance = mylib.computePD(playlistSorted, optimal_playlist["playlist"])
  if (abs(optimal_playlist["weight"]-distance) > 0.00001):
    print ("ERROR: the playlist pattern distance computed by getBestPlaylistInSongSet() is not correct.")
  print ("Pool " + str(hour), ", Length: " + str(len(recommended_playlist_pool)), optimal_playlist["weight"])
  temp = {"acronym": "HYB-4", "hour": "{:02d}".format(hour), "listening history length": len(playlistSorted), "songset type": "Recommender_pool_" + "{:02d}".format(hour), "songset file": "", "songset length": len(recommended_playlist_pool), "playlist type": "Spotify Recommender (pool + dynamic algorithm) - pool 4x", "playlist length": len(optimal_playlist["playlist"]), "PPD value": distance, "computed playlist": optimal_playlist["playlist"]}
  res.append(temp)


# save results to file
# check folder existence/create folder
if not os.path.exists(resFolder):
  os.makedirs(resFolder)

# save tsv with PPD values
with open(resFolder + resValuesFilename, "w") as outfile:
  outfile.write("Hour  Acronym  Listening history length  songset type  songset file  songset length  playlist type  playlist length  PPD value  playlist filename\n")
  for line in res:
    outfile.write("%s\t%s\t%d\t%s\t%s\t%d\t%s\t%d\t%s\t%s\n" % (line["hour"], line["acronym"], line["listening history length"], line["songset type"], line["songset file"], line["songset length"], line["playlist type"], line["playlist length"], str("{:.3f}".format(line["PPD value"])), line["acronym"] + "-" + line["hour"] + ".tsv") )

# save tsv with songs (optimal playlist)
for line in res:
  with open(resFolder + line["acronym"] + "-" + line["hour"] + ".tsv", "w") as outfile:
    outfile.write("track id\n")
    for song in line["computed playlist"]:
      outfile.write(str(song["trackId"]) + "\n")
"""
