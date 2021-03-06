import json 
import os
from glob import glob
import sys
import time
import conf
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util

#import pandas as pd 
import ntpath

#json_input_folder = "../listening-history/json/new/"
json_input_folder = "../data/input/listening-history-json/new/"
csv_output_folder = "../data/input/listening-history-csv/"
json_features_file = "features.json"


def getFeatures(song, retryLimit=2, sleepTime=0.3):
  token = util.prompt_for_user_token(
    username=conf.username,
    scope=conf.scope,
    client_id=conf.client_id,
    client_secret=conf.client_secret,
    redirect_uri=conf.redirect_uri)

  found=False
  retry = 0
  while token and retry < retryLimit:
    try:
      sp = spotipy.Spotify(auth=token)
      time.sleep(sleepTime)
      songsInfo = sp.search(f"track:{song['trackName']} artist:{song['artistName']}", type="track", limit=1)
      #print(song)
      if len(songsInfo["tracks"]["items"]) != 1:
        print("ERROR in getFeatures(): len res != 1. Retry.")
        retry += 1
        time.sleep(4)
        continue
      songInfo = songsInfo["tracks"]["items"][0]
      songFeatures = sp.audio_features(songInfo["id"])[0]
      # sarebbe meglio mettere una condizione sul risultato
      found=True
      break
    except Exception as e:
      print (f"getFeatures()- Retry #{retry} - song: {song['trackName']}")
      print(e)
      retry += 1
      time.sleep(4)
    
    #if retry >= retryLimit:
    #  print (f"ERROR: getFeatures() retry limit {retryLimit} - song: {song['trackName']}")
    #  sys.exit()
    
  if found:
    return {"seed": f"{song['artistName']}-_-{song['trackName']}", "song": song, "features": songFeatures, "info": songInfo} #"artistName": song["artistName"], "trackName": song["trackName"], 
  else:
    return None


"""
# filtro canzoni uniche da cercare => toSearchSongs
contents = glob(f"{json_input_folder}/*.json")
contents.sort()
toSearchSongs = list()
toSearchSeeds = list()
C1 = 0
for json_file in contents:
  print (json_file)
  with open(json_file) as f:
    songs = json.load(f)
    counter=1
    C1 += len(songs)
    for song in songs:
      seed = f"{song['artistName']}-_-{song['trackName']}"
      if seed not in toSearchSeeds:
        toSearchSeeds.append(seed)
        toSearchSongs.append(song)
print(f"Canzoni totali: {C1} - canzoni uniche (senza rip.): {len(toSearchSeeds)}")


# load saved searches from file
if os.path.isfile(json_features_file):
  with open(json_features_file) as f:
    searchResults = json.load(f)
else:
  searchResults = dict()

#c = 0
#for k in searchResults:
#  if searchResults[k] is None:
#    c+=1
#    #print(k)
#print(c)
#sys.exit()

# scarico info di ogni song unica in toSearchSongs =>
counter = 0
for song in toSearchSongs:
  
  counter += 1
  print(f"{counter}/{len(toSearchSongs)}")

  # skip song if already downloaded (i.e. in saved searches) 
  seed = f"{song['artistName']}-_-{song['trackName']}"
  if seed in searchResults:
    print("found")
    continue
  
  # sometimes...
  if counter % 500 == 0:
    # ...save results and
    with open(json_features_file, "w") as f:
      json.dump(searchResults, f, ensure_ascii=False, indent = 2)

  res = getFeatures(song, retryLimit=3, sleepTime=0.3)
  searchResults[seed] = res
    
  
# save results to file
with open(json_features_file, "w") as f:
  json.dump(searchResults, f, ensure_ascii=False, indent = 2)
"""


# load saved searches from file
if os.path.isfile(json_features_file):
  with open(json_features_file) as f:
    searchResults = json.load(f)
else:
  searchResults = dict()

contents = glob(f"{json_input_folder}/*.json")
contents.sort()
for json_file in contents:
  results = "trackName\tartistName\tendTime\tmsPlayed\tTrackID\tPopularity\tmsDuration\tRelease_day\tUri\tAcousticness\tDanceability\tEnergy\tSpeechiness\tInstrumentalness\tLiveness\tValence\tLoudeness\tTempo\tTime_signature\tKey\tMode\turi2\tid2\n"
  print (json_file)
  with open(json_file) as f:
    songs = json.load(f)
    for song in songs:
      seed = f"{song['artistName']}-_-{song['trackName']}"
      match = searchResults[seed]
      if match is not None and match["features"] is not None:
        #song = match["song"]
        info = match["info"]
        features = match["features"]
        #print(features["id"])
        #print(info["id"])
        #sys.exit()
        try:
          results += f"{song['trackName']}\t{song['artistName']}\t{song['endTime']}\t{song['msPlayed']}\t{features['id']}\t{info['popularity']}\t{features['duration_ms']}\t{info['album']['release_date']}\t{info['uri']}\t{features['acousticness']}\t{features['danceability']}\t{features['energy']}\t{features['speechiness']}\t{features['instrumentalness']}\t{features['liveness']}\t{features['valence']}\t{features['loudness']}\t{features['tempo']}\t{features['time_signature']}\t{features['key']}\t{features['mode']}\t{info['uri']}\t{features['id']}\n"
        except Exception as e:
          print(e)
          print(match)
          sys.exit()
      else:
        print("None!")
      
    if not os.path.exists(csv_output_folder):
      os.makedirs(csv_output_folder)
    baseFilename = ntpath.basename(json_file).replace(".json",".csv")
    with open(f"{csv_output_folder}/{baseFilename}", 'w') as f:
      f.write(results)









"""
def getFeatures_old(song, retryLimit=3, sleepTime=0.3):
  token = util.prompt_for_user_token(
    username=conf.username,
    scope=conf.scope,
    client_id=conf.client_id,
    client_secret=conf.client_secret,
    redirect_uri=conf.redirect_uri)
  
  found=False
  retry = 0
  while token and retry < retryLimit:
    try:
      sp = spotipy.Spotify(auth=token)
      songsInfo = sp.search(f"track:{song['trackName']} artist:{song['artistName']}", type="track", limit=1)
      #print(song)
      if len(songsInfo["tracks"]["items"]) != 1:
        print("ERROR in getFeatures(): len res != 1. Exit.")
        retry += 1
        time.sleep(5)
        continue
      time.sleep(sleepTime)
      songInfo = songsInfo["tracks"]["items"][0]
      songFeatures = sp.audio_features(songInfo["id"])[0]
      # sarebbe meglio mettere una condizione sul risultato
      found=True
      break
    except Exception as e:
      print (f"getFeatures()- Retry #{retry} - song: {song['trackName']}")
      print(e)
      retry += 1
      time.sleep(5)
    
    if retry >= retryLimit:
      print (f"ERROR: rgetFeatures() retry limit {retryLimit} - song: {song['trackName']}")
      sys.exit()
    
  if found:
    return {"features": songFeatures, "info": songInfo}
  else:
    return None
"""


"""
res = "trackName;artistName;endTime;msPlayed;TrackID;Popularity;msDuration;Release_day;Uri;Acousticness;Danceability;Energy;Speechiness;Instrumentalness;Liveness;Valence;Loudeness;Tempo;Time_signature;Key;Mode;uri2;id2\n"
    

    for song in songs:
      #seed = f"{song['artistName']}-{song['trackName']}"
      #if seed in song_seeds:
      #  continue
      #else:
      #  song_seeds.append(seed)
      
      print(f"{counter}/{len(songs)}")
      
      temp = getFeatures_old(song)
      
      if temp is not None:        
        info = temp["info"]
        features = temp["features"]
        res += f"{song['trackName']};{song['artistName']};{song['endTime']};{song['msPlayed']};{features['id']}\{info['popularity']};{features['duration_ms']};{info['album']['release_date']};{info['uri']};{features['acousticness']};{features['danceability']};{features['energy']};{features['speechiness']};{features['instrumentalness']};{features['liveness']};{features['valence']};{features['loudness']};{features['tempo']};{features['time_signature']};{features['key']};{features['mode']};{info['uri']};{features['id']}\n"
      counter += 1
    if not os.path.exists(csv_output_folder):
      os.makedirs(csv_output_folder)
    with open(f"{csv_output_folder}/{song.json_file('.json','.csv')}", 'w') as f:
      f.write(results)
"""
