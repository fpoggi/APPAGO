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
#import ntpath

json_input_folder = "../listening-history/json/new/"
csv_output_folder = "../listening-history/csv/new/"
json_features_file = "features.json"


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

def getFeatures(song, token, retryLimit=2, sleepTime=0.3):
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



# filtro canzoni uniche da cercare => toSearchSongs
contents = glob(f"{json_input_folder}/*.json")
contents.sort()
toSearchSongs = list()
toSearchSeeds = list()
C1 = 0
#song_seeds = list()
for json_file in contents:
  print (json_file)
  with open(json_file) as f:
    res = "trackName;artistName;endTime;msPlayed;TrackID;Popularity;msDuration;Release_day;Uri;Acousticness;Danceability;Energy;Speechiness;Instrumentalness;Liveness;Valence;Loudeness;Tempo;Time_signature;Key;Mode;uri2;id2\n"
    songs = json.load(f)
    counter=1
    C1 += len(songs)
    for song in songs:
      seed = f"{song['artistName']}-_-{song['trackName']}"
      if seed not in toSearchSeeds:
        toSearchSeeds.append(seed)
        toSearchSongs.append(song)

print(f"Canzoni totali: {C1} - canzoni uniche (senza rip.): {len(toSearchSeeds)}")


# scarico info di ogni song unica in toSearchSongs =>
searchResults = dict()
token = util.prompt_for_user_token(
  username=conf.username,
  scope=conf.scope,
  client_id=conf.client_id,
  client_secret=conf.client_secret,
  redirect_uri=conf.redirect_uri)
counter = 1
for song in toSearchSongs:
  if counter % 500 == 0:
    with open(json_features_file, "w") as f:
      json.dump(searchResults, f, ensure_ascii=False, indent = 2)
  #if counter > 20:
  #  break
  print(f"{counter}/{len(toSearchSongs)}")
  res = getFeatures(song, token, retryLimit=3, sleepTime=0.3)
  if res is not None:
    searchResults[res['seed']] = res
  counter += 1
with open(json_features_file, "w") as f:
  json.dump(searchResults, f, ensure_ascii=False, indent = 2)

  
"""
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
