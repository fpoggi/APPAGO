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


def getFeatures(song, retryLimit=3, sleepTime=0.3):
  token = util.prompt_for_user_token(
    username=conf.username,
    scope=conf.scope,
    client_id=conf.client_id,
    client_secret=conf.client_secret,
    redirect_uri=conf.redirect_uri)
  
  retry = 0
  while token and retry < retryLimit:
    try:
      sp = spotipy.Spotify(auth=token)
      songsInfo = sp.search(f"track:{song['trackName']} artist:{song['artistName']}", type="track", limit=1)
      print(song)
      if len(songsInfo["tracks"]["items"]) != 1:
        print("ERROR in getFeatures(): len res != 1. Exit.")
        continue
      time.sleep(sleepTime)
      songInfo = songsInfo["tracks"]["items"][0]
      songFeatures = sp.audio_features(songInfo["id"])[0]
      # sarebbe meglio mettere una condizione sul risultato
      break
      #print(trackID)
      #print(song_features)
    except Exception as e:
      print (f"getFeatures()- Retry #{retry} - song: {song['trackName']}")
      print(e)
      retry += 1
      time.sleep(5)
    
    if retry >= retryLimit:
      print (f"ERROR: rgetFeatures() retry limit {retryLimit} - song: {song['trackName']}")
      sys.exit()
    
  return {"features": songFeatures, "info": songInfo}
    


contents = glob(f"{json_input_folder}*.json")
contents.sort()
for json_file in contents:
  print (json_file)
  with open(json_file) as f:
    res = "trackName;artistName;endTime;msPlayed;TrackID;Popularity;msDuration;Release_day;Uri;Acousticness;Danceability;Energy;Speechiness;Instrumentalness;Liveness;Valence;Loudeness;Tempo;Time_signature;Key;Mode;uri2;id2\n"
    songs = json.load(f)
    counter=0
    for song in songs:
      counter += 1
      print(f"{counter}/{len(songs)}")
      #print(song)
      temp = getFeatures(song)
      info = temp["info"]
      features = temp["features"]
      res += f"{song['trackName']};{song['artistName']};{song['endTime']};{song['msPlayed']};{features['id']}\{info['popularity']};{features['duration_ms']};{info['album']['release_date']};{info['uri']};{features['acousticness']};{features['danceability']};{features['energy']};{features['speechiness']};{features['instrumentalness']};{features['liveness']};{features['valence']};{features['loudness']};{features['tempo']};{features['time_signature']};{features['key']};{features['mode']};{info['uri']};{features['id']}\n"
    if not os.path.exists(csv_output_folder):
      os.makedirs(csv_output_folder)
    with open(f"{csv_output_folder}/{song.replace('.json','.csv')}", 'w') as f:
      f.write(results)
