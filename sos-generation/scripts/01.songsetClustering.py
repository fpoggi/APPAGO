import json 
import pandas as pd 
import ntpath
import os

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
#sklearn.neighbors.NearestCentroid
import numpy as np
import math
from scipy.special import gammainc

from scipy.spatial import distance

from glob import glob
import sys

import time
import conf
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util

import mylib

recommender_sleepTime = 0.5

csv_folder = "../data/input/listening-history-csv/" #"../listening-history/csv/"
output_folder = "../data/process/songset/" #"../songset/"
minSongsHour = 10
minMsPlayed = 30000
#maxClusters = 10
#num_tracks = 100
maxClusters = 10
numTracks = 100



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
  df_copy = df.copy(deep=True)
  if type(h) == str and len(h) != 2:
    print("ERROR in songs_byHour()")
    sys.exit()
  if type(h) == int:
    h = "{:02d}".format(h)
  return df_copy[df_copy["time-hour"] == h]
  


"""
def centroid(l,M):
  c = [ ]
  # per ogni componente = colonna d M
  for j in range(0,len(M[0])):
    v = 0
    #calcolo la media sulle righe di M
    for i in range(0, len(l)):
      v = (v + M[l[i]][j])
         
    v = v / len(l)    
    c.append(v)
  return c
"""

#Section 3.3
# calcola l'indice Davis-Boudin. Input cc = centroidi; cl = clustering come indici di riga di M;
# M = matrice dati. Restituisce indice numerico
def DaviesBouldinIndex(cc, cl, M):
  S = [ ]
  # per ogni cluster i calcolo i valori S
  for i in range(0,len(cl)):
    
    s = 0
    c = cl[i]
    # distanze di ogni elemento j dal centroide
    for j in range(0,len(c)):
      s = s + distance.euclidean(cc[i],M[c[j]])
    s = s / len(c)
    S.append(s)

  # calcolo D_i per ogni  cluster 
  D = [ ]
    
  # per ogni coppia di cluster (i,j) con (i != j) calcolo i R_i,j e prendo il max
  # non e' ottimizzato, ricalcolo gli stessi valori piu' volte
  # per ogni cluster i 
  for i in range (0,len(cl)):
    d = 0
    # per ogni altro cluster j
    for j in range (0,len(cl)):
      if (i != j):
        r = (S[i] + S[j]) / distance.euclidean(cc[i],cc[j])
        if r > d:
          d = r
      D.append(d)

  #calcolo indice finale
  DB = 0 
  for i in range(0,len(D)):
    DB = DB + D[i]

  DB = DB / len(D)
  return DB


def best_k_means(df, maxClusters=10, approach="exclude_K_less_4_songs"):
  # primo cluster contenente tutto l'insieme di elementi
  bDB = 1000
    
  #costruisco gli altri cluster - voglio che si possa fare in maniera incrementale
  # k e' l'effettivo numero di cluster: il primo viene costruito prima del while
  
  #print(df_features.head().values.tolist())

  for k in range(2,maxClusters+1):
    
    #print(f"k={k}")
    kmeans = KMeans(n_clusters=k, random_state=1).fit(df)
  
    #TODO: DELETE
    #cl=kmeans.labels_.tolist()
    K = kmeans.labels_
    
    clf = NearestCentroid()
    clf.fit(df, K)
    centroids = clf.centroids_

    # add cluster labels to the DF
    df_temp = df.copy(deep=True)
    df_temp.insert(loc=len(df_temp.columns),column="kLabel",value=K)
    #df_count = df_temp.groupby(df_temp["kLabel"]).count()
    #sys.exit()
    
    # costruisco il clustering come serve a me
    cl = [ ] 
    for i in range(0,k):
      cl.append([ ])
    for i in range(0,len(K)):
      cl[K[i]].append(i)
    
    # manage clusters with less than 4 songs
    clustersToExclude = list()
    for i in range(1,k):
      numSongs = len(cl[i])
      if numSongs < 4:
        clustersToExclude.append(i)
    
    if len(clustersToExclude) > 0:
      if approach == "exclude_K_less_4_songs":
        # escludo clusterizzazione con k cluster (uscendo dal ciclo, quindi senza verificare se è best con DaviesBouldinIndex)
        continue

      elif approach == "exclude_cluster_less_4_songs":
        print("ERROR: best_k_means() with param. approach='exclude_cluster_less_4_songs' not yet implemented. EXIT.")
        sys.exit()
        """
        print(clustersToExclude)
        print(df_temp.shape[0])
        df_temp2 = df_temp[~df_temp["kLabel"].isin(clustersToExclude)]
        df = df_temp2.drop(columns=["kLabel"])
        print(df)
        
        print(centroids)
        centroids = [centroids[i] for i in range(0,len(centroids)) if i not in clustersToExclude]
        print(centroids)
        
        K = pd.Series(df_temp2["kLabel"]).values
        
        # costruisco il clustering come serve a me
        cl = [ ] 
        for i in range(0,k):
          cl.append([ ])
        for i in range(0,len(K)):
          cl[K[i]].append(i)
        """
    
    db = DaviesBouldinIndex(centroids, cl, df.values.tolist())
    if db < bDB:
      bDB = db
      bcl = cl
      bLabels = K
    print("\tnumber of clusters = ", len(cl))#, " - minSongs = ", minSongs)
    print("\tDB = ", db)
  
  # if no k satisfies the condition #songs-in-cluster >= 4, return all the songs (i.e. one cluster) - *S2.1*
  try:
    print("\t* best number of clusters = ", len(bcl))
    print("\t* bestDB = ", bDB)
    # restituisce il numero di cluster e le liste con i numeri di riga degli elementi in ogni cluster (centro incluso)
    return {"best-length": len(bcl), "best-index-list": bcl, "best-centroids": centroids, "best-labels": bLabels}
  except:
    print("\t* WARNING: best number of clusters = 1")
    # Aggiungo una canzone con feat. sintetiche (random) al songset e la considero l'unica del secondo cluster, 
    # così posso calcolare il centroide del mio unico cluster 
    df_temp2 = df.copy(deep=True)
    df_temp2.loc[-1] = np.random.randn(12).tolist()

    # creo labels dei cluster (tutti 0 tranne uno ad 1 in fondo per la canzone aggiunta) 
    temp = [0 for i in range(0, df.shape[0])]
    temp.append(1)
    K_temp = pd.Series(temp)

    # calcolo clusters
    clf = NearestCentroid()
    clf.fit(df_temp2, K_temp)
    centroids_temp = clf.centroids_
    
    # creo valori "giusti" da ritornare
    bcl = list()
    bcl.append([0 for i in range(0, df.shape[0])])
    centroids = np.delete(centroids_temp, 1, axis=0)
    bLabels = np.array([0 for i in range(0, df.shape[0])])
    return {"best-length": 1, "best-index-list": bcl, "best-centroids": centroids, "best-labels": bLabels}

def linearHeuristic(df, best_Kmeans):
  
  df_copy = df.copy(deep=True)

  kLength = best_Kmeans["best-length"]
  #kIndexes = best_Kmeans["best-index-list"]
  kCentroids = best_Kmeans["best-centroids"]
  kLabels = best_Kmeans["best-labels"]

  # add cluster labels to the DF
  df_copy.insert(loc=len(df_copy.columns),column="kLabel",value=kLabels)
  
  # add coordinates of centroids to the DF
  centroids = df_copy["kLabel"].apply(lambda x: kCentroids[x])
  df_copy.insert(loc=len(df_copy.columns),column="kCentroid",value=centroids)
  
  # add cluster labels to the DF
  coordinates = list()
  for index, row in df_copy.iterrows():
    coordinates.append([row["Acousticness"],row["Danceability"],row["Energy"],row["Instrumentalness"],row["Key"],row["Liveness"],row["Loudeness"],row["Mode"],row["Speechiness"],row["Tempo"],row["Time_signature"],row["Valence"]])
  df_copy.insert(loc=len(df_copy.columns),column="kCoordinates",value=coordinates)
  
  # add cluster labels to the DF
  distances = list()
  for index, row in df_copy.iterrows():
    distances.append(distance.euclidean(row["kCoordinates"],row["kCentroid"]))
  df_copy.insert(loc=len(df_copy.columns),column="kDistance",value=distances)
  
  df_copy.sort_values(by=["kLabel","kDistance"], inplace=True)
  
  # choose 4 songs in each cluster
  songsDFs = list()
  for i in range(0,kLength):
    df_slice = df_copy[df_copy["kLabel"] == i]
    numPoints = df_slice.shape[0]
    songsDFs.append(df_slice.iloc[[0, numPoints // 3 - 1, (numPoints // 3) * 2 - 1, numPoints - 1]])
    
  return songsDFs #pd.concat(songsDFs)  #{"songsByCluster": songsDFs, "centroids": kCentroids}





#n_per_sphere e' il numero di punti che si vogliono generare

def sample(center,radius,n_per_sphere):
  r = radius
  ndim = center.size
  x = np.random.normal(size=(n_per_sphere, ndim))
  ssq = np.sum(x**2,axis=1)
  fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
  frtiled = np.tile(fr.reshape(n_per_sphere,1),(1,ndim))
  p = center + np.multiply(x,frtiled)
  return p

"""
#calcola il raggio (come max delle distanze dal centroide)
def raggioMAX(cl,c,M):
  maxD = EDist(c,M[cl[0]]) #inizializzo la distanza massima a quella con il centro
  farthest = cl[0] #inizializzo il piu' lontano al centro 
  # distanze di ogni elemento j dal centroide
  for j in range(1,len(cl)):
    dist = EDist(c,M[cl[j]])
    if dist > maxD:
      maxD = dist
      farthest = cl[j]
  return [maxD, farthest]
"""


def getPointAllowedFeatureValues():
  print()
  
  
def sphereHeuristic(df, best_Kmeans, num_track = 50):
  
  # results here
  minDistSongs = list()
  
  df_copy = df.copy(deep=True)

  kLength = best_Kmeans["best-length"] # numero di cluster
  #kIndexes = best_Kmeans["best-index-list"]
  kCentroids = best_Kmeans["best-centroids"]
  kLabels = best_Kmeans["best-labels"]

  print(f"sphereHeuristic(): number of clusters = {kLength}")
  #sys.exit()
  
  # add cluster labels to the DF
  df_copy.insert(loc=len(df_copy.columns),column="kLabel",value=kLabels)
  
  # add coordinates of centroids to the DF
  centroids = df_copy["kLabel"].apply(lambda x: kCentroids[x])
  df_copy.insert(loc=len(df_copy.columns),column="kCentroid",value=centroids)
  
  # add cluster labels to the DF
  coordinates = list()
  for index, row in df_copy.iterrows():
    coordinates.append([row["Acousticness"],row["Danceability"],row["Energy"],row["Instrumentalness"],row["Key"],row["Liveness"],row["Loudeness"],row["Mode"],row["Speechiness"],row["Tempo"],row["Time_signature"],row["Valence"]])
  df_copy.insert(loc=len(df_copy.columns),column="kCoordinates",value=coordinates)
  
  # add cluster labels to the DF
  distances = list()
  for index, row in df_copy.iterrows():
    distances.append(distance.euclidean(row["kCoordinates"],row["kCentroid"]))
  df_copy.insert(loc=len(df_copy.columns),column="kDistance",value=distances)
  
  df_copy.sort_values(by=["kLabel","kDistance"], inplace=True)
  
  #print(df_copy[df_copy["TrackID"] == "5Jl1pMK3ffjw5nkbFUlseM"])
  #sys.exit()
  
  for i in range(0,kLength):
    df_slice = df_copy[df_copy["kLabel"] == i]
    
    #RandC = raggioMAX(CL[1][i],centroidi[i],M)
    #radius = RandC[0]
    farthestPoint = df_slice.iloc[df_slice.shape[0]-1]
    radius = farthestPoint["kDistance"]
    #RandC = [radius, farthestPoint]
    
    #genera 5 punti a caso nella sfera con centro il centroide e raggio quello che e' --> f matrice con un punto per riga
    #center = np.array(centroidi[i])
    #p = sample(center,radius,100)
    #f = np.array(p).tolist()
    center = np.array(farthestPoint["kCentroid"])
    p = sample(center,radius,100)
    f = np.array(p).tolist()
    
    ##################################################
    # TODO FPOGGI: questo va portato fuori dal ciclo #
    ##################################################
    #controllo che i valori siano nei range, altrimenti li modifico accordingly
    # conto quanti punti diversi trovo con track_id diversi --> np fino a 5
    #num_track = 50
    
    #pp = 0
    #totp = int(num_track/(4*kLength)) + 1 
    # tengo lista track_id
    #tr = [ ]
    ##################################################
    # END TODO FPOGGI: questo va portato fuori dal ciclo #
    ##################################################
    
    j = 0
    print ("ciclo i:" , i, " - num songs in cluster:", df_slice.shape[0])
    
    minDistTrackIds = list()
    # ciclo da ripetere totp ()volte
    #while (j < len(f) and pp <= totp ):
    while (j<len(f) and len(minDistTrackIds)<4):
    #for j in range (0,4):
      currRandomPoint = f[j]
      print("\t",i,j)
      #print(currRandomPoint)
      #print("\t",df_slice["kCoordinates"])#.shape[0])
      
      #1. minD, closest (indice della canz. nel cluster) e seed (la canzone del cluster) = del punto del cluster più vicino al j-esimo casuale
      #2. newP = punto con coordinate nel range previsto delle audio feature, partendo da j-esimo punto casuale
      minDist = -1
      for index, row in df_slice.iterrows():
        dist = distance.euclidean(currRandomPoint, row["kCoordinates"])
        #print(f"{dist} - {row['id2']}")
        if dist < minDist or minDist == -1:
          minDist = dist
          minDistSong = row
      
      # TODO FPOGGI: verificare
      # in teoria se punto più vicino al pto random è già preso dovrei skippare, ma così li scarto tutti.
      # Quindi decido di scegliere il punto più vicino al punto random e poi toglierlo da quelli possiibili. Così in 4 passaggi ho fatto. 
      """
      if minDistSong["TrackID"] not in minDistTrackIds:
        minDistTrackIds.append(minDistSong["TrackID"])
        minDistSongs.append({"minDist": minDist, "minDistSong": minDistSong})
      else:
        print("Already selected: skip.")
      """
      minDistTrackIds.append(minDistSong["TrackID"])
      minDistSongIndex = len(minDistSongs) #i*4+j
      minDistSongs.append({"index": minDistSongIndex, "randomPoint": currRandomPoint, "minDistSong": minDistSong, "minDist": minDist})
      df_slice.drop(df_slice[df_slice["TrackID"] == minDistSong["TrackID"]].index) #, inplace=True)
  
      j += 1
      
    print(len(minDistSongs))
    #sys.exit()  
  
  return minDistSongs






"""
def recommenderPrevNextSong(songs, numSongs, useFeatures=True, retryLimit=3):
	if numSongs > len(songs):
		print ("ERROR in recommenderPrevNextSong(): requested %d songs for a playlist composed of %d songs." % (numSongs, len(songs)))
		sys.exit()
	
	recommended_playlist = list()
	
	token = util.prompt_for_user_token(
		username=conf.username,
		scope=conf.scope,
		client_id=conf.client_id,
		client_secret=conf.client_secret,
		redirect_uri=conf.redirect_uri)

	# NB: per ogni listening history file, tutte le playlist generate dall'algoritmo sopra hanno lo stesso numero di canzoni => prendo #canzoni ultima playlist generata 
	for i in range(0, numSongs):
		if i == 0:
			songs_seeds = [songs[i]["trackId"], songs[i+1]["trackId"]]
		elif i == (numSongs-1):
			songs_seeds = [songs[i-1]["trackId"], songs[i]["trackId"]]
		else:
			songs_seeds = [songs[i-1]["trackId"], songs[i]["trackId"],songs[i+1]["trackId"]]
			
		retry = 0
		while token and retry < retryLimit:
			try:
				sp = spotipy.Spotify(auth=token)
				if useFeatures:
					song = songs[i]
					result = sp.recommendations(seed_tracks=songs_seeds,target_acousticness=song["features"][0],danceability=song["features"][1],target_energy=song["features"][2],target_instrumentalness=song["features"][3],target_key=int(song["features"][4]),target_liveness=song["features"][5],target_loudness=song["features"][6],target_mode=int(song["features"][7]),target_speechiness=song["features"][8],target_tempo=song["features"][9],target_time_signature=int(song["features"][10]),target_valence=song["features"][11],limit=1)
				else:
					result = sp.recommendations(seed_tracks=songs_seeds,limit=1)
				recommendedTrackId = result["tracks"][0]["id"]

				recommendedAudioFeatures = sp.audio_features([recommendedTrackId])
				recommended_song = {"trackId": recommendedTrackId, "features": [recommendedAudioFeatures[0]["acousticness"],recommendedAudioFeatures[0]["danceability"],recommendedAudioFeatures[0]["energy"],recommendedAudioFeatures[0]["instrumentalness"],recommendedAudioFeatures[0]["key"],recommendedAudioFeatures[0]["liveness"],recommendedAudioFeatures[0]["loudness"],recommendedAudioFeatures[0]["mode"],recommendedAudioFeatures[0]["speechiness"],recommendedAudioFeatures[0]["tempo"],recommendedAudioFeatures[0]["time_signature"],recommendedAudioFeatures[0]["valence"]]}
					
				recommended_playlist.append(recommended_song)
				break
			except:
				print (songs[i])
				print ("recommenderPrevNextSong()- Retry #" + str(retry) + " problem:", songs[i]["trackId"])
				retry += 1
				time.sleep(5)
		
		if retry >= retryLimit:
			print ("ERROR: recommenderPrevNextSong() retry limit %d - song %s" % (retryLimit,song["trackId"]))
			sys.exit()
	
	return recommended_playlist
"""

def isSongInSongsList(trackId, songsList, results):
  songs = songsList + results
  found = False
  for song in songs:
    #print(f"Comparing {trackId} - {song['id']}")
    if trackId == song["id"]:
      print(f"\t\t* Song Found! trackId: {trackId}")
      found = True
      break
  return found
  
def recommenderGetSongs(seed_track, features, num_songs, songsList, retryLimit=3, sleepTime=0.5, maxRequests=3):
  
  results = list()
  #client_credentials_manager = SpotifyClientCredentials(client_id=conf.client_id, client_secret=conf.client_secret)
  #sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
  token = util.prompt_for_user_token(
    username=conf.username,
    scope=conf.scope,
    client_id=conf.client_id,
    client_secret=conf.client_secret,
    redirect_uri=conf.redirect_uri)
  
  #print(f"len features = {len(features)}")
  retry = 0
  numRequests = 0
  foundSongs = 0
  #print(foundSongs, num_songs)
  while token and retry < retryLimit and foundSongs < num_songs: #(not found): #and token: 
    #print("ciao1")
    time.sleep(sleepTime)
    #print("ciao2")
    try:
      numRequests += 1
      print(f"\t\tnum_songs: {num_songs} - seed_track: {seed_track} (numRequests: {numRequests}/{maxRequests})")
      # TODO FPOGGI - CONTROLLARE
      if numRequests <= maxRequests:
        toRequest = num_songs-foundSongs
      elif maxRequests < numRequests <= maxRequests+2:
        toRequest = 100
      else:
        print("WARNING IN recommenderGetSongs(): the recommender returned 0 songs.")
        return results
      sp = spotipy.Spotify(auth=token)
      if len(features)>0:
        #print("Qui-1")
        recomms = sp.recommendations(seed_tracks=[seed_track],
          limit=toRequest,
          target_acousticness=features["Acousticness"],
          danceability=features["Danceability"],
          target_energy=features["Energy"],
          target_instrumentalness=features["Instrumentalness"],
          target_key=int(features["Key"]),
          target_liveness=features["Liveness"],
          target_loudness=features["Loudeness"],
          target_mode=int(features["Mode"]),
          target_speechiness=features["Speechiness"],
          target_tempo=features["Tempo"],
          target_time_signature=int(features["Time_signature"]),
          target_valence=features["Valence"])
        #print("Qui-2")
      else:
        #print("Quo-1")
        recomms = sp.recommendations(seed_tracks=[seed_track], limit=toRequest)
        #print("Quo-2")
    except Exception as e:
      print (f"recommenderGetSongs()- Retry #{retry} - song: {seed_track}")
      print(e)
      retry += 1
      time.sleep(5)
      continue
    
    #print(features)
    #print(recomms)
    print(f"\t\t\tLength results = {len(results)} - Recommended Songs = {len(recomms['tracks'])}")
    for track in recomms["tracks"]:
      # TODO FPOGGI - CONTROLLARE
      track_id = track["id"]
      #print(track_id)
      track_features = sp.audio_features(track_id)[0]
      
      track_features["trackName"]  = track["name"]
      track_features["artistName"] = " ".join([artist["name"] for artist in track["artists"]])
      track_features["popularity"] = track["popularity"]
       
      album_id = track["album"]["id"]
      #print(f"URL: https://open.spotify.com/album/{album_id}?highlight=spotify:track:{track_id}")
      #Acousticness	Danceability	Energy	Speechiness	Instrumentalness	Liveness	Valence	Loudeness	Tempo	Time_signature	Key	Mode
      
      if not isSongInSongsList(track_id, songsList, results):
        results.append(track_features)
        #num_songs -= 1
        foundSongs += 1
      if foundSongs >= num_songs:
        #print("\n\n*** QUI QUI QUI ***")
        break
    print(f"\t\t\tLength results = {len(results)}")
    
    
    if retry >= retryLimit:
      print (f"ERROR: recommenderGetSongs() retry limit {retryLimit} - song: {seed_track}")
      sys.exit()
  
  return results

#res = recommenderGetSongs("7CDaY0pk8qGFoahgxVVbaX", 3, list(), retryLimit=1)
#print(res)
#sys.exit()



def saveSongset(songsList, folderName, fileBasename, time_hour, numberOfCluster, cluster_method, heuristic_method):
  
  if type(time_hour) == str and len(time_hour) != 2:
    print("ERROR in saveSongset()")
    sys.exit()
  if type(time_hour) == int:
    time_hour = "{:02d}".format(time_hour)
  
  if type(numberOfCluster) == str and len(numberOfCluster) != 2:
    print("ERROR in saveSongset()")
    sys.exit()
  if type(numberOfCluster) == int:
    numberOfCluster = "{:02d}".format(numberOfCluster)
  
  
  #{'danceability': 0.582, 'energy': 0.314, 'key': 2, 'loudness': -11.886, 'mode': 0, 'speechiness': 0.313, 'acousticness': 0.458, 'instrumentalness': 0.342, 'liveness': 0.0986, 'valence': 0.427, 'tempo': 73.525, 'type': 'audio_features', 'id': '5KBiox7vnG3cnljh8MPBp8', 'uri': 'spotify:track:5KBiox7vnG3cnljh8MPBp8', 'track_href': 'https://api.spotify.com/v1/tracks/5KBiox7vnG3cnljh8MPBp8', 'analysis_url': 'https://api.spotify.com/v1/audio-analysis/5KBiox7vnG3cnljh8MPBp8', 'duration_ms': 92168, 'time_signature': 1, 'trackName': 'altissima', 'artistName': 'evän', 'popularity': 38}
  results = "track_id	acousticness	danceability	energy	instrumentalness	key	liveness	loudness	mode	speechness	tempo	time_signature	valence\n"
  for song in songsList:
    results += f"{song['id']}\t{song['acousticness']}\t{song['danceability']}\t{song['energy']}\t{song['instrumentalness']}\t{song['key']}\t{song['liveness']}\t{song['loudness']}\t{song['mode']}\t{song['speechiness']}\t{song['tempo']}\t{song['time_signature']}\t{song['valence']}\n"
  
  outdir = f"{folderName}/{fileBasename}/"
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  
  with open(f"{outdir}/{time_hour}-{cluster_method}_{numberOfCluster}_{heuristic_method}_{len(songsList)}.tsv", 'w') as f:
    f.write(results)
  





# cluster_method: "KM|FBF"
# heuristic_method = "LINEAR|SPHERE"
def generateSongset(csv_file, output_folder, cluster_method="KM", heuristic_method="LINEAR", min_songs_hour=10, min_ms_played=10000, max_clusters=10, num_tracks=100):
  
  print(csv_file)
  if ".csv" not in csv_file and ".tsv" not in csv_file:
    print("ERROR: only tsv and csv input files are allowed. Skip file.")
    
  cluster_method = cluster_method.upper()
  heuristic_method = heuristic_method.upper()
  
  df = mylib.loadData(csv_file, min_ms_played, delimiter="\t")
  
  # 3.1
  ntna_ntka = computeNTNA_NTKA(df)
    
  for time_hour in range(0,24):
    if time_hour < 21:
      continue
    
    # 3.2 - FILTERING
    df_h = songs_byHour(df, time_hour)
    
    # Remove duplicate songs
    df_h.drop_duplicates(subset ="TrackID", keep = "first", inplace = True)
    
    # controllo su numero di canzoni nella fascia oraria 
    if df_h.shape[0] < min_songs_hour:
      print(f"* hour {time_hour}: skip ({df_h.shape[0]} songs).")
      continue
    print(f"* hour {time_hour}: {df_h.shape[0]} songs.")
    
    #3.3 - CLUSTERING
    df_h_feat = df_h[["Acousticness", "Danceability", "Energy", "Instrumentalness", "Key", "Liveness", "Loudeness", "Mode", "Speechiness", "Tempo", "Time_signature", "Valence"]]
    
    if cluster_method == "KM":
      best_clustering = best_k_means(df_h_feat, max_clusters, "exclude_K_less_4_songs") #"exclude_cluster_less_4_songs")
      #print(best_clustering)
    elif cluster_method == "FBF":
      print(f"ERROR in generateSongset(): {cluster_method} not yet implemented. Exit.")
      sys.exit()
    else:
      print(f"ERROR in generateSongset(): cluster method {cluster_method} not defined. Exit.")
      sys.exit()
    
    kLength = best_clustering["best-length"]
    numReqs_perPoint = int(num_tracks/(4*kLength)) + 1
    feature_names = ["Acousticness","Danceability","Energy","Speechiness","Instrumentalness","Liveness","Valence","Loudeness","Tempo","Time_signature","Key","Mode"]

    ########################
    ### LINEAR HEURISTIC ###
    ########################
    if heuristic_method == "LINEAR":
      linear_kMeans = linearHeuristic(df_h, best_clustering)
      ###########################
      ### RECOMMENDER SPOTIFY ###
      ###########################
      results = list()
      kIndex = 0
      for df_group in linear_kMeans:
        # FIRST SONG
        print(f"\t{time_hour}) CLUSTER KM #{kIndex}/{kLength} - SONG #0/4")
        firstPoint = df_group.iloc[0]
        trackId = firstPoint["TrackID"]
        # get features
        centroidFeatures_list = firstPoint["kCentroid"]
        features = dict()
        for index in range(0,len(centroidFeatures_list)):
          features[feature_names[index]] = centroidFeatures_list[index]
        tracks = recommenderGetSongs(trackId, features, numReqs_perPoint, results, retryLimit=2, sleepTime=recommender_sleepTime)
        results.extend(tracks)
        #print(len(results))
        #res = recommenderGetSongs("7CDaY0pk8qGFoahgxVVbaX", numReqs_perPoint, list(), retryLimit=2, sleepTime=recommender_sleepTime)
        # OTHER THREE SONGS
        for i in range(1,4):
          print(f"\t{time_hour}) CLUSTER KM LINEAR #{kIndex}/{kLength} - SONG #{i}/4")
          point = df_group.iloc[i]
          trackId = point["TrackID"]
          tracks = recommenderGetSongs(trackId, point, numReqs_perPoint, results, retryLimit=2, sleepTime=recommender_sleepTime)
          results.extend(tracks)
          #print(len(results))
        kIndex += 1
    
    ########################
    ### SPHERE HEURISTIC ###
    ########################  
    elif heuristic_method == "SPHERE":
      sphere_kMeans = sphereHeuristic(df_h, best_clustering)
      
      ###########################
      ### RECOMMENDER SPOTIFY ###
      ###########################
      results = list()
      for item in sphere_kMeans:
        print(f"\t{time_hour}) {kLength} CLUSTER KM SPHERE - SONG #{item['index']}/{len(sphere_kMeans)}")
        #{"index": minDistSongIndex, "randomPoint": currRandomPoint, "minDistSong": minDistSong, "minDist": minDist}
        randomPoint = item["randomPoint"]
        features = dict()
        for index in range(0,len(randomPoint)):
          features[feature_names[index]] = randomPoint[index]
        #for index in range(0,len(centroidFeatures_list)):
        #  features[feature_names[index]] = centroidFeatures_list[index]
        clusterMinDistSong = item["minDistSong"]
        trackId = clusterMinDistSong["TrackID"]
        tracks = recommenderGetSongs(trackId, features, numReqs_perPoint, results, retryLimit=2, sleepTime=recommender_sleepTime)
        results.extend(tracks)
    else:
      print(f"ERROR in generateSongset(): heuristic {cluster_method} not defined. Exit.")
      sys.exit()
      
    #print(results)
    output_file_start = ntpath.basename(csv_file).replace(".csv","").replace(".tsv","")
    saveSongset(results, output_folder, output_file_start, time_hour, kLength, cluster_method, heuristic_method)







contents = glob(f"{csv_folder}*.csv")
#contents = glob(csv_folder + "*.csv")

contents.sort()
for csv_file in contents:
  print (csv_file)
  for clusterMethod in ["KM"]:
    for heuristic in ["SPHERE"]: #["LINEAR","SPHERE"]:
      generateSongset(csv_file, output_folder, clusterMethod, heuristic, min_songs_hour=minSongsHour, min_ms_played=minMsPlayed, max_clusters=maxClusters, num_tracks=numTracks) #min_songsHour = 10 min_msPlayed = 10000



'''
def test_kmeans(df):
  k=2
  temp = df[["Acousticness", "Danceability", "Energy", "Instrumentalness", "Key", "Liveness", "Loudeness", "Mode", "Speechiness", "Tempo", "Time_signature", "Valence"]]
  #temp = pd.DataFrame([[1,1],[2,2],[2,1],[1,2],[4,4],[5,5],[4,5],[5,4]])
  kmeans = KMeans(n_clusters=k, random_state=1).fit(temp)
  
  clf = NearestCentroid()
  clf.fit(temp, kmeans.labels_)
  print(clf.centroids_)

  """
  K = kmeans.labels_
  #K = np.array([1,1,1,1,2,2,2,2])
  #K = 
  # costruisco il clustering come serve a me
  L = [ ] 
  for i in range(0,k):
    L.append([ ])
  for i in range(0,len(K)):
    L[K[i]].append(i)
  C = [k,L]
  cl = C[1] # indici dei cluster
  #cc = centroid_list(cl,M)
  cc = [ ]
  K = len(cl)
  for k in range (0,K): 
    cc.append(centroid(cl[k],temp.values.tolist())) #M))
  #return cc
  print(cc)
  """
'''

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
0.522sphereHeuristic(
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
