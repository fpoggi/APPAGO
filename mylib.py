import csv
import sys
from datetime import datetime
from scipy.spatial import distance
import math

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pprint
import time

import conf



def getBestPlaylistInSongSet(playlist_original, songs, removeSelectedVertex=True):
	
	# controllo se len(playlist) > len(songs) - in caso affermativo, taglio playlist tenendo i primi len(songs)-esimi elementi
	if len(playlist_original) > len(songs):
		print ("WARNING: len(playlist) > len(songs) - %d > %d" % (len(playlist_original),len(songs)))
		playlist = playlist_original[0:len(songs)]
	else:
		playlist = list(playlist_original)
	
	
	# vettore con distanze coppie vertici consecutivi della listening history
	# lh con 20 canzoni, indici n = 0..19
	# dp = [0,D(h0,h1),...,D(hn-1,hn)]
	dp = list()
	dp.append(0)
	for i in range(1,len(playlist)):
		dp.append(euclideanDistance(playlist[i-1],playlist[i]))
		
	# matrice n x m con distanza fra list.history e canzoni candidate
	# list.history su righe (n) e canzoni candidate su colonne (m)
	dhc = list()
	for i in range (0,len(playlist)):
		row = list()
		for j in range (0,len(songs)):
			row.append(euclideanDistance(playlist[i],songs[j]))
		dhc.append(row)

	# matrice m x m con distanze fra coppie di canzoni candidate
	# NB: la matrice ha diagonale con valori a 0. Inoltre è a specchio sulla diagonale => POSSIILE OTTIMIZZAZIONE
	dcc = list()
	for i in range (0,len(songs)):
		row = list()
		for j in range (0,len(songs)):
			if i == j:
				row.append(0)
			else:
				row.append(euclideanDistance(songs[i],songs[j]))
		dcc.append(row)	
	
	# matrice n x m di costi, dove M[i,j] indica minor costo (PPD) ottenibile con percorso lungo i che termina in j	
	# inizializzato a 0
	M = list()
	temp = list()
	for j in range (0,len(songs)):
		temp.append(0)
	for i in range (0,len(playlist)):
		M.append(list(temp))
	
	# matrice n x m di vettori percorso, dove P[i,j] indica il percorso (è un vettore di canzoni candidate) lungo i che termina in j che porta al minor costo 
	# inizializzato con liste vuote
	P = list()
	temp = list()
	for j in range (0,len(songs)):
		temp.append(list())
	for i in range (0,len(playlist)):
		P.append(list(temp))

	# ricerca ottimo: compilo matrici M e P
	for i in range (0,len(playlist)):
		for j in range (0,len(songs)):
			if i == 0:
				M[0][j] = dhc[0][j]
				P[0][j].append(j)
			else:
				temp = list()
				for k in range(0,len(songs)):
					# se non voglio ripetizioni, imposto valore della funzione da ottimizzare a infinito
					# TODO: verificare che si ottenga ancora l'ottimo
					if removeSelectedVertex and (j in P[i-1][k]):
						temp.append(float("inf"))
					else:
						temp.append(M[i-1][k] + abs(dp[i] - dcc[k][j]))
				index_best = temp.index(min(temp))
				
				M[i][j] = dhc[i][j] + temp[index_best]
				temp2 = list(P[i-1][index_best])
				temp2.append(j)
				P[i][j] = temp2

	optimum_lastSongIndex = M[len(playlist)-1].index(min(M[len(playlist)-1]))
	playlist_songs_indexes = (P[len(playlist)-1][optimum_lastSongIndex])
	playlist_songs = list()
	for index in playlist_songs_indexes:
		playlist_songs.append(songs[index])
	playlist_weight = (M[len(playlist)-1][optimum_lastSongIndex])
	return {"weight": playlist_weight, "playlist": playlist_songs, "playlist-indexes": playlist_songs_indexes}


def recommenderSingleSong(songs, numSongs, useFeatures=True, retryLimit=3):
	
	if numSongs > len(songs):
		print ("ERROR in recommenderSingleSong(): requested %d songs for a playlist composed of %d songs." % (numSongs, len(songs)))
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
		song = songs[i]
		
		retry = 0
		while token and retry < retryLimit:
			try:
				sp = spotipy.Spotify(auth=token)
				if useFeatures:
					result = sp.recommendations(seed_tracks=[song["trackId"]],target_acousticness=song["features"][0],danceability=song["features"][1],target_energy=song["features"][2],target_instrumentalness=song["features"][3],target_key=int(song["features"][4]),target_liveness=song["features"][5],target_loudness=song["features"][6],target_mode=int(song["features"][7]),target_speechiness=song["features"][8],target_tempo=song["features"][9],target_time_signature=int(song["features"][10]),target_valence=song["features"][11],limit=1)
				else:
					result = sp.recommendations(seed_tracks=[song["trackId"]],limit=1)
				recommendedTrackId = result["tracks"][0]["id"]
				
				recommendedAudioFeatures = sp.audio_features([recommendedTrackId])
				recommended_song = {"trackId": recommendedTrackId, "features": [recommendedAudioFeatures[0]["acousticness"],recommendedAudioFeatures[0]["danceability"],recommendedAudioFeatures[0]["energy"],recommendedAudioFeatures[0]["instrumentalness"],recommendedAudioFeatures[0]["key"],recommendedAudioFeatures[0]["liveness"],recommendedAudioFeatures[0]["loudness"],recommendedAudioFeatures[0]["mode"],recommendedAudioFeatures[0]["speechiness"],recommendedAudioFeatures[0]["tempo"],recommendedAudioFeatures[0]["time_signature"],recommendedAudioFeatures[0]["valence"]]}
					
				recommended_playlist.append(recommended_song)
				break
			except:
				print ("recommenderSingleSong - Retry #" + str(retry) + " problem:", song["trackId"])
				retry += 1
				time.sleep(5)
		
		if retry >= retryLimit:
			print ("ERROR: recommenderSingleSong() retry limit %d - song %s" % (retryLimit,song["trackId"]))
			sys.exit()
	
	return recommended_playlist


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


def recommenderPoolSongs(songs, numFinalSongs, retryLimit=3):
	
	poolSize = 5
	
	#if numFinalSongs > len(songs):
	#	print ("ERROR in recommenderPoolSongs(): requested %d songs for a playlist composed of %d songs." % (numFinalSongs, len(songs)))
	#	sys.exit()
	
	recommended_pool = list()
	
	token = util.prompt_for_user_token(
		username=conf.username,
		scope=conf.scope,
		client_id=conf.client_id,
		client_secret=conf.client_secret,
		redirect_uri=conf.redirect_uri)

	numPools = len(songs) // poolSize
	pools = list()
	for i in range (0,numPools):
		songs_seeds = list()
		poolSongs = songs[i*poolSize:(i+1)*poolSize]
		for poolSong in poolSongs:
			songs_seeds.append(poolSong["trackId"])
		pools.append(songs_seeds)
	
	if (len(songs) % poolSize) > 0:
		songs_seeds = list()
		poolSongs = songs[len(songs)-poolSize:len(songs)] #songs[numFinalSongs-poolSize:numFinalSongs]
		for poolSong in poolSongs:
			songs_seeds.append(poolSong["trackId"])
		pools.append(songs_seeds)
	
	for pool in pools:
		retry = 0
		while token and retry < retryLimit:
			try:
				sp = spotipy.Spotify(auth=token)
				results = sp.recommendations(seed_tracks=pool,limit=numFinalSongs // poolSize)
				
				for result in results["tracks"]:
					recommendedTrackId = result["id"]
					recommendedAudioFeatures = sp.audio_features([recommendedTrackId])
					recommended_song = {"trackId": recommendedTrackId, "features": [recommendedAudioFeatures[0]["acousticness"],recommendedAudioFeatures[0]["danceability"],recommendedAudioFeatures[0]["energy"],recommendedAudioFeatures[0]["instrumentalness"],recommendedAudioFeatures[0]["key"],recommendedAudioFeatures[0]["liveness"],recommendedAudioFeatures[0]["loudness"],recommendedAudioFeatures[0]["mode"],recommendedAudioFeatures[0]["speechiness"],recommendedAudioFeatures[0]["tempo"],recommendedAudioFeatures[0]["time_signature"],recommendedAudioFeatures[0]["valence"]]}
					recommended_pool.append(recommended_song)
				break
				
			except:
				print (pool)
				print ("recommenderPoolSongs()- Retry #" + str(retry) + " problem:", recommendedTrackId)
				retry += 1
				time.sleep(5)
	
	# elimino canzoni del pool che sono nella listening history E le canzoni ripetute
	recommended_pool_difference = set(map(lambda x: x["trackId"],recommended_pool)).difference(set(map(lambda x: x["trackId"],songs)))
	if len(recommended_pool_difference) != len(recommended_pool):
		print ("WARNING recommenderPoolSongs(): ci sono canzoni nel pool presenti nella listening history.")
		res = list(filter(lambda x: x["trackId"] in recommended_pool_difference, recommended_pool))
		if len(res) >= numFinalSongs:
			return res[0:numFinalSongs]
		else:
			return res
	return recommended_pool
	



def getTimeStr():
	now = datetime.now()
	return now.strftime("%Y-%m-%dT%H:%M:%SZ") #"%Y-%m-%dT%H:%M:%S.%fZ"




def euclideanDistance(song1,song2):
	if len(song1["features"]) != len(song2["features"]):
		print ("ERROR (euclideanDistance): the argument lengths are different")
		sys.exit()
	return distance.euclidean(song1["features"],song2["features"])


def computePD(history, optimal_playlist):
	if len(optimal_playlist) > len(history):
		print ("ERROR: lunghezza history < playlist fornita")
		sys.exit()
	res = 0
	for i in range(0,len(optimal_playlist)):
		if i == 0:
			res += euclideanDistance(history[0],optimal_playlist[0])
		else:
			res += euclideanDistance(history[i],optimal_playlist[i]) + abs(euclideanDistance(history[i],history[i-1]) - euclideanDistance(optimal_playlist[i],optimal_playlist[i-1]))
	return res


def printPlaylist(playlist):
	for song in playlist:
		print ("%s: %s - %s" % (song['played_at'], song['track_name'], song['artist_name']))


def getSongInfo(trackId):
	client_credentials_manager = SpotifyClientCredentials(client_id=conf.client_id, client_secret=conf.client_secret)
	spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
	trackInfo = spotify.track(trackId)
	album = trackInfo["album"]["name"]
	artists = list()
	for artistRecord in trackInfo["artists"]:
		artists.append(artistRecord["name"])
	artist = " AND ".join(artists)
	trackName = trackInfo["name"]
	return {"artist": artist, "trackName": trackName, "album": album}


