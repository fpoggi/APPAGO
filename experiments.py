import csv
import sys
from datetime import datetime
import math
import glob
import os

import mylib

# Params
listeningHistoryFolder = "input/listeningHistory/"
songsetFolder_1 = "input/songset_1/"
songsetFolder_2 = "input/songset_2/"
resFolder = "output/" + mylib.getTimeStr() + "/"
resValuesFilename = "results.tsv"
noRepeatedSongs = False


def getSongSetAcronym(songsetType):
	acronymDict = {
		"FPF": "DYN-1",
		"KM": "DYN-2",
		"FPF-MAX-req": "DYN-3",
		"KM-MAX-req": "DYN-4"
	}
	return acronymDict[songsetType]
	
	
def getSongSetFiles(hour):
	res = dict()
	
	temp = glob.glob(songsetFolder_1 + "csv_" + str(hour) + "_FPF_*.csv")
	if len(temp) != 1:
		print ("ERROR: missing FPF file for hour " + str(hour) )
		sys.exit()
	res["FPF"] = temp[0]
	
	temp = glob.glob(songsetFolder_1 + "csv_" + str(hour) + "_KM_*.csv")
	if len(temp) != 1:
		print ("ERROR: missing KM file for hour " + str(hour) )
		sys.exit()
	res["KM"] = temp[0]
	
	temp = glob.glob(songsetFolder_2 + "csv_" + str(hour) + "_FPF_*MAX_richieste*.csv")
	if len(temp) != 1:
		print ("ERROR: missing FPF_MAX_richieste file for hour " + str(hour) )
		sys.exit()
	res["FPF-MAX-req"] = temp[0]
	
	temp = glob.glob(songsetFolder_2 + "csv_" + str(hour) + "_KM_*MAX_richieste*.csv")
	if len(temp) != 1:
		print ("ERROR: missing KM_MAX file for hour " + str(hour) )
		sys.exit()
	res["KM-MAX-req"] = temp[0]
	
	return res


####################################################
# Required to get listeningHistoryFile header      #
csvBigFile = 'input/CSV_BIG.csv'

featureList = ['acousticness','danceability','energy','instrumentalness','key','liveness','loudness','mode','speechness','tempo','time_signature','valence']

# Produce a list of dictionaries, one for each song.
# Each dictionary contains two fields: trackId (-> string) and features (-> list of features)
def getArrayFeatures(songsDictionary, features):
	res = list()
	for songDictionary in songsDictionary:
		trackId = songDictionary["track_id"]
		temp = list()
		for feature in features:
			temp.append(float(songDictionary[feature]))
		res.append({"trackId": trackId, "features": temp})
	return res


# given the csv file of the songs, returns the longest playlist
# (i.e. the longest list of songs played in a day) sorted by
# listening time
def getLongestPlaylist(filename):
	firstLine = []
	with open(csvBigFile, encoding="ISO-8859-1") as f:
		l = f.readline()
		firstLine = l.replace('\n','').split(';')
	
	playlist = list()
	with open(filename, encoding="ISO-8859-1") as csvfile:
		reader = csv.DictReader(csvfile, delimiter=';', fieldnames=firstLine)
		try:
			sortedlist = sorted(reader, key=lambda row: datetime.strptime(row['played_at'], '%Y-%m-%dT%H:%M:%S.%fZ'), reverse=False)
		except:
			print (filename)
			sys.exit()
		
		# ricerca giorno con playlist piu lunga
		daysSet = set()
		daysList = list()
		for song in sortedlist:
			daysSet.add(song['date'])
			daysList.append(song['date'])
		daysCountDict = dict()
		for day in daysSet:
			count = sum(1 for x in daysList if x == day)
			daysCountDict[day] = count
		maxDay = max(daysCountDict, key=lambda key: daysCountDict[key])
		maxCount = daysCountDict[maxDay]
		
		# filtro canzoni: giorno con piu ascolti
		for song in sortedlist:
			if song['date'] == maxDay:
				playlist.append(song)
	
	return getArrayFeatures(playlist, featureList)
	

def getSongSet(filename):
	res = list()
	with open(filename, encoding="ISO-8859-1") as csvfile:
		reader = csv.DictReader(csvfile, delimiter=';')
		for song in reader:
			res.append(song)
	return getArrayFeatures(res, featureList)


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
	outfile.write("Hour	Acronym	Listening history length	songset type	songset file	songset length	playlist type	playlist length	PPD value	playlist filename\n")
	for line in res:
		outfile.write("%s\t%s\t%d\t%s\t%s\t%d\t%s\t%d\t%s\t%s\n" % (line["hour"], line["acronym"], line["listening history length"], line["songset type"], line["songset file"], line["songset length"], line["playlist type"], line["playlist length"], str("{:.3f}".format(line["PPD value"])), line["acronym"] + "-" + line["hour"] + ".tsv") )

# save tsv with songs (optimal playlist)
for line in res:
	with open(resFolder + line["acronym"] + "-" + line["hour"] + ".tsv", "w") as outfile:
		outfile.write("track id\n")
		for song in line["computed playlist"]:
			outfile.write(str(song["trackId"]) + "\n")

