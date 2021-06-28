import pandas as pd 

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
#sklearn.neighbors.NearestCentroid
import numpy as np
import math
from scipy.special import gammainc

from scipy.spatial import distance

from glob import glob
import sys
  
csv_folder = "../listening-history/csv/"
min_msPlayed = 10000
maxClusters = 10


# loads csv data into a dataframe.
# filters songs by size (length >= min_ms_played)
# computes new columns, i.e. date, time and time-hour
def loadData(filename, min_msPlayed):
  df = pd.read_csv(filename,delimiter=";", encoding="UTF-8") #"ISO-8859-1")  
  numLines = df.shape[0]
  df["endTime"] = pd.to_datetime(df["endTime"], format="%Y-%m-%d %H:%M")
  df = df.sort_values(by=["endTime"])
  df_filtered = df[df["msPlayed"] >= min_msPlayed]
  print(f"File: {filename} - Size: {df_filtered.shape[0]}/{numLines}")
  #print("File: " + filename + " - Size: " + str(df_filtered.shape[0]) + "/" + str(numLines))
  
  #df_filtered["date"] = df_filtered["endTime"].apply(lambda x: x.date())
  temp = df_filtered["endTime"].apply(lambda x: str(x.date()))
  df_filtered.insert(loc=3,column="date",value=temp)
  temp = df_filtered["endTime"].apply(lambda x: str(x).split(" ")[1]) #str(x.hour) + ":" + str(x.minute) )
  df_filtered.insert(loc=4,column="time",value=temp)
  temp = df_filtered["endTime"].apply(lambda x: str(x).split(" ")[1].split(":")[0]) #str(x.hour) + ":" + str(x.minute) )
  df_filtered.insert(loc=5,column="time-hour",value=temp)
  
  return df_filtered



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
  bDB = 100
    
  #costruisco gli altri cluster - voglio che si possa fare in maniera incrementale
  # k e' l'effettivo numero di cluster: il primo viene costruito prima del while
  
  #print(df_features.head().values.tolist())

  for k in range(2,maxClusters+1):
        
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
      # 
      if approach == "exclude_K_less_4_songs":
      #cerco cluster con minor numero di songs (per controllare che #songs nei cluster sia >= 4 - per passi seguenti)
      #minSongs = len(cl[0])
      #for i in range(1,k):
      #  numSongs = len(cl[i])
      #  if numSongs < minSongs:
      #    minSongs = numSongs
      #if  minSongs < 4:
      #  continue
        continue

      elif approach == "exclude_cluster_less_4_songs":
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
        

    
    
    
    db = DaviesBouldinIndex(centroids, cl, df.values.tolist())
    if db < bDB:
      bDB = db
      bcl = cl
      bLabels = K
    print("number of cluster = ", len(cl))#, " - minSongs = ", minSongs)
    print("DB = ", db)
        
  print("best number of cluster = ", len(bcl))
  print("bestDB = ", bDB)
  
  # restituisce il numero di cluster e le liste con i numeri di riga degli elementi in ogni cluster (centro incluso)
  return {"best-length": len(bcl), "best-index-list": bcl, "best-centroids": centroids, "best-labels": bLabels}
  

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
    
  return songsDFs #{"songsByCluster": songsDFs, "centroids": kCentroids}





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
  
  df_copy = df.copy(deep=True)

  kLength = best_Kmeans["best-length"] # numero di cluster
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
 
  for i in range(0,kLength):
    df_slice = df_copy[df_copy["kLabel"] == i]
    
    #RandC = raggioMAX(CL[1][i],centroidi[i],M)
    #radius = RandC[0]
    farthestPoint = df_slice.iloc[df_slice.shape[0]-1]
    radius = farthestPoint["kDistance"]
    RandC = [radius, farthestPoint]
    
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
    pp = 0
    totp = int(num_track/(4*kLength)) + 1 
    # tengo lista track_id
    tr = [ ]
    ##################################################
    # END TODO FPOGGI: questo va portato fuori dal ciclo #
    ##################################################
    
    j = 0
    print ("ciclo i" , i)
    
    minDistTrackIds = list()
    minDistSongs = list()
    # ciclo da ripetere totp ()volte
    #while (j < len(f) and pp <= totp ):
    while (i==4 and j<len(f) and len(minDistSongs)<4):
    #for j in range (0,4):
      currRandomPoint = f[j]
      print("\t",i,j)
      #print(currRandomPoint)
      print("\t",df_slice["kCoordinates"])#.shape[0])
      
      #1. minD, closest (indice della canz. nel cluster) e seed (la canzone del cluster) = del punto del cluster più vicino al j-esimo casuale
      #2. newP = punto con coordinate nel range previsto delle audio feature, partendo da j-esimo punto casuale
      minDist = -1
      for index, row in df_slice.iterrows():
        #print(row)
        #sys.exit()
        dist = distance.euclidean(currRandomPoint, row["kCoordinates"])
        if dist < minDist or minDist == -1:
          minDist = dist
          minDistSong = row
      
      if minDistSong["TrackID"] not in minDistTrackIds:
        minDistTrackIds.append(minDistSong["TrackID"])
        minDistSongs.append({"minDist": minDist, "minDistSong": minDistSong})
      else:
        print("Already selected: skip.")
      j += 1
    
    print(len(minDistSongs))
    #sys.exit()  
      
                
  return "ciao"





    
contents = glob(f"{csv_folder}*.csv")
#contents = glob(csv_folder + "*.csv")

contents.sort()
for filename in contents:
  print (filename)
  df = loadData(filename, min_msPlayed)
  
  #df.to_csv("test.csv")
  #print(df.groupby(["date","time-hour"]).count())
  
  # 3.1
  ntna_ntka = computeNTNA_NTKA(df)
  
  # 3.2
  #df_h = songs_byHour(df, "09")
  df_h = df
  
  # Remove duplicate songs
  df_h.drop_duplicates(subset ="TrackID",
                     keep = False, inplace = True)
  
  #3.3
  #K-MEANS
  #print(df_filtered.columns.values)
  #test_kmeans(df_filtered)
  df_h_feat = df_h[["Acousticness", "Danceability", "Energy", "Instrumentalness", "Key", "Liveness", "Loudeness", "Mode", "Speechiness", "Tempo", "Time_signature", "Valence"]]
  
  best_Kmeans = best_k_means(df_h_feat, maxClusters, "exclude_cluster_less_4_songs")
  sys.exit()
  #linear_kMeans = linearHeuristic(df_h, best_Kmeans)
  sphere_kMeans = sphereHeuristic(df_h, best_Kmeans)
  #print(res)
  
  sys.exit()
  
  #for index, row in df_filtered.iterrows():
  #  print(row["endTime"])
  #  print(row["endTime"].date())
  #  print(row["endTime"].hour)
  
  #break






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
