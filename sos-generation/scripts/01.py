import pandas as pd 

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
#sklearn.neighbors.NearestCentroid
import numpy as np
import math

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
  return df[df["time-hour"] == h]
  



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

#Section 3.3
"""
# prende la matrice M e calcola il miglior numero di cluster usando DB
def best_k_means_Manuela(M):
  # primo cluster contenente tutto l'insieme di elementi
  m = len(M)
  bDB = 100
    
  #costruisco gli altri cluster - voglio che si possa fare in maniera incrementale
  # k e' l'effettivo numero di cluster: il primo viene costruito prima del while
  k = 2
    
  while k < 10:
        
    C = k_means(M,k)
    #cl = indici_cluster(C) # non va con k-means perche' C e' come in FPF
    cl = C[1] # indici dei cluster
    cc = centroid_list(cl,M)
    db = DB(cc,cl,M)
    # per forzarlo a fare 3 o 4 cluster
    #if (len(cl) == 2 or len(cl) == 3):
    #    db = 10
    if db < bDB:
      bDB = db
      bcl = cl
    print "number of cluster = ", len(cl)
    print "DB = ", db
    k = k + 1
        
  print "best number of cluster = ", len(bcl)
  print "bestDB = ", bDB
  # restituisce il numero di cluster e le liste con i numeri di riga degli elementi in ogni cluster (centro incluso)
  return (len(bcl),bcl)
"""

# Funzione che calcola la distanza euclidea tra due vettori
def EDist(X,Y):
  d = 0
  if len(X) == len(Y):
    for i in [0,len(X) -1]:
      d = d + (X[i] - Y[i])**2
  return math.sqrt(d)
    
  
# calocla l'indice Davis-Boudin. Input cc = centroidi; cl = clustering come indici di riga di M;
# M = matrice dati. Restituisce indice numerico
def DB(cc, cl, M):
  S = [ ]
  # per ogni cluster i calcolo i valori S
  for i in range(0,len(cl)):
    s = 0
    c = cl[i]
    # distanze di ogni elemento j dal centroide
    for j in range(0,len(c)):
      s = s + EDist(cc[i],M[c[j]])
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
        r = (S[i] + S[j]) / EDist(cc[i],cc[j])
        if r > d:
          d = r
      D.append(d)

  #calcolo indice finale
  DB = 0 
  for i in range(0,len(D)):
    DB = DB + D[i]

  DB = DB / len(D)
  return DB

def best_k_means(df):
  # primo cluster contenente tutto l'insieme di elementi
  #m = len(M)
  bDB = 100
    
  #costruisco gli altri cluster - voglio che si possa fare in maniera incrementale
  # k e' l'effettivo numero di cluster: il primo viene costruito prima del while
  maxClusters = 10
    
  df_features = df[["Acousticness", "Danceability", "Energy", "Instrumentalness", "Key", "Liveness", "Loudeness", "Mode", "Speechiness", "Tempo", "Time_signature", "Valence"]]
  print(df_features.head().values.tolist())

  for k in range(2,maxClusters+1):
        
    """
    C = k_means(M,k)
    #cl = indici_cluster(C) # non va con k-means perche' C e' come in FPF
    cl = C[1] # indici dei cluster
    cc = centroid_list(cl,M)
    """
    
    #df_features = pd.DataFrame([[1,1],[2,2],[2,1],[1,2],[4,4],[5,5],[4,5],[5,4]])
    kmeans = KMeans(n_clusters=k, random_state=1).fit(df_features)
  
    #TODO: DELETE
    #cl=kmeans.labels_.tolist()
    K = kmeans.labels_
    #print K
    
    clf = NearestCentroid()
    clf.fit(df_features, K)
    centroids = clf.centroids_
    #print(centroids)

    # costruisco il clustering come serve a me
    cl = [ ] 
    for i in range(0,k):
      cl.append([ ])
    for i in range(0,len(K)):
      cl[K[i]].append(i)
    
    db = DB(centroids, cl, df_features.values.tolist())
    if db < bDB:
      bDB = db
      bcl = cl
    print("number of cluster = ", len(cl))
    print("DB = ", db)
        
  print("best number of cluster = ", len(bcl))
  print("bestDB = ", bDB)
  # restituisce il numero di cluster e le liste con i numeri di riga degli elementi in ogni cluster (centro incluso)
  return (len(bcl),bcl)
  """
    db = DB(cc,cl,M)
    # per forzarlo a fare 3 o 4 cluster
    #if (len(cl) == 2 or len(cl) == 3):
    #    db = 10
    if db < bDB:
      bDB = db
      bcl = cl
    print "number of cluster = ", len(cl)
    print "DB = ", db
        
  print "best number of cluster = ", len(bcl)
  print "bestDB = ", bDB
  # restituisce il numero di cluster e le liste con i numeri di riga degli elementi in ogni cluster (centro incluso)
  return (len(bcl),bcl)
  """
  
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


contents = glob(f"{csv_folder}*.csv")
#contents = glob(csv_folder + "*.csv")

contents.sort()
for filename in contents:
  print (filename)
  df = pd.read_csv(filename,delimiter=";", encoding="UTF-8") #"ISO-8859-1")  
  numLines = df.shape[0]
  df["endTime"] = pd.to_datetime(df["endTime"], format="%Y-%m-%d %H:%M")
  df = df.sort_values(by=["endTime"])
  df_filtered = df[df["msPlayed"] > min_msPlayed]
  print(f"File: {filename} - Size: {df_filtered.shape[0]}/{numLines}")
  #print("File: " + filename + " - Size: " + str(df_filtered.shape[0]) + "/" + str(numLines))
  
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

  # 3.1
  #ntna_ntka = computeNTNA_NTKA(df_filtered)
  #print(ntna_ntka)

  # 3.2
  #print(songs_byHour(df_filtered, "09"))
  #print(df_filtered.info())
  #sys.exit()
  
  #3.3
  #K-MEANS
  #print(df_filtered.columns.values)
  #test_kmeans(df_filtered)
  best_k_means(df_filtered)
  sys.exit()
  
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
