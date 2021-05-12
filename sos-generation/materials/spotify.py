# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:00:19 2018

@author: Manuela
"""

import numpy as np
import pandas as pd
import random
import math
from sklearn.cluster import KMeans
from scipy.special import gammainc
#import new_list as nl
#import iterativeFPF as iFPF



def ReadInput( filename ):

    name_of_first_column_to_be_ignored = 'track_id'
    name_of_second_column_to_be_ignored = ' track_name'
    name_of_third_column_to_be_ignored = ' duration_ms'
    name_of_fourth_column_to_be_ignored = ' album_name'
    name_of_fifth_column_to_be_ignored = ' album_type'
    name_of_sixth_column_to_be_ignored = ' artist_name'
    name_of_seventh_column_to_be_ignored = ' track_popularity'
    name_of_eighth_column_to_be_ignored = ' explicit'
    name_of_nineth_column_to_be_ignored = ' artist_genres'
    name_of_tenth_column_to_be_ignored = ' album_genres'
    name_of_eleventh_column_to_be_ignored = ' played_at'  
    

# Read the CSV file with the Pandas lib.
#path_dir = ".\\"
    dataframe = pd.read_csv( filename, encoding = "cp1252", sep = ';' ) # "ISO-8859-1")
    df = dataframe.copy(deep=True)


# Prepares the dataset to delete the columns that aren't used:

    if name_of_first_column_to_be_ignored in df.columns:
        df = df.drop(name_of_first_column_to_be_ignored, 1)

    if name_of_second_column_to_be_ignored in df.columns:
        df = df.drop(name_of_second_column_to_be_ignored, 1)

    if name_of_third_column_to_be_ignored in df.columns:
        df = df.drop(name_of_third_column_to_be_ignored, 1)

    if name_of_fourth_column_to_be_ignored in df.columns:
        df = df.drop(name_of_fourth_column_to_be_ignored, 1)

    if name_of_fifth_column_to_be_ignored in df.columns:
        df = df.drop(name_of_fifth_column_to_be_ignored, 1)

    if name_of_sixth_column_to_be_ignored in df.columns:
        df = df.drop(name_of_sixth_column_to_be_ignored, 1)

    if name_of_seventh_column_to_be_ignored in df.columns:
        df = df.drop(name_of_seventh_column_to_be_ignored, 1)

    if name_of_eighth_column_to_be_ignored in df.columns:
        df = df.drop(name_of_eighth_column_to_be_ignored, 1)

    if name_of_nineth_column_to_be_ignored in df.columns:
        df = df.drop(name_of_nineth_column_to_be_ignored, 1)

    if name_of_tenth_column_to_be_ignored in df.columns:
        df = df.drop(name_of_tenth_column_to_be_ignored, 1)

    if name_of_eleventh_column_to_be_ignored in df.columns:
        df = df.drop(name_of_eleventh_column_to_be_ignored, 1)
    #restituisce una matrice (e non un dataframe di pandas)
    M = df.values
    #rc = df.count()
    return M 

def WriteOutput( M , filename):
    file = open( filename, "w")
    np.savetxt(file, M)
    file.close()
    return

# Funzione che calcola la distanza euclidea tra due vettori
def EDist(X,Y):
    d = 0
    if len(X) == len(Y):
        for i in [0,len(X) -1]:
         d = d + (X[i] - Y[i])**2
    return math.sqrt(d)

#per ordinare lista di liste in base ad una chiave specifica (in questo caso la seconda della coppia)
def getKey(item):
    return item[1]

# calcola il centroide a partire da una lista di indici di righe (che rappresentano gli elementi di un cluster)
# e la matrice in cui stanno i vettori
# restitutisce una lista con i valori del centroide
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

# prende lista di liste che rappresenta clustering (indici matrice) e matrice valori e restituisce una lista
# con i centroidi (valori) dei cluster 
def centroid_list(cl,M):
    #il for prende i cluster calcola il centroide per ogni cluster (si puo' fare alla fine del for precedente)
    # cc memorizza i centroidi
    cc = [ ]
    K = len(cl)
    for k in range (0,K): 
        cc.append(centroid(cl[k],M))

    #print cc
    return cc


# calcola K cluster usando FPF, i cluster risultanti sono in cl (lista di liste, ogni lista contiene gli indici di M relativi agli elementi nel cluster)

def FPF(K,M):
    # primo cluster contenente tutto l'insieme di elementi

    m = len(M)
    c = random.randint(0,m-1)
    #print "cluster 1"
    #print "primo centro scelto a caso" , c

    L = [ ]
    C = [ ]

    for i in range(0,m):
        L.append([i, EDist(M[i],M[c])])

    L = sorted(L, key=getKey, reverse=True)

    L.remove([c, 0.0])

    C = [ [c, L] ]

    #costruisco gli altri cluster 

    for k in range(1,K):
        #print "clustser", k+1

        #cerco l'elemento piu' lontano dal suo centro tra tutti i cluster
        #all'inizio e' il primo della prima lista non vuota - indice h 
        h = 0 
        while C[h][1] == [ ] and h < len(C):
            h + 1

        max_e = C[h][1][0][0]
        max_d = C[h][1][0][1]

        # il primo cluster potrebbe essersi svuotato 
        #max_e = C[0][1][0][0]
        #max_d = C[0][1][0][1]

        #print "max_e=", max_e, "max_d=", max_d
        #cerco nelle altre liste (ordinate in senso decrescente), quindi controllo solo il primo (se c'e')
        # e parto da h+1 perche' prima so gia' che non c'e' niente
        for i in range(h+1,len(C)):

                # si possono contare i cluster con un solo elemento se si vuole
                #if C[i][1] == []:
                #        print "cluster ", i+1 , "solo centro"
                if C[i][1] != [ ]: #controllo che un cluster non sia solo centro, in quel caso non posso trovare elementi
                        if max_d < C[i][1][0][1]:  
                                max_d = C[i][1][0][1]
                                max_e = C[i][1][0][0]

        #print "nuovo centro", max_e
        #max_e e' l'indice di riga di M in cui si trova il nuovo centro 
        # formo un nuovo cluster con gli elementi piu' vicini a max_e che al centro attuale

        L = [ ]
        for i in range(0,k):
                j = 0
                #stop = False
    
                # scorro la lista dell'i-esimo cluster (fino a che trovo distanze minori (??))
                # j e' l'indice di una coppia [elemento, distanza] nella lista dell'i-esimo cluster
                while (j < len(C[i][1])):
                        e = C[i][1][j][0]
                        old_d = C[i][1][j][1]
                        #print "e =", e, "old_d =", old_d
                        new_d = EDist(M[max_e], M[e])
                        if new_d < old_d:
                                L.append([e,new_d])
                                oldL = C[i][1]
                                oldL.remove([e,old_d])
                                #else:
                                #    stop = True
                        j = j+1

        L = sorted(L, key=getKey, reverse=True)
        #print "cluster" , k+1 , "elementi",  L
        #if k == 1:
        L.remove([max_e, 0.0])
        #print "cluster" , k+1 , "elementi",  L

        C.append([max_e, L])


    # cl memorizza il clustering con gli indici di riga di M
    cl = [ ]

    #il for prende i cluster, estrae i numeri di riga in M

    for k in range(0,K):
        j = 0
        l = [C[k][0]] #mette il centro del cluster nella lista
        while (j < len(C[k][1])):
            l.append(C[k][1][j][0]) #aggiunge tutti gli altri elementi
            j = j+1
        l = sorted(l)
        cl.append(l)
 
    #print cl
    return cl 

# calcola il cluster migliore in base all'indice DB eseguendo il clutering tante volte per lo stesso k e per k = 2,.., 20
#restituice quello con indice piu' basso 
# scl e' la scelta del clustering da usare se =0 allora FPF se = 1 allora KM

def best_clusters(M,scl):
    bcl = [ ] # best clustering
    bDB = 1000 # inizializzazione indice DB

    c1 = [ ] # per calcolare il primo DB uso tutta la matrice in un unico clustser c1
    for i in range(0,len(M)):
        c1.append(i)

    bcc = centroid(c1,M) # centroide di c1

    bcl.append(c1) #clustering con un solo cluster
    #bDB = DB(cc1, bcl, M) # inizializzo best BD al DB del clustering che ha solo un cluster

    #print "k = ", 1 , "DB = ", bDB
    #print "clustering ", c1
    
    for k in range(2,20):
        db = 0.1 
        for j in range(0,1):
            
            if scl == 0:
                cl = FPF(k,M)
            else:
                cl = k_means(M,k)
            cc = centroid_list(cl,M)
            db = DB(cc, cl, M)
            if db < bDB:
                bDB = db
                bcl = cl
                bcc = cc
                
        print "number of cluster = ", len(cl)
        print "DB = ", db
    #print "best cluststering" , bcl
    print "best number of cluster = ", len(bcl)
    print "bestDB = ", bDB
    return (len(bcl),bcl,cc)

############################# FUNZIONI per VERSIONE ITERATIVA ##############

def indici_cluster(C):
    # cl memorizza il clustering con gli indici di riga di M
    cl = [ ]
    K = len(C)
    
    #il for prende i cluster, estrae i numeri di riga in M

    for k in range(0,K):
        j = 0
        l = [C[k][0]] #mette il centro del cluster nella lista
        while (j < len(C[k][1])):
            l.append(C[k][1][j][0]) #aggiunge tutti gli altri elementi
            j = j+1
        l = sorted(l)
        cl.append(l)
 
    #print cl
    return cl 


# restituisce una coppia: il primo e' il numero di cluster, il secondo e' una lista di liste con ogni lista un cluster
def FPFi(M):
    # primo cluster contenente tutto l'insieme di elementi

    m = len(M)
    c = random.randint(0,m-1)
    #print "cluster 1"
    #print "primo centro scelto a caso" , c

    L = [ ]
    C = [ ]

    for i in range(0,m):
        L.append([i, EDist(M[i],M[c])])

    L = sorted(L, key=getKey, reverse=True)

    L.remove([c, 0.0])

    # primo cluster
    C = [ [c, L] ]
    bcl = indici_cluster(C)
    # suo centroide
    bcc = centroid_list(bcl,M)
    # primo best DB
    #bDB = DB(bcc,bcl,M)
    bDB = 100
    

    
    #costruisco gli altri cluster - voglio che si possa fare in maniera incrementale
    # k e' l'effettivo numero di cluster: il primo viene costruito prima del while
    k = 1
    
    while k < 10:
        
        C = iterazione(C,M)
        cl = indici_cluster(C)
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





    
# prende in input un clustering C e aggiunge un nuovo centro e un nuovo cluster
def iterazione(C,M):
    k = len(C)

    #costruisco un nuovo cluster, il K+1 esimo

    
    #cerco l'elemento piu' lontano dal suo centro tra tutti i cluster
    #all'inizio e' il primo della prima (e unica) lista in C
    h = 0 
    while C[h][1] == [ ]and h < len(C):
        h + 1

    max_e = C[h][1][0][0]
    max_d = C[h][1][0][1]

       
    #cerco nelle altre liste (ordinate in senso decrescente), quindi controllo solo il primo
    for i in range(h+1,len(C)):
        # si possono contare i cluster con un solo elemento se si vuole
        #if C[i][1] == []:
        #        print "cluster ", i+1 , "solo centro"
        if C[i][1] != [ ]: #controllo che un cluster non sia solo centro, in quel caso non posso trovare elementi
                    if max_d < C[i][1][0][1]:  
                            max_d = C[i][1][0][1]
                            max_e = C[i][1][0][0]

        
    #max_e e' l'indice di riga di M in cui si trova il nuovo centro 

    # formo un nuovo cluster con gli elementi piu' vicini a max_e che al centro attuale

    L = [ ]
    for i in range(0,k):
            j = 0
            
            # scorro la lista dell'i-esimo cluster (fino a che trovo distanze minori (??))
            # j e' l'indice di una coppia [elemento, distanza] nella lista dell'i-esimo cluster
            while (j < len(C[i][1])):
                    e = C[i][1][j][0]
                    old_d = C[i][1][j][1]
                    new_d = EDist(M[max_e], M[e])
                    if new_d < old_d:
                            L.append([e,new_d])
                            oldL = C[i][1]
                            oldL.remove([e,old_d])
                               
                    j = j+1

    L = sorted(L, key=getKey, reverse=True)
    if L != [ ]:
        L.remove([max_e, 0.0])

    C.append([max_e, L])

    return C

###############################GENERAZIONE RICHIESTE PER SPOTIFY#########################3
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

# calcolo il raggio come la media delle distanze dei punti del cluster dal centroide
# cl e' il cluster (indici della matrice) - c e' il centroide
def raggio_average(cl,c,M):
    r = 0.0
    # distanze di ogni elemento j dal centroide
    for j in range(0,len(cl)):
        r = r + EDist(c,M[cl[j]])
    r = r / len(cl)
    return r

#raggio come la mediana
def raggio_median(cl,c,M):
    r = 0.0
    D = [ ] #lista distanze
    # distanze di ogni elemento j dal centroide
    for j in range(0,len(cl)):
        D.extend(EDist(c,M[cl[j]]))
    D.sorted(D)
    i = len(D)//2
    r = D[i]
    return r

def nuovi_punti(cl,c,M):

    radius = raggio_average(cl,c,M)
    print "raggio" , radius
    center = np.array(c)
    p = sample(center,radius,5)
    print "nuovi punti",  p

    return p

#calcola il raggio (come media delle distanze) e l'indice di riga dell'elemento piu' vicino al centroide
def raggioAV_and_closest(cl,c,M):
    r = 0.0
    minD = EDist(c,M[cl[0]])
    closest = cl[0]
    # distanze di ogni elemento j dal centroide
    for j in range(0,len(cl)):
        dist = EDist(c,M[cl[j]])
        r = r + dist
        if dist < minD:
            minD = dist
            closest = cl[j]
    r = r / len(cl)
    return [r,closest]

#calcola il raggio (come max tra media e mediana delle distanze) e l'indice di riga dell'elemento piu' vicino al centroide
def raggioMAX_and_closest(cl,c,M):
    minD = EDist(c,M[cl[0]]) #inizializzo la distanza minima a quella con il centro
    closest = cl[0] #inizializzo il piu' vicino al centro 
    D = [ minD ] #lista distanze
    R = minD #somma delle distanze
    print D
    # distanze di ogni elemento j dal centroide
    for j in range(1,len(cl)):
        
        dist = EDist(c,M[cl[j]])
        
        D.append(dist)
       
        R = R + dist
        if dist < minD:
            minD = dist
            closest = cl[j]
    D.sort()
    i = (len(D)//3)*2
    R = R / len(D)
    if R > D[i]:
        print "media"
        return [R,closest]
    else:
        print "mediana"
        return [D[i], closest]

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


# dato un punto p preso a caso nella sfera e un cluster cl, restituisce il punto del cluster piu' vicino  

def closest_to_point(cl,p,M):
    minD =  EDist(p,M[cl[0]])#inizializzo la distanza minima a quella con il centro
    closest = cl[0] #inizializzo il piu' vicino al centro
    for i in range(0,len(cl)):
        dist = EDist(p,M[cl[i]])
        if dist < minD:
            minD = dist
            closest = cl[i]
    return closest
    
# trova le nuove richeste da fare a spotify - chiede in input M, un clustering CL su M, e centroidi di CL e il file dove srvirere le richieste da fare (feature dei# punti da chiedere) sr = scelta raggio; se e' =0 allora usa la mediana/media se e' =1 allora usa la massima distanza da centroide;
# scl e' la scelta del clustering da usare se e' = 0 allora e' FPF se e' =1 allora e' k-means
def genera_richieste(M, CL, centroidi, fileprefix, scl, sr):

    #apre il file dove cerchera' i seed delle richieste
    seedfile = fileprefix +'.csv'
    dataframe = pd.read_csv(seedfile, sep = ';' ) # "ISO-8859-1")
    df = dataframe.copy(deep=True)
    # in S ci sono i valori di seed da leggere
    S = df.values
    print 'lunghezza S', len(S)
    
    #in outfile verranno scritte le richieste da fare a spotify : c'e' la riga di intestazione, poi coppie di righe seed + feature a caso
    # prima mette nel nome del file il clustering che usa
    if scl == 0:
        fileprefix = fileprefix + "_FPF"
    else:
        fileprefix = fileprefix + "_KM"

    #poi aggiunge il numero di cluster
    fileprefix = fileprefix + '_' + str(CL[0])
    
    # controlla quale raggio deve usare e lo scrive nel nome del file di output
    if sr == 0:
        fileprefix = fileprefix + "_MEDIA"
    else:
        fileprefix = fileprefix + "_MAX"
   
    outfile = fileprefix + "_richieste.csv"
    outfile = open(outfile, 'w')

    # scrive la prima riga di intestazione 
    #prima_riga = "track_id;track_name;duration_ms;album_name;album_type;artist_name;track_popularity;explicit;artist_genres;album_genres;acousticness;danceability;energy;instrumentalness;key;liveness;loudness;mode;speechness;tempo;time_signature;valence;played_at;date;datetime"
    prima_riga = "track_id;acousticness;danceability;energy;instrumentalness;key;liveness;loudness;mode;speechness;tempo;time_signature;valence;played_at;date;datetime"
    outfile.write(prima_riga + '\n')

    #per ogni cluster genera 5 richieste
    for i in range(0,len(CL[1])):
        print "ciclo i" , i 
        if sr == 0:
            RandC = raggioMAX_and_closest(CL[1][i],centroidi[i],M)
        else:
            RandC = raggioMAX(CL[1][i],centroidi[i],M)
        radius = RandC[0]
        #closest = RandC[1]
        #NewP.extend(M[closest])
        #print "indie di riga", closest
        #print "riga", M[closest]

        #filename2 e' il file csv in cui va a cercare il seed completo che copia in seedfile
        #filename2 = filesuffix + '.csv'
        #seed = find_seed(filename2, closest)
        

        #genera 5 punti a caso nella sfera con centro il centroide e raggio quello che e' --> f matrice con un punto per riga
        center = np.array(centroidi[i])
        p = sample(center,radius,100)
        f = np.array(p).tolist()
        #controllo che i valori siano nei range, altrimenti li modifico accordingly
        # conto quanti punti diversi trovo con track_id diversi --> np fino a 5
        num_track = 50
        pp = 0
        totp = int(num_track/(4*CL[0])) +1
        # tengo lista track_id
        tr = [ ]
        j = 0
        print "ciclo i" , i
        
        while (j < len(f) and pp <= totp ):
            
            print 'punto causale numero', j, 'ciclo ' , i 
            #s = ''
            if scl == 0:
                #print i, CL[1][i]
                #print j, f[j]
                
                seed = S[closest_to_point(CL[1][i],f[j],M)-1] #capire!!!! perche' in S sono scalata di uno indietro
            if scl == 1: 
                print len(CL[1])
                print "ciclo i dopo", i
                
                cl = CL[1][i] #cluster corrente
                minD =  EDist(f[j],M[cl[0]]) #inizializzo la distanza minima a quella del primo elemento 
                closest = cl[0] #inizializzo il piu' vicino al primo elemento 
                for h in range(1,len(cl)):
                    dist = EDist(f[j],M[cl[h]])
                    if dist < minD:
                        minD = dist
                        closest = cl[h]
                seed = S[closest -1]
            
            track_id = seed[0]
            if not (track_id in tr):
                pp = pp + 1
                tr.append(track_id)
                NewP = genera_punto(f[j], seed[19])
              
            #print "punto piu' vicino", S[closest_to_point(CL[1][i],f[j],M)] 
                riga = str(track_id)  + ";"
                for item in NewP:
                    riga = riga + str(item) + ";"
                outfile.write(riga  + '\n')
            j = j +1
    #filename e' il file in cui va a scrivere le richieste 
    #filename = filesuffix + 'richieste.txt'
    
    outfile.close()

def genera_punto(f,seed):
    #tutti quelli che sono compresi tra 0 e 1 
    lista = [0,1,2,3,5,8,11]
    for t in lista:
        if f[t] > 1:
            f[t] = 1
        elif f[t] < 0:
            f[t] = 0
    # il 4 in (0..11)
    f[4] = int(round(f[4]))
    if f[4] < 0:
        f[4] = 0
    elif f[4] > 11:
        f[4] = 11
    # il 6 in [-60,0]
    if f[6] < -60:
        f[6] = -60
    elif f[6] > 0:
        f[6] = 0
    # il 7 in {0,1}
    if f[7] >= 0.5:
        f[7] = 1
    else:
        f[7] = 0
    # il 9 vicino al valore originale - si puo' aggiungere o togliere a random 10 se e' vero
    if abs(f[9] - seed) > 10:
        f[9] = seed
    # il 10 in {3,4,5}
    f[10] = int(round(f[10]))
    if f[10] < 3:
        f[10] = 3
    elif f[10] > 4:
        f[10] = 4
    return f


    
################################TROVA IL SEED_TRACK###########################
#la funzione raggio*_and_closets restituisce l'indice di riga della canzone piu' vicina al cenroide. Da li' bisogna
#recuperare l'id della track
# Bisogna andare a prendere la riga corrispondente nel CVS - la riga e' line -> l 

def find_seed(filename, l):
    
    #dataframe = pd.read_csv( filename, encoding = "cp1252", sep = ';' ) # "ISO-8859-1")
    dataframe = pd.read_csv( filename, sep = ';' ) # "ISO-8859-1")
    df = dataframe.copy(deep=True)
    M = df.values

    print "seed" , M[l]
    return M[l]
        

################################K-means#######################################
# eseguo k-means e genero clustering compatibile con il resto. Al momento il numero di cluster lo devo dare io.

def k_means(M,k):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(M)
    # lista con le label di cluster di ogni signola riga di M 
    K = kmeans.labels_
    #print K
    # costruisco il clustering come serve a me
    L = [ ] 
    for i in range(0,k):
        L.append([ ])
    for i in range(0,len(K)):
        L[K[i]].append(i)
    return [k,L]

# prende la matrice M e calcola il miglior numero di cluster usando DB
def best_k_means(M):
    # primo cluster contenente tutto l'insieme di elementi

    m = len(M)
    #c = random.randint(0,m-1)
    #print "cluster 1"
    #print "primo centro scelto a caso" , c

    #L = [ ]
    #for i in range(0,m):
    #    L.append(i)
    
    #C = [ [1,L] ]

    #for i in range(0,m):
    #    L.append([i, EDist(M[i],M[c])])

    #L = sorted(L, key=getKey, reverse=True)

    #L.remove([c, 0.0])

    # primo cluster
    #C = [ [c, L] ]
    #bcl = indici_cluster(C)
    # suo centroide
    #bcc = centroid_list(bcl,M)
    # primo best DB
    #bDB = DB(bcc,bcl,M)
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

    
############################## Per CHIAMARE FPF #########################################

# M e' una matrice 2x2 che contiene (nelle righe) i vettori da clusterizzare
# K e' il numero di cluster finali, k quelli calcolati fino ad un certo momento

def richieste_FPF(M,fileprefix):
    

        print "versione incrementale"

        CL =  FPFi(M)
        #print CL
        centroidi = centroid_list(CL[1],M)

        #rfile e' il file di richieste --> output
        #rfile = item +'_richieste.txt'
        #filename = "richieste_pomeriggio.txt"

        
        # usa FPF e MEDIA
        #genera_richieste(M, CL, centroidi, fileprefix, 0, 0)

        # usa FPF e MAX
        genera_richieste(M, CL, centroidi, fileprefix, 0, 1)

        # usa FPF e metodo Jessica
        richieste_Jessica(M,CL,centroidi, fileprefix, 0)

########################### GENERA RICHIESTE PER K-means #####################################

# genera le richieste per k-menas con k cluster se k = 0 cerca il meglio, altrimenti usa quel numero di cluster
def richieste_KMEANS(M,k,fileprefix):
    
    
    # restitutisce un clustering come negli altri casi
    if k == 0:
        CL = best_k_means(M)
    else:
        CL = k_means(M,k)
    #print L 
    centroidi = centroid_list(CL[1],M)

    richieste_Jessica(M,CL,centroidi, fileprefix, 1)
    
    # usa k-means e MEDIA
    #genera_richieste(M, CL, centroidi, fileprefix,1, 0)

    # usa k-means e MAX
    genera_richieste(M, CL, centroidi, fileprefix ,1, 1)




##############################METODO JESSICA ###########################
#parto da clustering e numero di cluster. Ordino le distanze dal centroide in ordine crescente.
#Divido ogni cluster in 4 segmenti uguali e prendo come seed la canzone che sta a quella distanza 
# scl = 0 se ho usato FPF = 1 se ho usato k-means per produrre il clustering

def richieste_Jessica(M, CL, centroidi, fileprefix, scl):
    #apre il file dove cerchera' i seed delle richieste
    seedfile = fileprefix +'.csv'
    dataframe = pd.read_csv(seedfile, sep = ';' ) # "ISO-8859-1")
    df = dataframe.copy(deep=True)
    # in S ci sono i valori di seed da leggere
    S = df.values

    # apre il file di output in base al numero di cluster e algoritmo usato
    k = CL[0]
    if scl == 0:
        outfile = fileprefix + "_FPF_" + str(k) + "_J.csv"
    else:
        outfile = fileprefix + "_KM_" + str(k) + "_J.csv"

    outfile = open(outfile, 'w')
    # riga di intestazione nel file
    prima_riga = "track_id;acousticness;danceability;energy;instrumentalness;key;liveness;loudness;mode;speechness;tempo;time_signature;valence;played_at;date;datetime"
    outfile.write(prima_riga + '\n')
        
    
    cl = CL[1]
    #per ogni clustesr
    for i in range(0, len(cl)):
        c = []
        #per ogni elemento nel cluster
        for j in range(0,len(cl[i])):
            c.append([cl[i][j],EDist(M[j],centroidi[i])])

        # ordino in base alla seconda chiave in ordine crescente
        c = sorted(c, key=getKey)

        
        
        # metto nel file la prima richiesta: track_id del piu' vicino, features del centroide
        track_id = S[c[0][0] -1][0]
        riga = track_id + ";"
        for j in range(0,len(centroidi[i])):
            riga = riga + str(centroidi[i][j]) + ";"
        outfile.write(riga  + '\n')
        
            
        # costruisco lista con indici dei seed
        ls = [ ]
        p = len(c)//3
        ls.append(p)
        p = (len(c)//3)*2
        ls.append(p)
        ls.append(len(c)-1)

        print "lunghezza c", len(c),
        print "lista indici richieste", ls

        # prendo gli indci di M in cui si trovano le richieste da fare
        for item in ls:
            track_id = S[c[item][0]-1][0]
            riga = track_id + ";"
            for j in range(10,22):
                riga = riga + str(S[c[item][0]-1][j]) + ";"
            outfile.write(riga  + '\n')

    outfile.close()


############################## GENERA MATRICE DI INPUT DA FILE CSV ######################

def leggi_file(filename):
        M = []
        
        with open(filename) as f:
            TMP = f.read().splitlines( )

        #print TMP

        for i in range(0,len(TMP)):

            
            rigaTMP = TMP[i].split(";")
            l = []
            for j in range(10,22):
                l.append(float(rigaTMP[j]))
            M.append(l)
        return M

################################# MAIN ##################################

file_list = []

for i in range(0,24):
        filename = 'csv_' + str(i)
        file_list.append(filename)


    
#file_list = ['mattina', 'pomeriggio', 'sera']

#file_list = ['mattina']

for item in file_list:
        
        
        #inputfile e' il file che va a leggere
        inputfile = item + '.csv'
    
        #M = np.genfromtxt(inputfile)
        M = leggi_file(inputfile)
        
        print "file letto" , str(inputfile)
        print M
        
        richieste_FPF(M,item)

        richieste_KMEANS(M,0,item)

