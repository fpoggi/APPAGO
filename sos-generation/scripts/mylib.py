import pandas as pd


# loads csv data into a dataframe.
# filters songs by size (length >= min_ms_played)
# computes new columns, i.e. date, time and time-hour
def loadData(filename, min_msPlayed):
  df = pd.read_csv(filename, delimiter=";", encoding="UTF-8") #"ISO-8859-1")  
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
