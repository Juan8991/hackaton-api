import pandas as pd

df_data = None

def load_data():
    global df_data
    df_data = pd.read_csv('app/data/jobs_in_data.csv')
    df_data = df_data.drop_duplicates().reset_index(drop=True)
def getDataFrameOriginal():
    global df_data
    return df_data