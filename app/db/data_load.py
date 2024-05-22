import pandas as pd

df_data = None

def load_data():
    global df_data
    df_data = pd.read_csv('app/data/jobs_in_data.csv')
    df_data = df_data.drop_duplicates().reset_index(drop=True)
def getDataFrameOriginal():
    global df_data
    if df_data is None:
        load_data()  # Cargar los datos si aún no se han cargado
    return df_data