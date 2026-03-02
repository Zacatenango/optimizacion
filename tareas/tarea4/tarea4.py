import pandas
import pulp
import haversine

def cargar_limpiar_datos(path):
   dataframe = pandas.read_csv(path, on_bad_lines="skip")
   dataframe["price"] = dataframe["price"].str.replace("$", "", regex=False)
   dataframe["price"] = dataframe["price"].str.replace(",", "", regex=False)
   dataframe["price"] = dataframe["price"].astype(float)
   columnas_relevantes = [ "price", "accommodates", "review_scores_rating" ]
   dataframe = dataframe.dropna(subset=columnas_relevantes)
   dataframe = dataframe.reset_index(drop=True)
