import pandas
import pulp


# Tenemos apartadas las columnas de detalles de calificación porque uno de
# nuestros escenarios requiere verlas todas. Estas variables indican detalles
# de la calificación: veracidad (poner datos falsos en el inmueble reduce esa
# calificación), limpieza, fácil check-in, comunicación con el encargado,
# qué tan bien ubicado está, y si vale el dinero
columnas_detalles_calificacion = \
[
   "review_scores_accuracy",
   "review_scores_cleanliness",
   "review_scores_checkin",
   "review_scores_communication",
   "review_scores_location",
   "review_scores_value"
]


def cargar_limpiar_datos(path):
   # Leemos el CSV
   dataframe = pandas.read_csv(path, on_bad_lines="skip")

   # Convertimos el precio de texto a punto flotante
   dataframe["price"] = dataframe["price"].str.replace("$", "", regex=False)
   dataframe["price"] = dataframe["price"].str.replace(",", "", regex=False)
   dataframe["price"] = dataframe["price"].astype(float)
   
   # Tumbamos todos los registros que no tengan al menos los 3 esenciales:
   # precio, capacidad y calificación
   columnas_relevantes = \
   [
      "price",
      "accommodates",
      "review_scores_rating"
   ]
   dataframe = dataframe.dropna(subset=columnas_relevantes)
   
   # Reseteamos el índice y tiramos el dataframe
   dataframe = dataframe.reset_index(drop=True)
   return dataframe



def resolver_problema(df, **kwargs):
   meta_presupuesto = kwargs["presupuesto"]
   max_propiedades = kwargs["propiedades"]
   min_capacidad = kwargs["capacidad"]
   min_calificacion = kwargs["calificacion"]
   rho = kwargs["penalizacion_exceso_presupuesto"]
   nombre = kwargs["nombre"]
   
   # Primero: la restricción de que la calificación deba ser 4.5 o más podría
   # añadirse al modelo, pero va a ser más práctico implementar esa restricción
   # directamente en el dataframe.
   candidatos = df[ df["review_scores_rating"] >= min_calificacion ].copy()

   # Caso especial: si el escenario es Exigente, aplicamos ese filtro también
   # a todas las columnas de calificación
   if nombre == "Exigente":
      for una_columna in columnas_detalles_calificacion:
         candidatos = candidatos[ candidatos[una_columna] >= min_calificacion ]
   
   # Actualizamos índice
   candidatos = candidatos.reset_index(drop=True)

   # Pruebas de cordura: determinamos a priori si de entrada nuestro juego de
   # datos tiene suficiente capacidad. Si nos quedamos sin inmuebles o si
   # en todo el conjunto de inmuebles no hay suficiente capacidad, desistimos
   if len(candidatos) == 0:
      print("Nos quedamos sin inmuebles, no podemos proceder")
      return None
   elif candidatos["accommodates"].sum() < min_capacidad:
      print("Nos quedamos sin suficientes inmuebles para alojar a todos")
      return None
   
   # Ahora sí, procedemos al modelo.
   # Paso 1: planteamiento matemático
   # Conjuntos e índices
   I = range(len(candidatos)) # I = el conjunto de todos los inmuebles
   # Variables de decisión
   x = pulp.LpVariable.dicts("x", I, cat="Binary") # x = ¿se eligió el inmueble?
   # Parámetros
   c = candidatos["price"].values # c = costo por noche de cada inmueble
   k = candidatos["accommodates"].values # k = kapacidad de cada inmueble
   q = candidatos["review_scores_rating"].values # q = quality de c/inmueble
   # Variables de desviación
   exceso = pulp.LpVariable("dmas", lowBound=0, cat="Continuous")
   deficit = pulp.LpVariable("dmenos", lowBound=0, cat="Continuous")
   # Variables de penalización
   rho = 1  # rho = penalización del exceso presupuestario

   # Paso 2: función objetivo
   




