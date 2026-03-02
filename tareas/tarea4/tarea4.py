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
   modelo = pulp.LpProblem("Tarea 4 - Escenario {nombre}", pulp.LpMinimize)
   modelo += ( pulp.lpSum( c[i]*x[i] for i in I ) + (rho * exceso), "Función objetivo" )

   # Paso 3: restricciones
   modelo += ( pulp.lpSum( k[i]*x[i] for i in I) >= min_capacidad, "Capacidad mínima" )
   modelo += ( pulp.lpSum( x[i] for i in I) >= max_propiedades, "Límite de reservaciones" )
   modelo += ( pulp.lpSum( c[i]*x[i] for i in I) + deficit - exceso == meta_presupuesto, 
               "Presupuesto")

   # Ahora sí, resolvemos
   modelo.solve(pulp.PULP_CBC_CMD(msg=0))
   
   # Revisamos si se resolvió el modelo
   status = pulp.LpStatus[modelo.status]
   print(f"Problema resuelto, status: {status}")
   if status != "Optimal":
      print("No se encontro solucion optima")
      return None
   
   # Si sí, extraemos los resultados
   seleccionados_indices = [i for i in I if x[i].varValue > 0.5]
   resultado_df = candidatos.iloc[seleccionados_indices].copy()
   costo_total = sum(c[i] * x[i].varValue for i in I)
   exceso_resultante = exceso.varValue
   deficit_resultante = deficit.varValue
   z_optimo = pulp.value(modelo.objective)

   # Presentamos resultados
   print(f"\n  {'-'*60}")
   print(f"  RESULTADOS")
   print(f"  {'-'*60}")
   print(f"  Valor óptimo Z = ${z_optimo:,.2f}")
   print(f"  Costo total real = ${costo_total:,.2f}")
   print(f"  Presupuesto meta = ${meta_presupuesto:,.0f}")
   print(f"  Desviación positiva (d⁺, exceso) = ${exceso_resultante:,.2f}")
   print(f"  Desviación negativa (d⁻, deficit) = ${deficit_resultante:,.2f}")
   print(f"  Número de inmuebles seleccionados: {len(seleccionados_indices)}")
   print(f"  Capacidad total: {resultado_df['accommodates'].sum():.0f} personas")
   print(f"  Rating promedio: {resultado_df['review_scores_rating'].mean():.2f}")
   
   print(f"\n  {'-'*60}")
   print(f"  INMUEBLES SELECCIONADOS")
   print(f"  {'-'*60}")
   
   for indice, (_, row) in enumerate(resultado_df.iterrows(), 1):
      print(f"\n  {indice}. {row['name'][:60]}")
      print(f"     URL: {row['listing_url']}")
      print(f"     Precio: ${row['precio']:,.0f} MXN/noche")
      print(f"     Capacidad: {row['accommodates']:.0f} personas")
      print(f"     Rating: {row['review_scores_rating']:.2f}")
   
   return {
      'estado': status,
      'z_optimo': z_optimo,
      'costo_total': costo_total,
      'presupuesto_meta': meta_presupuesto,
      'd_plus': exceso_resultante,
      'd_minus': deficit_resultante,
      'num_propiedades': len(seleccionados_indices),
      'capacidad_total': resultado_df['accommodates'].sum(),
      'rating_promedio': resultado_df['review_scores_rating'].mean(),
      'propiedades': resultado_df,
      'modelo': modelo,
      'candidatas': candidatos
   }


# ======================================================================
# ESCENARIO BASE
# ======================================================================
df = cargar_limpiar_datos('listings.csv')
resultado_base = resolver_problema(
   df,
   presupuesto=70000,
   propiedades=12,
   capacidad=50,
   calificacion=4.5,
   penalizacion_exceso_presupuesto=2.0,
   nombre="Base"
)

# ======================================================================
# ESCENARIO 1: AUSTERIDAD
# Presupuesto $40,000 MXN, máximo 8 casas
# ======================================================================
resultado_austeridad = resolver_problema(
   df,
   presupuesto=40000,
   propiedades=8,
   capacidad=50,
   calificacion=4.5,
   penalizacion_exceso_presupuesto=2.0,
   nombre="Austeridad"
)

# ======================================================================
# ESCENARIO 2: INFLUENCER
# Rating mínimo 4.8
# ======================================================================
resultado_influencer = resolver_problema(
   df,
   presupuesto=70000,
   propiedades=12,
   capacidad=50,
   calificacion=4.8,
   penalizacion_exceso_presupuesto=2.0,
   nombre="Influencer"
)

# ======================================================================
# ESCENARIO 3: EXIGENTE
# Rating mínimo 4.7 en TODOS los aspectos
# ======================================================================
resultado_exigente = resolver_problema(
   df,
   presupuesto=70000,
   propiedades=12,
   capacidad=50,
   calificacion=4.7,
   penalizacion_exceso_presupuesto=2.0,
   nombre="Exigente"
)

# ======================================================================
# ESCENARIO 4: UNIDOS COMO FAMILIA
# Máximo 3 propiedades
# ======================================================================
resultado_familia = resolver_problema(
   df,
   presupuesto=70000,
   propiedades=3,
   capacidad=50,
   calificacion=4.5,
   penalizacion_exceso_presupuesto=2.0,
   nombre="Unidos"
)



