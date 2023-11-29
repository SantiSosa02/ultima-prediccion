import json
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys


def realizar_prediccion(productos):
    # Convertir la lista de productos a un DataFrame de Pandas
    df_productos = pd.DataFrame(productos)

    # Agregar una nueva columna 'etiqueta' basada en la condición especificada
    df_productos['etiqueta'] = df_productos.apply(lambda row: 'comprar' if row['cantidad'] < 5 and row['estado'] else 'no comprar', axis=1)

    # Filtrar solo los productos con estado igual a true
    df_productos = df_productos[df_productos['estado']]

    # Seleccionar características específicas
    columnas_caracteristicas = ['idproducto', 'idcategoria', 'nombre', 'stock_minimo', 'cantidad', 'precio_venta', 'estado']
    X_productos = df_productos[columnas_caracteristicas]

    # Vectorizar las características
    vectorizer = DictVectorizer(sparse=False)
    X_productos = vectorizer.fit_transform(X_productos.to_dict(orient='records'))

    # Entrenar el modelo
    modelo = MultinomialNB()
    modelo.fit(X_productos, df_productos['etiqueta'].tolist())

    # Realizar la predicción para todos los productos
    predicciones = modelo.predict(X_productos)

    # Obtener los nombres de los productos
    nombres_productos = df_productos['nombre'].tolist()

    # Devolver un diccionario con las predicciones y nombres de productos
    resultados = {'predicciones': [{'prediccion': p, 'nombre_producto': n} for p, n in zip(predicciones, nombres_productos)]}
    
    return resultados

# Parsear el contenido del archivo JSON
productos = json.loads(contenido_archivo)

# Llamar a la función de predicción con los productos
resultado_prediccion = realizar_prediccion(productos)

# Devolver la respuesta como JSON
print(json.dumps(resultado_prediccion))
