import argparse
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


""" 
Este código te permite cargar un modelo entrenado y un normalizador guardado desde archivos pickle y realizar 
predicciones en nuevos datos de entrada especificados en un archivo CSV. Puedes ejecutar este script desde la 
terminal proporcionando las rutas a los archivos del modelo, el normalizador y los datos de entrada. Por ejemplo:
python main.py --modelo modelo_entrenado.pkl --normalizador normalizador.pkl --entrada datos_entrada.csv
"""
# Función para cargar el modelo y el normalizador
def cargar_modelo_y_normalizador(modelo_path, normalizador_path):
    with open(modelo_path, 'rb') as model_file:
        modelo = pickle.load(model_file)
    
    with open(normalizador_path, 'rb') as scaler_file:
        normalizador = pickle.load(scaler_file)
    
    return modelo, normalizador

# Función para realizar predicciones
def predecir(modelo, normalizador, datos_entrada):
    # Aplica la transformación de normalización a los datos de entrada
    datos_normalizados = normalizador.transform(datos_entrada)
    
    # Realiza predicciones
    predicciones = modelo.predict(datos_normalizados)
    
    return predicciones

if __name__ == "__main__":
    # Configurar argumentos desde la terminal
    parser = argparse.ArgumentParser(description="Realizar predicciones con un modelo entrenado.")
    parser.add_argument("--modelo", required=True, help="Ruta al archivo del modelo entrenado (pickle).")
    parser.add_argument("--normalizador", required=True, help="Ruta al archivo del normalizador (pickle).")
    parser.add_argument("--entrada", required=True, help="Ruta al archivo CSV de entrada para predicciones.")
    args = parser.parse_args()

    # Cargar el modelo y el normalizador
    modelo, normalizador = cargar_modelo_y_normalizador(args.modelo, args.normalizador)

    # Cargar datos de entrada para predicciones desde un archivo CSV
    datos_entrada = pd.read_csv(args.entrada)

    # Realizar predicciones
    predicciones = predecir(modelo, normalizador, datos_entrada)

    # Imprimir las predicciones
    print("Predicciones:")
    print(predicciones)