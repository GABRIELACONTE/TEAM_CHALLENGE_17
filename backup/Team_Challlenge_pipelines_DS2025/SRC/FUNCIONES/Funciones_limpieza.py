import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os


def load_data(file_name="ds_salaries.csv"):
    """
    Carga el dataset 

    Devuelve: df cargado en un DataFrame
    """
    file_path = os.path.join(os.path.dirname(__file__), "../data", file_name)

    try:
        df = pd.read_csv(file_path)
        print(
            f"✅ Archivo '{file_name}' cargado correctamente. Dimensiones: {df.shape}")
        return df
    except FileNotFoundError:
        print(
            f"❌ Error: El archivo '{file_name}' no se encontró en '{file_path}'.")
        return None


def clean_data(df):
    """
    Realiza la limpieza del dataset: Elimina valores nulos / Elimina duplicados / Normaliza los nombres de las columnas.

   Devuelvee:  Dataset limpio.
    """
    if df is None:
        print("❌ Error: No se puede limpiar un DataFrame vacío.")
        return None

    # Elimina duplicados
    df = df.drop_duplicates()
    print(f"🔹 Se han eliminado duplicados. Dimensiones actuales: {df.shape}")

    # Elimina valores nulos
    df = df.dropna()
    print(
        f"🔹 Se han eliminado valores nulos. Dimensiones actuales: {df.shape}")

    # Normaliza nombres de columnas (Convertir a minúsculas y reemplazar espacios con "_")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    print("🔹 Los nombres de las columnas han sido normalizados.")

    return df
