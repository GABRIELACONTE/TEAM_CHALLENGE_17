
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# Obtener la ruta correcta del dataset
ruta_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ruta_data = os.path.join(ruta_proyecto, "src/data")
ruta_csv = os.path.join(ruta_data, "ds_salaries.csv")

# Cargar el dataset una sola vez
df = pd.read_csv(ruta_csv)

# Selección de columnas relevantes
df = df[["work_year", "experience_level", "employment_type", "job_title", 
         "salary_in_usd", "company_location", "company_size"]]

# Definir variables categóricas y numéricas antes de procesarlas
num_features = ["work_year"]  # ✅ Eliminar "salary_in_usd" y revisar "remote_ratio"
cat_features = ["experience_level", "employment_type", "job_title", 
                "company_location", "company_size"]

# Convertir variables categóricas en tipo 'category'
df[cat_features] = df[cat_features].astype("category")

# Separar en variables predictoras (X) y objetivo (y)
X = df.drop(columns=["salary_in_usd"])  # ❌ NO incluir salary_in_usd en `X`
y = df["salary_in_usd"]

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")

# Definir transformaciones
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),  
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features)  
])

# Modelos y parámetros para GridSearchCV
models_params = {
    "LinearRegression": {"model": [LinearRegression()]},
    "RandomForest": {
        "model": [RandomForestRegressor(random_state=42)],
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20]
    },
    "GradientBoosting": {
        "model": [GradientBoostingRegressor(random_state=42)],
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2]
    },
    "MLPRegressor": {
        "model": [MLPRegressor(max_iter=1000, random_state=42)],
        "model__hidden_layer_sizes": [(64,), (64, 32)],
        "model__alpha": [0.0001, 0.01]
    }
}

# Entrenar modelos
results = []

for model_name, param_grid in models_params.items():
    print(f"\n🔹 Training model: {model_name} with GridSearchCV")
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", param_grid["model"][0])  
    ])

    # GridSearch con validación cruzada
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Mejor modelo encontrado
    best_model = grid_search.best_estimator_

    # Guardar resultados
    results.append({
        "Model": model_name,
        "Best Score": grid_search.best_score_,
        "Best Params": grid_search.best_params_
    })

    # Guardar el modelo entrenado
    model_path = os.path.join(ruta_proyecto, f"models/{model_name}.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    
    print(f"✅ Best {model_name} model saved at {model_path}")



df_results = pd.DataFrame(results).sort_values(by="Best Score", ascending=False)

# Mostrar la tabla ordenada
print("\n📊 Model Comparison (Sorted by R² Score):")
print(df_results)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar el mejor modelo
best_model_name = df_results.iloc[0]["Model"]
best_model_path = os.path.join(ruta_proyecto, f"models/{best_model_name}.pkl")
best_model = joblib.load(best_model_path)

# Hacer predicciones en el conjunto de prueba
y_pred = best_model.predict(X_test)

# Evaluar métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n🏆 Best Model: {best_model_name}")
print(f"🔹 MAE: {mae:.2f}")
print(f"🔹 MSE: {mse:.2f}")
print(f"🔹 R² Score: {r2:.3f}")