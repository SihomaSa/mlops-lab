# 🔧 Script de Producción del MLOps Engineer
import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import yaml
import logging
from mlflow.tracking import MlflowClient

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Argumentos de línea de comandos configurables."""
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()

def get_model_instance(name, params):
    """Factory para instanciar modelos desde configuración."""
    model_map = {
        'LinearRegression': LinearRegression,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'HistGradientBoosting': HistGradientBoostingRegressor,
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)

def load_and_split_data(data_path, config):
    """Carga y divide datos usando configuración."""
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    # Usar características seleccionadas de la experimentación
    selected_features = config['model']['feature_sets']['rfe']
    logger.info(f"Using {len(selected_features)} selected features")

    X = data[selected_features]
    y = data[config['model']['target_variable']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """Entrena y evalúa el modelo."""
    logger.info("Training model...")
    model.fit(X_train, y_train)

    # Predicciones y métricas
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    metrics = {
        'mae': mae,
        'r2': r2,
        'rmse': rmse
    }

    logger.info(f"Model performance - R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return model, metrics

def save_model_artifacts(model, config, metrics, models_dir):
    """Guarda modelo y artefactos."""
    import os
    os.makedirs(models_dir, exist_ok=True)

    # Guardar modelo entrenado
    model_path = os.path.join(models_dir, 'trained/final_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Actualizar configuración con métricas finales
    config['model']['final_metrics'] = {
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2']),
        'rmse': float(metrics['rmse'])
    }

    # Guardar configuración actualizada
    config_path = os.path.join(models_dir, 'trained/house_price_model.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    logger.info(f"✅ Model saved to {model_path}")
    logger.info(f"✅ Config saved to {config_path}")

    return model_path, config_path

def register_model_in_mlflow(model, config, metrics, mlflow_tracking_uri):
    """Registra modelo en MLflow con metadatos completos."""
    if not mlflow_tracking_uri:
        logger.info("No MLflow tracking URI provided, skipping MLflow logging")
        return None

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("House Price Prediction - Production")

    model_name = config['model']['name']

    with mlflow.start_run(run_name="production_training"):
        # Log parámetros del modelo
        mlflow.log_params(config['model']['parameters'])

        # Log métricas de rendimiento
        mlflow.log_metrics(metrics)

        # Log información adicional
        mlflow.log_param("selected_features_count",
                        config['model']['feature_sets']['selected_features_count'])
        mlflow.log_param("feature_selection_method",
                        config['model']['feature_sets']['rfe_method'])

        # Registrar modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=mlflow.models.infer_signature(None, None)
        )

        # Transicionar a producción si las métricas son buenas
        if metrics['r2'] > 0.8:  # Threshold configurable
            client = MlflowClient()
            model_version = client.get_latest_versions(model_name, stages=["None"])[0]
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            logger.info(f"✅ Model promoted to Production stage in MLflow")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ Model logged to MLflow with run_id: {run_id}")
        return run_id

def main():
    """Pipeline principal de entrenamiento."""
    args = parse_args()

    # Cargar configuración del modelo
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Cargar y preparar datos
    X_train, X_test, y_train, y_test = load_and_split_data(args.data, config)

    # Instanciar modelo desde configuración
    model_name = config['model']['best_model']
    model_params = config['model']['parameters']
    model = get_model_instance(model_name, model_params)
    logger.info(f"Created {model_name} model with parameters: {model_params}")

    # Entrenar y evaluar
    trained_model, metrics = train_and_evaluate_model(
        model, X_train, y_train, X_test, y_test
    )

    # Guardar artefactos localmente
    model_path, config_path = save_model_artifacts(
        trained_model, config, metrics, args.models_dir
    )

    # Registrar en MLflow
    run_id = register_model_in_mlflow(
        trained_model, config, metrics, args.mlflow_tracking_uri
    )

    logger.info("🚀 Training pipeline completed successfully!")

    return {
        'model_path': model_path,
        'config_path': config_path,
        'metrics': metrics,
        'mlflow_run_id': run_id
    }

if __name__ == "__main__":
    main()