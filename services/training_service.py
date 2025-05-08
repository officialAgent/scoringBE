from config import DATA_PATH, MODEL_DIR, PLOTS_DIR
from models import train_and_save_model, generate_model_comparison_df
from utils.file_utils import (
    save_or_update_model,
    save_performance_metrics,
    get_existing_performance_df,
    activate_best_model,
    generate_filtered_roc_curve,
    generate_filtered_comparison_plot,
    save_visualization_record,
    get_all_trained_model_names
)
import pandas as pd
import os
from flask import jsonify

def train_models(data):
    models = data.get('models', [])
    if isinstance(models, str):
        models = [models]

    if not models:
        return jsonify({"error": "No models provided"}), 400

    models = [m for m in models if m in [
        'xgboost', 'random_forest', 'logistic_regression',
        'neural_net', 'catboost', 'svm', 'knn', 'naive_bayes'
    ]]

    if not models:
        return jsonify({"error": "No valid models provided"}), 400

    try:
        for model_name in models:
            train_and_save_model(DATA_PATH, MODEL_DIR, model_name)
            save_or_update_model(model_name)

        new_df = generate_model_comparison_df(DATA_PATH, MODEL_DIR, only=models)

        existing_df = get_existing_performance_df()
        df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset='Model', keep='last')

        activate_best_model(df)
        save_performance_metrics(new_df)

        roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
        comparison_path = os.path.join(PLOTS_DIR, "model_comparison.png")

        all_models = get_all_trained_model_names()
        generate_filtered_roc_curve(DATA_PATH, MODEL_DIR, all_models, roc_path)
        save_visualization_record('roc_curve', roc_path, all_models)

        generate_filtered_comparison_plot(df, comparison_path)
        save_visualization_record('model_comparison', comparison_path, all_models)

        return jsonify({"status": "success", "message": "Training and visualization complete."}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
