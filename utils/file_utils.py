import os
import pandas as pd
from database.session import get_session
from database.models import TrainedModel, ModelPerformance, ModelVisualization
from models import preprocess_data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

def save_or_update_model(model_name):
    session = get_session()
    model_path = os.path.join("models", f"{model_name}.pkl" if model_name != 'neural_net' else f"{model_name}.h5")

    existing = session.query(TrainedModel).filter_by(model_name=model_name).first()
    if existing:
        existing.version = 'v1'
        existing.status = 'inactive'
        existing.model_path = model_path
    else:
        session.add(TrainedModel(
            model_name=model_name,
            version='v1',
            status='inactive',
            model_path=model_path
        ))

    session.commit()
    session.remove()

def save_performance_metrics(df):
    session = get_session()
    try:
        for _, row in df.iterrows():
            perf = ModelPerformance(
                model_name=row['Model'],
                accuracy=row['Accuracy'],
                roc_auc=row['ROC AUC'],
                precision_1=row.get('Precision (1)', 0),
                recall_1=row.get('Recall (1)', 0),
                f1_score_1=row.get('F1-Score (1)', 0)
            )
            session.add(perf)
        session.commit()
    finally:
        session.remove()

def activate_best_model(df):
    session = get_session()
    try:
        best = df.sort_values('Accuracy', ascending=False).iloc[0]
        best_model_name = best['Model'].lower().replace(' ', '_')

        session.query(TrainedModel).update({TrainedModel.status: 'inactive'})
        best_model = session.query(TrainedModel).filter_by(model_name=best_model_name).first()

        if best_model:
            best_model.status = 'active'

        session.commit()
    finally:
        session.remove()

def save_visualization_record(viz_type, file_path, models):
    session = get_session()
    try:
        record = ModelVisualization(
            visualization_type=viz_type,
            file_path=file_path,
            related_models=', '.join(models)
        )
        session.add(record)
        session.commit()
    finally:
        session.remove()

def generate_filtered_roc_curve(data_path, model_dir, model_names, image_path):
    (X_train, X_test, y_train, y_test), _ = preprocess_data(data_path)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    for name in model_names:
        model_file = f"{name}.h5" if name == 'neural_net' else f"{name}.pkl"
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            continue

        if name == 'neural_net':
            model = load_model(model_path)
            y_proba = model.predict(X_test_scaled).flatten()
        elif name in ['logistic_regression', 'svm', 'knn', 'naive_bayes']:
            model = joblib.load(model_path)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model = joblib.load(model_path)
            y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        results[name.replace('_', ' ').title()] = {'fpr_tpr': (fpr, tpr), 'roc_auc': auc_score}

    plt.figure(figsize=(8, 6))
    for model_name, metrics in results.items():
        fpr, tpr = metrics['fpr_tpr']
        auc = metrics['roc_auc']
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)

def generate_filtered_comparison_plot(df, image_path):
    model_names = df['Model']
    accuracies = df['Accuracy']
    roc_aucs = df['ROC AUC']

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, accuracies, width, label='Accuracy')
    ax.bar(x + width/2, roc_aucs, width, label='ROC AUC')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Accuracy & ROC AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(image_path)

def get_existing_performance_df():
    session = get_session()
    try:
        records = session.query(ModelPerformance).all()
        data = [{
            "Model": r.model_name,
            "Accuracy": r.accuracy,
            "ROC AUC": r.roc_auc,
            "Precision (1)": r.precision_1,
            "Recall (1)": r.recall_1,
            "F1-Score (1)": r.f1_score_1,
        } for r in records]
        return pd.DataFrame(data)
    finally:
        session.remove()

def get_all_trained_model_names():
    session = get_session()
    try:
        return [m.model_name for m in session.query(TrainedModel.model_name).all()]
    finally:
        session.remove()
