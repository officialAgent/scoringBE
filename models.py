import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input

# ------------------ Preprocessing ------------------

def preprocess_data(path):
    df = pd.read_csv(path)
    df_closed = df[df['Loan_Status'] == 'closed'].copy()

    X = df_closed[['Age', 'Gender', 'Customer_Duration', 'Avg_Monthly_Bill_EUR',
                   'Billing_Delay_Count', 'crifScore', 'Device_Price_EUR', 'Loan_Price_EUR',
                   'Loan_Duration', 'Monthly_Installment_EUR', 'Payment_Delay_Count',
                   'Payment_Delay_Total_Days']].copy()

    y = df_closed['Loan_Settled'].astype(int)

    X['Payment_Delay_Count'] = pd.to_numeric(X['Payment_Delay_Count'], errors='coerce').fillna(0).astype(int)
    X['Payment_Delay_Total_Days'] = pd.to_numeric(X['Payment_Delay_Total_Days'], errors='coerce').fillna(0).astype(int)
    X['crifScore'] = pd.to_numeric(X['crifScore'], errors='coerce').fillna(X['crifScore'].mean())

    X_encoded = pd.get_dummies(X, columns=['Gender'], drop_first=True)

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_encoded, y)

    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0), X_encoded.columns.tolist()

# ------------------ Model Training ------------------

def train_and_save_model(data_path, model_dir, model_name):
    (X_train, X_test, y_train, y_test), feature_columns = preprocess_data(data_path)

    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    scaler_path = os.path.join(model_dir, "model_scaler.pkl")
    columns_path = os.path.join(model_dir, "model_columns.pkl")

    scaler = StandardScaler()

    tree_models = ['xgboost', 'random_forest', 'catboost']

    if model_name in tree_models:
        X_train_final = X_train
    else:
        X_train_final = scaler.fit_transform(X_train)
        joblib.dump(scaler, scaler_path)

    joblib.dump(feature_columns, columns_path)

    if model_name == 'xgboost':
        model = XGBClassifier(eval_metric='logloss').fit(X_train_final, y_train)
        joblib.dump(model, model_path)

    elif model_name == 'random_forest':
        model = RandomForestClassifier().fit(X_train_final, y_train)
        joblib.dump(model, model_path)

    elif model_name == 'logistic_regression':
        model = LogisticRegression(max_iter=1000).fit(X_train_final, y_train)
        joblib.dump(model, model_path)

    elif model_name == 'svm':
        model = SVC(probability=True).fit(X_train_final, y_train)
        joblib.dump(model, model_path)

    elif model_name == 'knn':
        model = KNeighborsClassifier().fit(X_train_final, y_train)
        joblib.dump(model, model_path)

    elif model_name == 'naive_bayes':
        model = GaussianNB().fit(X_train_final, y_train)
        joblib.dump(model, model_path)

    elif model_name == 'catboost':
        model = CatBoostClassifier(verbose=0).fit(X_train_final, y_train)
        joblib.dump(model, model_path)

    elif model_name == 'neural_net':
        model = Sequential([
            Input(shape=(X_train_final.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train_final, y_train, epochs=20, batch_size=32, verbose=0)
        model.save(os.path.join(model_dir, f"{model_name}.h5"))

# ------------------ Prediction ------------------

def load_and_predict(model_dir, model_name, input_data):
    feature_columns = joblib.load(os.path.join(model_dir, "model_columns.pkl"))

    df_input = pd.DataFrame([input_data])
    df_input_encoded = pd.get_dummies(df_input, columns=['Gender'], drop_first=True)
    df_input_encoded = df_input_encoded.reindex(columns=feature_columns, fill_value=0)

    scaler_needed = model_name in ['logistic_regression', 'svm', 'knn', 'neural_net']

    if scaler_needed:
        scaler = joblib.load(os.path.join(model_dir, "model_scaler.pkl"))
        X_input = scaler.transform(df_input_encoded)
    else:
        X_input = df_input_encoded

    if model_name == 'neural_net':
        model = load_model(os.path.join(model_dir, f"{model_name}.h5"))
        y_proba = model.predict(X_input).flatten()
        y_pred = (y_proba > 0.5).astype(int)
    else:
        model = joblib.load(os.path.join(model_dir, f"{model_name}.pkl"))
        y_pred = model.predict(X_input)
        y_proba = model.predict_proba(X_input)[:, 1]

    return y_pred[0], y_proba[0]

# ------------------ Metrics & Graphs ------------------

def generate_model_comparison_df(data_path, model_dir, only=None):
    (X_train, X_test, y_train, y_test), feature_columns = preprocess_data(data_path)

    models = only or ['xgboost', 'random_forest', 'logistic_regression',
                      'neural_net', 'catboost', 'svm', 'knn', 'naive_bayes']

    results = []
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name in models:
        model_file = f"{name}.h5" if name == 'neural_net' else f"{name}.pkl"
        model_path = os.path.join(model_dir, model_file)

        if not os.path.exists(model_path):
            continue

        if name in ['logistic_regression', 'svm', 'knn', 'naive_bayes', 'neural_net']:
            X_eval = X_test_scaled
        else:
            X_eval = X_test

        if name == 'neural_net':
            model = load_model(model_path)
            y_proba = model.predict(X_eval).flatten()
            y_pred = (y_proba > 0.5).astype(int)
        else:
            model = joblib.load(model_path)
            y_pred = model.predict(X_eval)
            y_proba = model.predict_proba(X_eval)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report.get('1', {}).get('precision', 0)
        recall = report.get('1', {}).get('recall', 0)
        f1 = report.get('1', {}).get('f1-score', 0)

        results.append((name.replace('_', ' ').title(), acc, auc_score, precision, recall, f1))

    return pd.DataFrame(results, columns=[
        'Model', 'Accuracy', 'ROC AUC', 'Precision (1)', 'Recall (1)', 'F1-Score (1)'
    ])

def generate_roc_curve(data_path, model_dir, image_path):
    df = generate_model_comparison_df(data_path, model_dir)
    (X_train, X_test, y_train, y_test), _ = preprocess_data(data_path)

    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    models = df['Model'].str.lower().str.replace(' ', '_').tolist()

    plt.figure(figsize=(10, 6))
    for model_name in models:
        model_file = f"{model_name}.h5" if model_name == 'neural_net' else f"{model_name}.pkl"
        model_path = os.path.join(model_dir, model_file)

        if not os.path.exists(model_path):
            continue

        if model_name in ['logistic_regression', 'svm', 'knn', 'naive_bayes', 'neural_net']:
            X_eval = X_test_scaled
        else:
            X_eval = X_test

        if model_name == 'neural_net':
            model = load_model(model_path)
            y_proba = model.predict(X_eval).flatten()
        else:
            model = joblib.load(model_path)
            y_proba = model.predict_proba(X_eval)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC={auc_score:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
