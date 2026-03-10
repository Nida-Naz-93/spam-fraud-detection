import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pipeline_b.generate_data import generate_behavioral_dataset


# ─────────────────────────────────────────
# Feature columns
# ─────────────────────────────────────────

FEATURE_COLS = [
    'messages_per_minute',
    'unique_recipients',
    'avg_time_between_msgs',
    'repeated_message_ratio',
    'night_activity_ratio',
    'failed_message_ratio'
]


# ─────────────────────────────────────────
# Load and normalize data
# ─────────────────────────────────────────

def load_and_normalize():
    """
    Load behavioral dataset and normalize
    all 6 features using StandardScaler.

    Returns:
        X_scaled  : normalized feature matrix
        X_raw     : original feature matrix
        df        : full dataframe
        scaler    : fitted StandardScaler
    """

    print("=" * 55)
    print("LOADING AND NORMALIZING DATA")
    print("=" * 55)

    # Load dataset
    df = pd.read_csv('data/behavioral_data.csv')
    print(f"Loaded : {len(df)} users")

    # Extract features only
    X_raw = df[FEATURE_COLS].values

    # Normalize using StandardScaler
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"Features normalized : {len(FEATURE_COLS)} columns")
    print(f"Mean after scaling  : {X_scaled.mean().round(4)}")
    print(f"Std after scaling   : {X_scaled.std().round(4)}")

    return X_scaled, X_raw, df, scaler


# ─────────────────────────────────────────
# Train all 3 anomaly detectors
# ─────────────────────────────────────────

def train_anomaly_models(X_scaled, contamination=0.08):
    """
    Train IsolationForest, OneClassSVM and
    LocalOutlierFactor on normalized data.

    Args:
        X_scaled      : normalized feature matrix
        contamination : expected % of anomalies (0.01-0.15)

    Returns:
        models  : dict of trained models
        results : dict of predictions per model
    """

    print("\n")
    print("=" * 55)
    print(f"TRAINING ANOMALY MODELS (contamination={contamination})")
    print("=" * 55)

    models = {
        'Isolation Forest': IsolationForest(
            contamination = contamination,
            random_state  = 42
        ),
        'One-Class SVM': OneClassSVM(
            nu      = contamination,
            kernel  = 'rbf',
            gamma   = 'scale'
        ),
        'Local Outlier Factor': LocalOutlierFactor(
            n_neighbors   = 20,
            contamination = contamination
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        preds = model.fit_predict(X_scaled)

        # -1 = anomaly, 1 = normal
        n_anomalies = (preds == -1).sum()
        n_normal    = (preds == 1).sum()

        results[name] = {
            'predictions': preds,
            'n_anomalies': n_anomalies,
            'n_normal':    n_normal,
            'anomaly_pct': round(n_anomalies / len(preds) * 100, 1)
        }

        print(f"  Anomalies flagged : {n_anomalies}")
        print(f"  Normal users      : {n_normal}")
        print(f"  Anomaly rate      : {results[name]['anomaly_pct']}%")

    return models, results


# ─────────────────────────────────────────
# Print comparison table
# ─────────────────────────────────────────

def print_comparison(results, df):
    """Print side by side comparison of all 3 models"""

    print("\n")
    print("=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)
    print(f"Total bots in dataset : {df['is_bot'].sum()}")
    print(f"Contamination used    : 0.08 (8%)")
    print()
    print(f"{'Model':<22} {'Flagged':>8} {'Normal':>8} {'Rate':>8}")
    print("-" * 55)

    for name, r in results.items():
        print(f"{name:<22} {r['n_anomalies']:>8} "
              f"{r['n_normal']:>8} {r['anomaly_pct']:>7}%")

    print("=" * 55)


# ─────────────────────────────────────────
# Generate risk scores
# ─────────────────────────────────────────

def generate_risk_scores(iso_forest, X_scaled, df):
    """
    Convert Isolation Forest scores to 0-100 risk scores.
    More negative score = more anomalous = higher risk.

    Args:
        iso_forest : fitted IsolationForest model
        X_scaled   : normalized feature matrix
        df         : full dataframe

    Returns:
        df with risk_score column added
    """

    raw_scores  = iso_forest.score_samples(X_scaled)

    # Normalize to 0-100
    # More negative = higher risk
    min_score   = raw_scores.min()
    max_score   = raw_scores.max()
    risk_scores = 100 * (max_score - raw_scores) / (max_score - min_score)
    risk_scores = np.clip(risk_scores, 0, 100).astype(int)

    df = df.copy()
    df['risk_score'] = risk_scores
    df['anomaly']    = (iso_forest.predict(X_scaled) == -1).astype(int)

    return df


# ─────────────────────────────────────────
# Save models
# ─────────────────────────────────────────

def save_models(models, scaler):
    """Save best model (Isolation Forest) and scaler"""

    os.makedirs('models', exist_ok=True)

    # Save Isolation Forest as primary model
    joblib.dump(models['Isolation Forest'], 'models/behavior_model.pkl')
    joblib.dump(scaler,                     'models/behavior_scaler.pkl')

    # Save all 3 models
    joblib.dump(models, 'models/all_anomaly_models.pkl')

    print("\nSaved files:")
    print("  models/behavior_model.pkl")
    print("  models/behavior_scaler.pkl")
    print("  models/all_anomaly_models.pkl")


# ─────────────────────────────────────────
# RUN FULL PIPELINE
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Step 1: Load and normalize
    X_scaled, X_raw, df, scaler = load_and_normalize()

    # Step 2: Train all 3 models
    models, results = train_anomaly_models(
        X_scaled, contamination=0.08
    )

    # Step 3: Print comparison
    print_comparison(results, df)

    # Step 4: Generate risk scores using Isolation Forest
    print("\nGenerating risk scores...")
    df_scored = generate_risk_scores(
        models['Isolation Forest'], X_scaled, df
    )

    print("\nTop 10 Riskiest Users:")
    print("=" * 55)
    top10 = df_scored.nlargest(10, 'risk_score')[
        ['user_id', 'risk_score', 'anomaly', 'is_bot',
         'messages_per_minute', 'unique_recipients']
    ]
    print(top10.to_string())

    print("\nRisk Score Distribution:")
    print(f"  High risk   (>70) : {(df_scored['risk_score']>70).sum()}")
    print(f"  Medium risk (40-70): {((df_scored['risk_score']>=40) & (df_scored['risk_score']<=70)).sum()}")
    print(f"  Low risk    (<40) : {(df_scored['risk_score']<40).sum()}")

    # Step 5: Save models
    print("\nSaving models...")
    save_models(models, scaler)

    print("\n")
    print("=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)
    print("Primary model : Isolation Forest")
    print("Saved to      : models/behavior_model.pkl")
    print("Run evaluate.py for contamination tuning analysis.")