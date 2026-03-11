import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


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
# Load data and scaler
# ─────────────────────────────────────────

def load_data():
    """
    Load behavioral dataset and fitted scaler.

    Returns:
        df       : full dataframe
        X_scaled : normalized feature matrix
        scaler   : fitted StandardScaler
    """

    print("=" * 55)
    print("LOADING DATA AND SCALER")
    print("=" * 55)

    df     = pd.read_csv('data/behavioral_data.csv')
    scaler = joblib.load('models/behavior_scaler.pkl')

    X_scaled = scaler.transform(df[FEATURE_COLS].values)

    print(f"Loaded : {len(df)} users")
    print(f"Bots   : {df['is_bot'].sum()}")

    return df, X_scaled, scaler


# ─────────────────────────────────────────
# Contamination sweep
# ─────────────────────────────────────────

def contamination_sweep(X_scaled, df):
    """
    Run IsolationForest with different contamination
    values and record flagged counts at each step.

    Args:
        X_scaled : normalized feature matrix
        df       : full dataframe

    Returns:
        sweep_results : list of dicts per contamination
    """

    contamination_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]
    actual_bots          = df['is_bot'].sum()
    sweep_results        = []

    print("\n")
    print("=" * 60)
    print("CONTAMINATION SWEEP")
    print("=" * 60)
    print(f"Actual bots in dataset : {actual_bots}")
    print()
    print(f"{'Contamination':>15} {'Flagged':>10} "
          f"{'Rate':>8} {'vs Actual':>12}")
    print("-" * 60)

    for c in contamination_values:
        model = IsolationForest(
            contamination = c,
            random_state  = 42
        )
        preds       = model.fit_predict(X_scaled)
        n_flagged   = (preds == -1).sum()
        rate        = round(n_flagged / len(preds) * 100, 1)
        diff        = n_flagged - actual_bots

        sweep_results.append({
            'contamination': c,
            'n_flagged':     n_flagged,
            'rate':          rate,
            'diff':          diff,
            'model':         model,
            'predictions':   preds
        })

        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{c:>15} {n_flagged:>10} "
              f"{rate:>7}% {diff_str:>12}")

    print("=" * 60)

    return sweep_results


# ─────────────────────────────────────────
# Pick best contamination
# ─────────────────────────────────────────

def pick_best_contamination(sweep_results, df):
    """
    Pick contamination value where flagged count
    is closest to actual bot count.

    Args:
        sweep_results : list of sweep result dicts
        df            : full dataframe

    Returns:
        best : best sweep result dict
    """

    actual_bots = df['is_bot'].sum()

    best = min(
        sweep_results,
        key=lambda x: abs(x['n_flagged'] - actual_bots)
    )

    print("\n")
    print("=" * 55)
    print("BEST CONTAMINATION VALUE")
    print("=" * 55)
    print(f"Actual bots      : {actual_bots}")
    print(f"Best value       : {best['contamination']}")
    print(f"Flagged          : {best['n_flagged']}")
    print(f"Difference       : {best['diff']}")
    print(f"Justification    : Closest to actual bot count")

    return best


# ─────────────────────────────────────────
# False positive analysis
# ─────────────────────────────────────────

def false_positive_analysis(best, df, X_scaled):
    """
    Manually inspect flagged users to identify
    false positives — normal users wrongly flagged.

    False positive = flagged as anomaly but is_bot=0

    Args:
        best     : best sweep result dict
        df       : full dataframe
        X_scaled : normalized feature matrix

    Returns:
        flagged_df : dataframe of all flagged users
        fp_df      : dataframe of false positives only
    """

    print("\n")
    print("=" * 55)
    print("FALSE POSITIVE ANALYSIS")
    print("=" * 55)

    preds = best['predictions']
    df    = df.copy()

    df['flagged'] = (preds == -1).astype(int)

    # All flagged users
    flagged_df = df[df['flagged'] == 1].copy()

    # False positives = flagged but NOT a bot
    fp_df = flagged_df[flagged_df['is_bot'] == 0].copy()

    # True positives = flagged AND is a bot
    tp_df = flagged_df[flagged_df['is_bot'] == 1].copy()

    print(f"Total flagged    : {len(flagged_df)}")
    print(f"True positives   : {len(tp_df)}  (bots correctly flagged)")
    print(f"False positives  : {len(fp_df)}  (normal users wrongly flagged)")

    if len(fp_df) > 0:
        print(f"\nFalse Positive Users (sample 10):")
        print("-" * 55)
        print(fp_df[FEATURE_COLS + ['risk_score'] 
                    if 'risk_score' in fp_df.columns 
                    else FEATURE_COLS].head(10).to_string())

        print(f"\nFalse Positive Feature Averages:")
        print("-" * 55)
        fp_means = fp_df[FEATURE_COLS].mean().round(3)
        for col, val in fp_means.items():
            print(f"  {col:<30} : {val}")
    else:
        print("\nNo false positives found! ✅")

    return flagged_df, fp_df


# ─────────────────────────────────────────
# Risk score with top feature
# ─────────────────────────────────────────

def generate_risk_scores(model, X_scaled, df):
    """
    Generate 0-100 risk scores and identify
    top contributing feature per user.

    Args:
        model    : fitted IsolationForest
        X_scaled : normalized feature matrix
        df       : full dataframe

    Returns:
        df with risk_score and top_feature columns
    """

    print("\n")
    print("=" * 55)
    print("GENERATING RISK SCORES WITH TOP FEATURE")
    print("=" * 55)

    df         = df.copy()
    raw_scores = model.score_samples(X_scaled)

    # Normalize to 0-100
    min_s      = raw_scores.min()
    max_s      = raw_scores.max()
    risk       = 100 * (max_s - raw_scores) / (max_s - min_s)
    df['risk_score'] = np.clip(risk, 0, 100).astype(int)

    # Top contributing feature per user
    # Feature with highest absolute normalized value
    top_features = []
    for row in X_scaled:
        abs_vals    = np.abs(row)
        top_idx     = np.argmax(abs_vals)
        top_features.append(FEATURE_COLS[top_idx])

    df['top_feature'] = top_features

    print(f"Risk scores generated for {len(df)} users")

    print("\nTop 10 Riskiest Users:")
    print("-" * 55)
    cols = ['user_id', 'risk_score', 'top_feature',
            'is_bot', 'messages_per_minute',
            'unique_recipients']
    print(df.nlargest(10, 'risk_score')[cols].to_string())

    print("\nTop Contributing Features (flagged users):")
    flagged = df[df['risk_score'] > 70]
    print(flagged['top_feature'].value_counts().to_string())

    return df


# ─────────────────────────────────────────
# Plot contamination sweep
# ─────────────────────────────────────────

def plot_sweep(sweep_results, actual_bots):
    """Save contamination sweep chart"""

    os.makedirs('reports', exist_ok=True)

    values   = [r['contamination'] for r in sweep_results]
    flagged  = [r['n_flagged']     for r in sweep_results]

    plt.figure(figsize=(8, 5))
    plt.plot(values, flagged, marker='o',
             color='steelblue', linewidth=2, label='Flagged Users')
    plt.axhline(y=actual_bots, color='red',
                linestyle='--', label=f'Actual Bots ({actual_bots})')
    plt.xlabel('Contamination Value')
    plt.ylabel('Number of Flagged Users')
    plt.title('Contamination Tuning — Isolation Forest')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/contamination_sweep.png')
    plt.close()
    print("Saved → reports/contamination_sweep.png")


# ─────────────────────────────────────────
# Save final model
# ─────────────────────────────────────────

def save_final_model(best_model, scaler):
    """Save final tuned model"""

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/behavior_model.pkl')
    joblib.dump(scaler,     'models/behavior_scaler.pkl')

    print("\nFinal model saved:")
    print("  models/behavior_model.pkl")
    print("  models/behavior_scaler.pkl")


# ─────────────────────────────────────────
# RUN FULL EVALUATION
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Step 1: Load data
    df, X_scaled, scaler = load_data()

    # Step 2: Contamination sweep
    sweep_results = contamination_sweep(X_scaled, df)

    # Step 3: Pick best contamination
    best = pick_best_contamination(sweep_results, df)

    # Step 4: False positive analysis
    flagged_df, fp_df = false_positive_analysis(
        best, df, X_scaled
    )

    # Step 5: Risk scores with top feature
    df_scored = generate_risk_scores(
        best['model'], X_scaled, df
    )

    # Step 6: Plot sweep chart
    print("\nSaving charts...")
    plot_sweep(sweep_results, df['is_bot'].sum())

    # Step 7: Save final model
    print("\nSaving final model...")
    save_final_model(best['model'], scaler)

    print("\n")
    print("=" * 55)
    print("EVALUATION COMPLETE")
    print("=" * 55)
    print(f"Best contamination : {best['contamination']}")
    print(f"Flagged users      : {best['n_flagged']}")
    print(f"False positives    : {len(fp_df)}")
    print(f"Chart saved        : reports/contamination_sweep.png")
    print(f"Model saved        : models/behavior_model.pkl")