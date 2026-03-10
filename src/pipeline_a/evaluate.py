import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


# ─────────────────────────────────────────
# Load saved training results
# ─────────────────────────────────────────

def load_results(results_file):
    """
    Load saved training results from train.py

    Args:
        results_file : path to .pkl results file

    Returns:
        results, trained_models, y_test, best_name
    """

    print(f"Loading results from {results_file}...")
    data           = joblib.load(results_file)
    results        = data['results']
    trained_models = data['trained_models']
    y_test         = data['y_test']
    best_name      = data['best_name']

    print(f"Loaded results for : {list(results.keys())}")
    return results, trained_models, y_test, best_name


# ─────────────────────────────────────────
# Print comparison table
# ─────────────────────────────────────────

def print_comparison_table(results, best_name, dataset_name):
    """Print side by side model comparison table"""

    print("\n")
    print("=" * 65)
    print(f"MODEL COMPARISON TABLE — {dataset_name}")
    print("=" * 65)
    print(f"{'Model':<22} {'Acc':>7} {'Prec':>7} "
          f"{'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 65)

    for name, m in results.items():
        marker = " <-- BEST" if name == best_name else ""
        print(f"{name:<22} {m['accuracy']:>7} {m['precision']:>7} "
              f"{m['recall']:>7} {m['f1']:>7} "
              f"{m['roc_auc']:>7}{marker}")

    print("=" * 65)


# ─────────────────────────────────────────
# Confusion matrix breakdown
# ─────────────────────────────────────────

def print_cm_analysis(trained_models, y_test, best_name, dataset_name):
    """Print TP, TN, FP, FN breakdown for best model"""

    model, y_pred, y_prob = trained_models[best_name]
    cm = confusion_matrix(y_test, y_pred)

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    print("\n")
    print("=" * 55)
    print(f"CONFUSION MATRIX ANALYSIS — {dataset_name} — {best_name}")
    print("=" * 55)
    print(f"  True Negative  (Ham correct)   : {tn}")
    print(f"  False Positive (Ham as spam)   : {fp}")
    print(f"  False Negative (Spam missed)   : {fn}")
    print(f"  True Positive  (Spam caught)   : {tp}")
    print(f"\n  Meaning:")
    print(f"  {tp} spam messages correctly caught")
    print(f"  {fn} spam messages slipped through")
    print(f"  {fp} legitimate messages wrongly flagged")
    print(f"  {tn} legitimate messages correctly allowed")


# ─────────────────────────────────────────
# Plot confusion matrices
# ─────────────────────────────────────────

def plot_confusion_matrices(trained_models, y_test, dataset_name):
    """Save confusion matrix image for each model"""

    os.makedirs('reports', exist_ok=True)
    prefix = dataset_name.lower()

    for name, (model, y_pred, y_prob) in trained_models.items():
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam']
        )
        plt.title(f'Confusion Matrix — {dataset_name} — {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()

        filename = (f"reports/cm_{prefix}_"
                    f"{name.lower().replace(' ', '_')}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved → {filename}")


# ─────────────────────────────────────────
# Plot ROC curves
# ─────────────────────────────────────────

def plot_roc_curves(results, trained_models, y_test, dataset_name):
    """Save ROC curves for all 3 models in one chart"""

    plt.figure(figsize=(8, 6))

    for name, (model, y_pred, y_prob) in trained_models.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = results[name]['roc_auc']
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves — {dataset_name} Spam Classifier')
    plt.legend()
    plt.tight_layout()

    filename = f"reports/roc_curves_{dataset_name.lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved → {filename}")


# ─────────────────────────────────────────
# Evaluate one dataset
# ─────────────────────────────────────────

def evaluate_dataset(results_file, dataset_name):
    """
    Run full evaluation for one dataset.

    Args:
        results_file : path to saved training results .pkl
        dataset_name : 'SMS' or 'Email'

    Returns:
        best_name    : name of best model
    """

    print("\n")
    print("*" * 55)
    print(f"EVALUATING — {dataset_name}")
    print("*" * 55)

    # Step 1: Load results saved by train.py
    results, trained_models, y_test, best_name = load_results(
        results_file
    )

    # Step 2: Print comparison table
    print_comparison_table(results, best_name, dataset_name)

    # Step 3: Print confusion matrix analysis
    print_cm_analysis(
        trained_models, y_test, best_name, dataset_name
    )

    # Step 4: Save confusion matrix images
    print("\nSaving confusion matrices...")
    plot_confusion_matrices(trained_models, y_test, dataset_name)

    # Step 5: Save ROC curves
    print("\nSaving ROC curves...")
    plot_roc_curves(results, trained_models, y_test, dataset_name)

    return best_name


# ─────────────────────────────────────────
# RUN EVALUATION — BOTH DATASETS
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Evaluate SMS model
    sms_best = evaluate_dataset(
        results_file = 'models/sms_training_results.pkl',
        dataset_name = 'SMS'
    )

    # Evaluate Email model
    email_best = evaluate_dataset(
        results_file = 'models/email_training_results.pkl',
        dataset_name = 'Email'
    )

    print("\n")
    print("=" * 55)
    print("EVALUATION COMPLETE — BOTH MODELS")
    print("=" * 55)
    print(f"SMS   Best Model : {sms_best}")
    print(f"Email Best Model : {email_best}")
    print(f"Reports saved    : reports/")