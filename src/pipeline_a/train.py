import os
import sys
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pipeline_a.preprocess import load_and_clean_sms, extract_features, load_email


# ─────────────────────────────────────────
# Train all 3 models (reusable function)
# ─────────────────────────────────────────

def train_all_models(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train Logistic Regression, Naive Bayes and SVM.
    Automatically uses LinearSVC for large datasets (faster).

    Args:
        X_train, X_test : feature matrices
        y_train, y_test : labels
        dataset_name    : 'SMS' or 'Email'

    Returns:
        results         : metrics dict
        trained_models  : trained model objects
    """

    # Use LinearSVC for large datasets — much faster
    if X_train.shape[0] > 10000:
        svm_model = CalibratedClassifierCV(
            LinearSVC(max_iter=2000)
        )
        print("Note: Using LinearSVC (faster) for large dataset.")
    else:
        svm_model = SVC(probability=True, kernel='linear')
        print("Note: Using SVC for small dataset.")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes':         MultinomialNB(),
        'SVM':                 svm_model
    }

    results        = {}
    trained_models = {}

    print("=" * 55)
    print(f"TRAINING ALL 3 CLASSIFIERS — {dataset_name}")
    print("=" * 55)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(
            y_test, y_pred, output_dict=True
        )

        results[name] = {
            'accuracy':  round(report['accuracy'], 4),
            'precision': round(report['1']['precision'], 4),
            'recall':    round(report['1']['recall'], 4),
            'f1':        round(report['1']['f1-score'], 4),
            'roc_auc':   round(roc_auc_score(y_test, y_prob), 4)
        }
        trained_models[name] = (model, y_pred, y_prob)

        print(f"  Accuracy  : {results[name]['accuracy']}")
        print(f"  Precision : {results[name]['precision']}")
        print(f"  Recall    : {results[name]['recall']}")
        print(f"  F1 Score  : {results[name]['f1']}")
        print(f"  ROC-AUC   : {results[name]['roc_auc']}")

    return results, trained_models


# ─────────────────────────────────────────
# Save best model
# ─────────────────────────────────────────

def save_best_model(results, trained_models,
                    vectorizer, y_test,
                    model_file, vectorizer_file,
                    results_file, dataset_name):
    """
    Select best model by F1 score.
    Save model, vectorizer and results as .pkl files.

    Args:
        results         : metrics dict
        trained_models  : trained model objects
        vectorizer      : fitted vectorizer (None for email)
        y_test          : test labels
        model_file      : path to save model
        vectorizer_file : path to save vectorizer
        results_file    : path to save results
        dataset_name    : 'SMS' or 'Email'

    Returns:
        best_name       : name of best model
    """

    best_name = max(results, key=lambda x: results[x]['f1'])
    best_model, _, _ = trained_models[best_name]

    print("\n")
    print("=" * 55)
    print(f"SAVING BEST MODEL — {dataset_name}")
    print("=" * 55)
    print(f"Best Model : {best_name}")
    print(f"F1 Score   : {results[best_name]['f1']}")
    print(f"ROC-AUC    : {results[best_name]['roc_auc']}")
    print(f"\nJustification:")
    print(f"  {best_name} selected based on highest F1 score.")
    print(f"  F1 balances precision and recall — critical")
    print(f"  for spam detection.")

    os.makedirs('models', exist_ok=True)

    # Save best model
    joblib.dump(best_model, model_file)

    # Save vectorizer if provided
    if vectorizer is not None and vectorizer_file is not None:
        joblib.dump(vectorizer, vectorizer_file)

    # Save full results for evaluate.py
    joblib.dump({
        'results':        results,
        'trained_models': trained_models,
        'y_test':         y_test,
        'best_name':      best_name
    }, results_file)

    print(f"\nSaved files:")
    print(f"  {model_file}")
    if vectorizer_file:
        print(f"  {vectorizer_file}")
    print(f"  {results_file}")

    return best_name


# ─────────────────────────────────────────
# RUN FULL TRAINING PIPELINE
# ─────────────────────────────────────────

if __name__ == "__main__":

    # ──────────────────────────────────────
    # PIPELINE 1: SMS Spam Model
    # ──────────────────────────────────────
    print("\n")
    print("*" * 55)
    print("PIPELINE 1: SMS SPAM MODEL")
    print("*" * 55)

    # Step 1: Load and clean SMS data
    print("\nLoading and preparing SMS data...")
    sms_df = load_and_clean_sms()

    # Step 2: Extract TF-IDF features
    print("\nExtracting TF-IDF features...")
    X_train_sms, X_test_sms, y_train_sms, y_test_sms, sms_vectorizer = extract_features(
        sms_df, vectorizer_type='tfidf'
    )

    # Step 3: Train all 3 models
    sms_results, sms_trained = train_all_models(
        X_train_sms, X_test_sms,
        y_train_sms, y_test_sms,
        dataset_name='SMS'
    )

    # Step 4: Save best SMS model
    sms_best = save_best_model(
        results         = sms_results,
        trained_models  = sms_trained,
        vectorizer      = sms_vectorizer,
        y_test          = y_test_sms,
        model_file      = 'models/sms_spam_model.pkl',
        vectorizer_file = 'models/sms_spam_vectorizer.pkl',
        results_file    = 'models/sms_training_results.pkl',
        dataset_name    = 'SMS'
    )

    # ──────────────────────────────────────
    # PIPELINE 2: Email Spam Model
    # ──────────────────────────────────────
    print("\n")
    print("*" * 55)
    print("PIPELINE 2: EMAIL SPAM MODEL")
    print("*" * 55)

    # Step 1: Load email data (already numerical)
    print("\nLoading Email data...")
    email_df = load_email()

    # Step 2: Split features and label
    X_email = email_df.drop('is_spam', axis=1)
    y_email = email_df['is_spam']

    # Step 3: Train/Test split
    X_train_email, X_test_email, y_train_email, y_test_email = train_test_split(
        X_email, y_email,
        test_size=0.2,
        random_state=42,
        stratify=y_email
    )
    print(f"Train size : {len(X_train_email)}")
    print(f"Test size  : {len(X_test_email)}")

    # Step 4: Train all 3 models
    # Note: No TF-IDF needed — email data already numerical
    email_results, email_trained = train_all_models(
        X_train_email, X_test_email,
        y_train_email, y_test_email,
        dataset_name='Email'
    )

    # Step 5: Save best Email model
    email_best = save_best_model(
        results         = email_results,
        trained_models  = email_trained,
        vectorizer      = None,
        y_test          = y_test_email,
        model_file      = 'models/email_spam_model.pkl',
        vectorizer_file = None,
        results_file    = 'models/email_training_results.pkl',
        dataset_name    = 'Email'
    )

    # ──────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────
    print("\n")
    print("=" * 55)
    print("TRAINING COMPLETE — BOTH MODELS")
    print("=" * 55)
    print(f"SMS   Best Model : {sms_best}")
    print(f"Email Best Model : {email_best}")
    print(f"\nSaved files:")
    print(f"  models/sms_spam_model.pkl")
    print(f"  models/sms_spam_vectorizer.pkl")
    print(f"  models/sms_training_results.pkl")
    print(f"  models/email_spam_model.pkl")
    print(f"  models/email_training_results.pkl")
    print(f"\nRun evaluate.py to see full metrics and charts.")