import pandas as pd
import re
import nltk

# Download required NLTK data (only runs once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────
# SPAM-20: preprocess_text() function
# ─────────────────────────────────────────

def preprocess_text(text):
    """
    Reusable text preprocessing function.

    Steps:
    1. Convert to lowercase
    2. Remove punctuation and numbers using regex
    3. Tokenize using nltk.word_tokenize
    4. Remove English stopwords

    Args:
        text (str): Raw input message

    Returns:
        str: Cleaned and preprocessed text
    """

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Step 3: Tokenize using NLTK word_tokenize
    tokens = word_tokenize(text)

    # Step 4: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Step 5: Join tokens back into clean string
    return ' '.join(tokens)


# ─────────────────────────────────────────
# Load and clean SMS dataset
# ─────────────────────────────────────────

def load_and_clean_sms():
    """
    Load the SMS spam dataset and apply text preprocessing.

    Returns:
        pd.DataFrame: Cleaned dataframe with columns:
                      label, message, clean_message, label_num
    """

    print("Loading SMS dataset...")
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    print("Cleaning text...")
    df['clean_message'] = df['message'].apply(preprocess_text)

    # Convert label to number: spam=1, ham=0
    df['label_num'] = (df['label'] == 'spam').astype(int)

    print("Done!")
    print(f"Total messages : {len(df)}")
    print(f"Spam           : {df['label_num'].sum()}")
    print(f"Ham            : {len(df) - df['label_num'].sum()}")

    return df


# ─────────────────────────────────────────
# Load email dataset
# ─────────────────────────────────────────

def load_email():
    """
    Load the email spam dataset.
    Already in numerical format, no text cleaning needed.

    Returns:
        pd.DataFrame: Email dataframe with features and is_spam label
    """

    print("\nLoading Email dataset...")
    df = pd.read_csv('data/spam_detection_dataset.csv', encoding='latin-1')

    print("Done!")
    print(f"Total emails : {len(df)}")
    print(f"Spam         : {df['is_spam'].sum()}")
    print(f"Ham          : {len(df) - df['is_spam'].sum()}")

    return df


# ─────────────────────────────────────────
# SPAM-22: Train/Test split + TF-IDF
# ─────────────────────────────────────────

def extract_features(df, vectorizer_type='tfidf', max_features=5000):
    """
    Split dataset and extract features using TF-IDF or CountVectorizer.

    IMPORTANT: Fit vectorizer on TRAIN set only to avoid data leakage.

    Args:
        df              : cleaned SMS dataframe from load_and_clean_sms()
        vectorizer_type : 'tfidf' or 'count'
        max_features    : maximum vocabulary size

    Returns:
        X_train_vec, X_test_vec, y_train, y_test, vectorizer
    """

    X = df['clean_message']
    y = df['label_num']

    # Step 1: Train/Test split FIRST (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y        # keeps spam/ham ratio same in both splits
    )

    print(f"Train size : {len(X_train)} messages")
    print(f"Test size  : {len(X_test)} messages")
    print(f"Train spam : {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"Test spam  : {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    # Step 2: Choose vectorizer
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
        print(f"\nVectorizer : TF-IDF (max_features={max_features})")
    else:
        vectorizer = CountVectorizer(max_features=max_features)
        print(f"\nVectorizer : CountVectorizer (max_features={max_features})")

    # Step 3: Fit on TRAIN only, transform both
    X_train_vec = vectorizer.fit_transform(X_train)   # learn + convert
    X_test_vec  = vectorizer.transform(X_test)         # only convert

    print(f"Train matrix shape : {X_train_vec.shape}")
    print(f"Test matrix shape  : {X_test_vec.shape}")

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


# ─────────────────────────────────────────
# SPAM-21: Apply to full dataset & verify
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Verify preprocess_text() on sample messages
    print("=" * 55)
    print("SPAM-20: PREPROCESSING VERIFICATION")
    print("=" * 55)

    test_messages = [
        "FREE entry!!! Win $1000 NOW!! Click HERE immediately!",
        "Hey, are you coming to the meeting tomorrow at 3pm?",
        "Congratulations! You've been selected for a PRIZE. Call 09012345678",
        "Don't forget to bring the documents when you come.",
    ]

    for msg in test_messages:
        cleaned = preprocess_text(msg)
        print(f"\nOriginal : {msg}")
        print(f"Cleaned  : {cleaned}")
        print("-" * 55)

    # Apply to full SMS dataset
    print("\n")
    print("=" * 55)
    print("SPAM-21: APPLYING TO FULL DATASET")
    print("=" * 55)

    sms_df = load_and_clean_sms()

    print("\nBEFORE CLEANING (first 3):")
    for msg in sms_df['message'].values[:3]:
        print(f"  - {msg}")

    print("\nAFTER CLEANING (first 3):")
    for msg in sms_df['clean_message'].values[:3]:
        print(f"  - {msg}")

    print("\nVERIFICATION CHECKS:")
    print(f"  Total rows processed     : {len(sms_df)}")
    print(f"  Empty messages after clean: {sms_df['clean_message'].str.strip().eq('').sum()}")
    print(f"  Null values after clean  : {sms_df['clean_message'].isnull().sum()}")

    # SPAM-22: Feature Extraction
    print("\n")
    print("=" * 55)
    print("SPAM-22: FEATURE EXTRACTION")
    print("=" * 55)

    print("\nTF-IDF Vectorizer:")
    print("-" * 55)
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vec = extract_features(
        sms_df, vectorizer_type='tfidf'
    )

    print("\nCountVectorizer:")
    print("-" * 55)
    X_train_count, X_test_count, y_train, y_test, count_vec = extract_features(
        sms_df, vectorizer_type='count'
    )

    print("\nTop 10 TF-IDF vocabulary:")
    print(tfidf_vec.get_feature_names_out()[:10])

    print("\nTop 10 Count vocabulary:")
    print(count_vec.get_feature_names_out()[:10])

    email_df = load_email()

    print("\nFeature extraction complete.")