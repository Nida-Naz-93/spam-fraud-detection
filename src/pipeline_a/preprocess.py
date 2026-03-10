import pandas as pd
import re
import nltk

# Download required NLTK data (only runs once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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
    print(f"  Total rows processed    : {len(sms_df)}")
    print(f"  Empty messages after clean: {sms_df['clean_message'].str.strip().eq('').sum()}")
    print(f"  Null values after clean : {sms_df['clean_message'].isnull().sum()}")

    email_df = load_email()

    print("\nBoth datasets loaded and ready.")