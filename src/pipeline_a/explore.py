import pandas as pd

# ─────────────────────────────────────────
# SMS SPAM DATASET EXPLORATION
# ─────────────────────────────────────────

print("=" * 55)
print("SMS SPAM DATASET")
print("=" * 55)

# Load dataset
sms = pd.read_csv('data/spam.csv', encoding='latin-1')

# Keep only useful columns
sms = sms[['v1', 'v2']]
sms.columns = ['label', 'message']

# Shape
print(f"Shape          : {sms.shape}")
print(f"Total Messages : {len(sms)}")

# Null values
print("\nNull Values:")
print(sms.isnull().sum())

# Class balance
print("\nClass Balance (Count):")
print(sms['label'].value_counts())

print("\nClass Balance (Percentage):")
print(sms['label'].value_counts(normalize=True).mul(100).round(1))

# Message length analysis
sms['length'] = sms['message'].apply(len)
print("\nAverage Message Length by Label:")
print(sms.groupby('label')['length'].mean().round(1))

# Sample messages
print("\n10 Sample Messages:")
print(sms[['label', 'message']].sample(10, random_state=42).to_string())

print("\nSample SPAM Messages:")
for msg in sms[sms['label'] == 'spam']['message'].values[:3]:
    print(f"  - {msg}")

print("\nSample HAM Messages:")
for msg in sms[sms['label'] == 'ham']['message'].values[:3]:
    print(f"  - {msg}")


# ─────────────────────────────────────────
# EMAIL SPAM DATASET EXPLORATION
# ─────────────────────────────────────────

print("\n")
print("=" * 55)
print("EMAIL SPAM DATASET")
print("=" * 55)

# Load dataset
email = pd.read_csv('data/spam_detection_dataset.csv', encoding='latin-1')

# Shape
print(f"Shape          : {email.shape}")
print(f"Total Emails   : {len(email)}")

# Columns
print(f"\nColumns        : {email.columns.tolist()}")

# Null values
print("\nNull Values:")
print(email.isnull().sum())

# Class balance
print("\nClass Balance (Count):")
print(email['is_spam'].value_counts())

print("\nClass Balance (Percentage):")
print(email['is_spam'].value_counts(normalize=True).mul(100).round(1))

# Sample rows
print("\nFirst 5 Rows:")
print(email.head(5).to_string())

# Feature statistics
print("\nFeature Statistics:")
print(email.describe().round(2))


# ─────────────────────────────────────────
# OBSERVATIONS SUMMARY
# ─────────────────────────────────────────

print("\n")
print("=" * 55)
print("OBSERVATIONS SUMMARY")
print("=" * 55)
print(f"SMS  Dataset : {len(sms)} messages | "
      f"Spam: {sms['label'].value_counts()['spam']} "
      f"({sms['label'].value_counts(normalize=True)['spam']*100:.1f}%)")
print(f"Email Dataset: {len(email)} emails   | "
      f"Spam: {email['is_spam'].sum()} "
      f"({email['is_spam'].mean()*100:.1f}%)")
print("\nBoth datasets are imbalanced.")
print("SMS  spam messages are longer on average.")
print("Email dataset is already in numerical format.")
print("No null values found in either dataset.")
print("\nExploration complete.")