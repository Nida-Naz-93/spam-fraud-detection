import numpy as np
import pandas as pd
import os

np.random.seed(42)


# ─────────────────────────────────────────
# Generate Normal Users
# ─────────────────────────────────────────

def generate_normal_users(n=1000):
    """
    Generate n normal user behavior records.

    Normal behavior ranges:
    - messages_per_minute    : 0.5 - 5
    - unique_recipients      : 1 - 15
    - avg_time_between_msgs  : 20 - 120 seconds
    - repeated_message_ratio : 0 - 10%
    - night_activity_ratio   : 0 - 20%
    - failed_message_ratio   : 0 - 5%

    Args:
        n : number of normal users

    Returns:
        pd.DataFrame
    """

    df = pd.DataFrame({
        'messages_per_minute':    np.random.uniform(0.5, 5.0, n),
        'unique_recipients':      np.random.randint(1, 15, n),
        'avg_time_between_msgs':  np.random.uniform(20, 120, n),
        'repeated_message_ratio': np.random.uniform(0.0, 0.10, n),
        'night_activity_ratio':   np.random.uniform(0.0, 0.20, n),
        'failed_message_ratio':   np.random.uniform(0.0, 0.05, n),
        'is_bot':                 0
    })

    return df


# ─────────────────────────────────────────
# Generate Bot Users
# ─────────────────────────────────────────

def generate_bot_users(n=75):
    """
    Generate n synthetic bot behavior records.

    Bot behavior ranges (extreme values):
    - messages_per_minute    : 100 - 500
    - unique_recipients      : 200 - 1000
    - avg_time_between_msgs  : 0 - 2 seconds
    - repeated_message_ratio : 70 - 100%
    - night_activity_ratio   : 50 - 100%
    - failed_message_ratio   : 10 - 50%

    Args:
        n : number of bot users

    Returns:
        pd.DataFrame
    """

    df = pd.DataFrame({
        'messages_per_minute':    np.random.uniform(100, 500, n),
        'unique_recipients':      np.random.randint(200, 1000, n),
        'avg_time_between_msgs':  np.random.uniform(0.0, 2.0, n),
        'repeated_message_ratio': np.random.uniform(0.70, 1.0, n),
        'night_activity_ratio':   np.random.uniform(0.50, 1.0, n),
        'failed_message_ratio':   np.random.uniform(0.10, 0.50, n),
        'is_bot':                 1
    })

    return df


# ─────────────────────────────────────────
# Combine and Save Dataset
# ─────────────────────────────────────────

def generate_behavioral_dataset(
        n_normal=1000, n_bots=75, save=True):
    """
    Generate complete behavioral dataset combining
    normal users and bots.

    Args:
        n_normal : number of normal users
        n_bots   : number of bot users
        save     : save to CSV if True

    Returns:
        pd.DataFrame
    """

    print("=" * 55)
    print("GENERATING BEHAVIORAL DATASET")
    print("=" * 55)

    # Generate both groups
    normal_df = generate_normal_users(n_normal)
    bot_df    = generate_bot_users(n_bots)

    print(f"Normal users generated : {len(normal_df)}")
    print(f"Bot users generated    : {len(bot_df)}")

    # Combine and shuffle
    df = pd.concat([normal_df, bot_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add user ID
    df.insert(0, 'user_id', range(1, len(df) + 1))

    print(f"Total users            : {len(df)}")
    print(f"Bot ratio              : {df['is_bot'].mean()*100:.1f}%")

    # Save to CSV
    if save:
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/behavioral_data.csv', index=False)
        print(f"Saved → data/behavioral_data.csv")

    return df


# ─────────────────────────────────────────
# Verify Dataset
# ─────────────────────────────────────────

def verify_dataset(df):
    """Print dataset statistics and verify ranges"""

    print("\n")
    print("=" * 55)
    print("DATASET VERIFICATION")
    print("=" * 55)

    print(f"\nShape          : {df.shape}")
    print(f"Total users    : {len(df)}")
    print(f"Normal users   : {(df['is_bot']==0).sum()}")
    print(f"Bot users      : {(df['is_bot']==1).sum()}")
    print(f"Null values    : {df.isnull().sum().sum()}")

    print("\nFeature Statistics — Normal Users:")
    normal_stats = df[df['is_bot']==0].drop(
        ['user_id','is_bot'], axis=1
    ).describe().round(3)
    print(normal_stats)

    print("\nFeature Statistics — Bot Users:")
    bot_stats = df[df['is_bot']==1].drop(
        ['user_id','is_bot'], axis=1
    ).describe().round(3)
    print(bot_stats)

    print("\nSample Normal Users (first 3):")
    print(df[df['is_bot']==0].head(3).to_string())

    print("\nSample Bot Users (first 3):")
    print(df[df['is_bot']==1].head(3).to_string())


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Generate dataset
    df = generate_behavioral_dataset(
        n_normal = 1000,
        n_bots   = 75,
        save     = True
    )

    # Verify
    verify_dataset(df)

    print("\n")
    print("=" * 55)
    print("DATA GENERATION COMPLETE")
    print("=" * 55)
    print("File saved : data/behavioral_data.csv")
    print("Next step  : train anomaly detection models")