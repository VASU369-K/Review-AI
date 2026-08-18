import os
import re
import csv
import random
import argparse

def clean_text(text):
    """
    Cleans text by removing HTML tags, normalizing whitespaces, and unescaping quotes.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    # Replace multiple whitespaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_and_clean_file(filepath, max_records=None, seed=42):
    """
    Reads a CSV file, cleans text, maps labels:
      1 (Negative) -> 0
      2 (Positive) -> 1
    Deduplicates records and returns a shuffled list of (label, text) tuples.
    """
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return []

    print(f"Loading and processing {filepath}...")
    records = []
    
    # Read to gather records
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        try:
            next(reader) # skip header
        except StopIteration:
            pass
            
        for row in reader:
            if len(row) >= 2:
                records.append((row[0], row[1]))

    # Shuffle before slicing to get a representative random sample
    random.seed(seed)
    random.shuffle(records)

    if max_records and len(records) > max_records:
        records = records[:max_records]

    cleaned_records = []
    seen_texts = set()
    
    for label, text in records:
        mapped_label = 0 if label.strip() == "1" else 1
        cleaned_txt = clean_text(text)
        if cleaned_txt and cleaned_txt not in seen_texts:
            seen_texts.add(cleaned_txt)
            cleaned_records.append((mapped_label, cleaned_txt))

    print(f"File {os.path.basename(filepath)}: Loaded {len(records)} raw -> Shuffled, cleaned, and deduplicated to {len(cleaned_records)} unique records.")
    return cleaned_records

def preprocess_and_split(dataset_dir="Dataset", sample_size=100000, seed=42):
    """
    Splits the Amazon dataset into:
      - Train Split (train_split.csv) from train.csv
      - Validation Split (val_split.csv) from train.csv
      - Test Split (test_split.csv) from test.csv
    
    Keeps splits strictly separate to prevent data leakage from the test set!
    """
    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: train.csv or test.csv not found in {dataset_dir}")
        return

    # Proportional sampling for total of target 'sample_size' keeping 70/15/15 ratio
    # 70% Train, 15% Val, 15% Test
    if sample_size and sample_size != "all":
        train_val_target = int(sample_size * 0.85) # 85% from train.csv (to be split 70/15)
        test_target = int(sample_size * 0.15)      # 15% from test.csv
    else:
        train_val_target = None
        test_target = None

    # Process train.csv (strictly for Train + Val)
    train_val_records = preprocess_and_clean_file(train_path, max_records=train_val_target, seed=seed)
    
    # Process test.csv (strictly for Test evaluation)
    test_records = preprocess_and_clean_file(test_path, max_records=test_target, seed=seed)

    # Split train_val_records into Train (70/85 = 82.35%) and Val (15/85 = 17.65%)
    total_train_val = len(train_val_records)
    train_end = int(total_train_val * (70.0 / 85.0))

    train_data = train_val_records[:train_end]
    val_data = train_val_records[train_end:]
    test_data = test_records

    print("\n" + "="*60)
    print("DATASET CONFIGURATION DOCUMENTATION")
    print("="*60)
    print(f"1. Train Split (train_split.csv): {len(train_data)} records")
    print("   -> SOURCE: Compiled and cleaned solely from original train.csv.")
    print("   -> PURPOSE: Model training (parameter adjustments).")
    print(f"2. Validation Split (val_split.csv): {len(val_data)} records")
    print("   -> SOURCE: Compiled and cleaned solely from original train.csv.")
    print("   -> PURPOSE: Model validation (parameter tuning & checkpoint selection).")
    print(f"3. Test Split (test_split.csv): {len(test_data)} records")
    print("   -> SOURCE: Compiled and cleaned solely from original test.csv.")
    print("   -> PURPOSE: Final evaluation & model comparison (unseen during training).")
    print("="*60 + "\n")

    # Label distribution summaries
    for name, split_data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        split_len = len(split_data)
        if split_len > 0:
            neg = sum(1 for l, _ in split_data if l == 0)
            pos = sum(1 for l, _ in split_data if l == 1)
            print(f"{name} distribution: Negative={neg} ({neg/split_len*100:.1f}%), Positive={pos} ({pos/split_len*100:.1f}%)")

    # Save splits
    os.makedirs(dataset_dir, exist_ok=True)
    
    splits = {
        "train_split.csv": train_data,
        "val_split.csv": val_data,
        "test_split.csv": test_data
    }

    for filename, r_data in splits.items():
        out_path = os.path.join(dataset_dir, filename)
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["label", "text"])
            writer.writerows(r_data)
        print(f"Saved {len(r_data)} records to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and split dataset keeping train and test separate.")
    parser.add_argument("--dataset_dir", type=str, default="Dataset", help="Directory where dataset files are stored")
    parser.add_argument("--sample_size", type=str, default="100000", help="Number of samples to extract (or 'all' for full dataset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    # Parse sample_size command line arg
    sz = args.sample_size
    if sz.lower() != "all":
        sz = int(sz)
    else:
        sz = "all"

    preprocess_and_split(args.dataset_dir, sample_size=sz, seed=args.seed)
