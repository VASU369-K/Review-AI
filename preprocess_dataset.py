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

def preprocess_and_split(dataset_dir="Dataset", sample_size=100000, seed=42):
    """
    Combines train.csv and test.csv, cleans the text, maps labels:
      1 (Negative) -> 0
      2 (Positive) -> 1
    Splits into 70% Train, 15% Validation, and 15% Test.
    Supports sampling to keep file sizes manageable for local training.
    """
    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: train.csv or test.csv not found in {dataset_dir}")
        return

    print("Loading datasets for preprocessing...")
    
    # We read line-by-line to avoid loading the entire 1.6GB file into memory if sample_size is smaller
    records = []
    
    # Read test.csv
    with open(test_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader) # skip header
        for row in reader:
            if len(row) >= 2:
                records.append((row[0], row[1]))

    # Read train.csv (streaming)
    # If a sample size is set, we read up to a limit or randomly sample.
    # To keep memory footprint low and still sample representatively:
    print(f"Read {len(records)} records from test.csv. Reading train.csv...")
    
    # We will load reviews from train.csv
    # If sample_size is set to a specific number, we will read until we have enough,
    # but to ensure we don't skew towards test.csv, we'll combine them and sample.
    with open(train_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        # Since train.csv has 3.6 million rows, we don't want to load all if sample_size is small.
        # However, to sample randomly, we either need to read them all or sample online.
        # A simple reservoir sampling or step-based sampling works.
        # Let's say if we want 100,000 total samples, we can take all from test.csv (400k) and train.csv (3.6M) and downsample.
        # To avoid reading all 3.6M rows into memory, we can read every N-th row from train.csv if sample_size is specified.
        if sample_size and sample_size != "all":
            # Target count from train.csv = sample_size - len(records)
            # If we want 100,000, and we already have 400,000 from test, we can just downsample the combined set later.
            # But let's read at most 500,000 rows total using a step size to save memory.
            step = 10  # read every 10th row from train
            count = 0
            for row in reader:
                count += 1
                if count % step == 0:
                    if len(row) >= 2:
                        records.append((row[0], row[1]))
                    if len(records) >= sample_size * 2:
                        break
        else:
            for row in reader:
                if len(row) >= 2:
                    records.append((row[0], row[1]))

    random.seed(seed)
    random.shuffle(records)

    if sample_size and sample_size != "all" and len(records) > sample_size:
        print(f"Sampling dataset down to {sample_size} records...")
        records = records[:sample_size]

    print("Cleaning text and mapping labels (1->0 (Neg), 2->1 (Pos))...")
    cleaned_records = []
    
    for label, text in records:
        # map label
        # label can be "1" or "2"
        # "1" -> 0 (Negative)
        # "2" -> 1 (Positive)
        mapped_label = 0 if label.strip() == "1" else 1
        cleaned_txt = clean_text(text)
        if cleaned_txt:
            cleaned_records.append((mapped_label, cleaned_txt))

    # Deduplicate by review text
    before_dedup = len(cleaned_records)
    seen_texts = set()
    unique_records = []
    for label, text in cleaned_records:
        if text not in seen_texts:
            seen_texts.add(text)
            unique_records.append((label, text))
    cleaned_records = unique_records
    after_dedup = len(cleaned_records)
    if before_dedup != after_dedup:
        print(f"Removed {before_dedup - after_dedup} duplicate reviews.")

    total_records = len(cleaned_records)
    print(f"Total cleaned records: {total_records}")

    # Label distribution summary
    neg_count = sum(1 for l, _ in cleaned_records if l == 0)
    pos_count = sum(1 for l, _ in cleaned_records if l == 1)
    print(f"Label distribution: Negative (0) = {neg_count} ({neg_count/total_records*100:.1f}%), Positive (1) = {pos_count} ({pos_count/total_records*100:.1f}%)")

    # Split 70% Train, 15% Val, 15% Test
    train_end = int(total_records * 0.70)
    val_end = train_end + int(total_records * 0.15)

    train_data = cleaned_records[:train_end]
    val_data = cleaned_records[train_end:val_end]
    test_data = cleaned_records[val_end:]

    print(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

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
    parser = argparse.ArgumentParser(description="Clean and split dataset.")
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
