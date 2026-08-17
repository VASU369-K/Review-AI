import csv
import os
import sys

def convert_file(input_path, output_path):
    print(f"Reading {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        return

    # Check file size to show progress
    total_size = os.path.getsize(input_path)
    processed_size = 0
    
    with open(input_path, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        
        writer = csv.writer(outfile)
        # Write header
        writer.writerow(['label', 'text'])
        
        count = 0
        for line in infile:
            # We encode back to utf-8 to approximate size for progress reporting
            processed_size += len(line.encode('utf-8'))
            line = line.strip()
            if not line:
                continue
            
            # fastText format: "__label__1 review text" or "__label__2 review text"
            if line.startswith("__label__"):
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    label_str = parts[0].replace("__label__", "")
                    text = parts[1]
                else:
                    label_str = line.replace("__label__", "")
                    text = ""
            else:
                label_str = ""
                text = line
            
            writer.writerow([label_str, text])
            count += 1
            if count % 100000 == 0:
                percent = (processed_size / total_size) * 100
                print(f"Processed {count} lines ({percent:.2f}%)...")
                
    print(f"Finished converting {input_path} to {output_path}. Total lines: {count}")

def main():
    dataset_dir = "Dataset"
    
    # Path coordinates
    # We will convert both test and train files.
    files_to_convert = [
        ("test.ft.txt", "test.csv"),
        ("train.ft.txt", "train.csv")
    ]
    
    for in_file, out_file in files_to_convert:
        input_path = os.path.join(dataset_dir, in_file)
        output_path = os.path.join(dataset_dir, out_file)
        convert_file(input_path, output_path)

if __name__ == "__main__":
    main()
