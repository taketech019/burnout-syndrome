import zipfile
import json
import os
from collections import defaultdict

data_dir = r"C:\Users\choic\Desktop\Projects_Dev\PR_BOAZ_2nd\data\references\심리상담데이터"
training_dir = os.path.join(data_dir, "Training", "02.라벨링데이터")
validation_dir = os.path.join(data_dir, "Validation", "02.라벨링데이터")

def analyze_zip_files(directory_path, label="Training"):
    print(f"\n{'='*70}")
    print(f"{label} Data Analysis")
    print(f"{'='*70}")
    
    anxiety_labels = defaultdict(int)
    total_files = 0
    anxiety_normal_files = 0
    
    zip_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.zip.part0')])
    
    for zip_file in zip_files:
        file_path = os.path.join(directory_path, zip_file)
        print(f"Processing: {zip_file}")
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
                print(f"  JSON files found: {len(json_files)}")
                
                for json_file in json_files:
                    try:
                        with zip_ref.open(json_file) as f:
                            data = json.load(f)
                            total_files += 1
                            
                            # Check if this is anxiety or normal data
                            is_anxiety = "불안" in zip_file
                            is_normal = "일반군" in zip_file
                            
                            if is_anxiety or is_normal:
                                anxiety_normal_files += 1
                                anxiety_label = data.get('anxiety')
                                if anxiety_label is not None:
                                    anxiety_labels[anxiety_label] += 1
                    except Exception as e:
                        print(f"  Error: {json_file}: {str(e)[:50]}")
        except Exception as e:
            print(f"  ZIP Error: {str(e)[:50]}")
    
    print(f"\n{label} Summary:")
    print(f"  Total JSON files processed: {total_files}")
    print(f"  Anxiety + Normal files: {anxiety_normal_files}")
    print(f"\n  Anxiety Label Distribution (0=no, 1=yes, 2=mild, 3=severe):")
    if anxiety_labels:
        for label_val in sorted(anxiety_labels.keys()):
            count = anxiety_labels[label_val]
            pct = (count / anxiety_normal_files * 100) if anxiety_normal_files > 0 else 0
            print(f"    Label {label_val}: {count} files ({pct:.1f}%)")
    else:
        print("    (No labels found)")

analyze_zip_files(training_dir, "TRAINING")
analyze_zip_files(validation_dir, "VALIDATION")
