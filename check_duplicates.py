import os
import hashlib
import json
from collections import defaultdict

def check_duplicates(folder_path):
    """Check for duplicate JSON files by content hash"""
    hash_map = defaultdict(list)
    file_count = 0
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                file_hash = hashlib.md5(content).hexdigest()
                hash_map[file_hash].append(filename)
                file_count += 1
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    # Find duplicates
    duplicates = {h: files for h, files in hash_map.items() if len(files) > 1}
    
    print(f"Folder: {folder_path}")
    print(f"Total files: {file_count}")
    print(f"Unique content: {len(hash_map)}")
    print(f"Duplicate groups: {len(duplicates)}")
    
    if duplicates:
        total_dup_count = sum(len(files) for files in duplicates.values()) - len(duplicates)
        print(f"Total duplicate files: {total_dup_count} ({100*total_dup_count/file_count:.1f}%)")
        print(f"\nDuplicate details (top 10):")
        for i, (file_hash, file_list) in enumerate(sorted(duplicates.items(), key=lambda x: -len(x[1]))[:10]):
            print(f"  Group {i+1} ({len(file_list)} files):")
            for f in file_list[:3]:
                print(f"    - {f}")
            if len(file_list) > 3:
                print(f"    ... and {len(file_list)-3} more")
    else:
        print("No duplicates found!")
    
    print("\n" + "="*60 + "\n")

# Check both folders
check_duplicates(r'd:\Univer\Nam_3\HKII\DL_DEMO\data\raw_ipn_merged_clean_strict')
check_duplicates(r'd:\Univer\Nam_3\HKII\DL_DEMO\data\raw_ipn_merged_clean')
