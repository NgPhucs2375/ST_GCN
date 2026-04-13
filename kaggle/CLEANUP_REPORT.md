# 📋 Project Cleanup Report - Unnecessary Files

## 🗂️ Files to Delete (Total: 19 files)

### ❌ Category 1: Temporary Output Files (5 files)
These are temporary outputs from development - **SAFE TO DELETE**

```
❌ tmp_compare_help.txt
   └─ Purpose: Help text output from compare_checkpoints.py
   └─ Size: ~1 KB
   └─ Usage: None (just temporary capture)
   └─ Delete: YES ✗

❌ tmp_compare_results.txt
   └─ Purpose: Results from model comparison
   └─ Size: ~2 KB
   └─ Usage: Already analyzed
   └─ Delete: YES ✗

❌ tmp_compare_strict_results.txt
   └─ Purpose: Strict mode comparison results
   └─ Size: ~1 KB
   └─ Usage: Already analyzed
   └─ Delete: YES ✗

❌ tmp_staged_files.txt
   └─ Purpose: Git staging status capture
   └─ Size: ~5 KB
   └─ Usage: Git handles this internally
   └─ Delete: YES ✗

❌ tmp_train_help_new.txt
   └─ Purpose: Train.py help output
   └─ Size: ~3 KB
   └─ Usage: Help is in --help flag
   └─ Delete: YES ✗
```

---

### ❌ Category 2: Obsolete Data Processing Scripts (3 files)
Old versions of extraction scripts - **KEEP ONLY LATEST**

```
❌ extract_videos_simple.py
   └─ Purpose: Simple video extraction (OLD VERSION)
   └─ Latest version: extract_videos_final.py ✓
   └─ Difference: extract_videos_final.py has all fixes
   └─ Delete: YES ✗ (use extract_videos_final.py instead)

❌ extract_videos_v2.py
   └─ Purpose: V2 of video extraction (OLD VERSION)
   └─ Latest version: extract_videos_final.py ✓
   └─ Difference: extract_videos_final.py is most recent
   └─ Delete: YES ✗ (use extract_videos_final.py instead)

✓ extract_videos_final.py
   └─ Purpose: Final, optimized video extraction
   └─ Status: KEEP THIS ONE ✓✓✓
   └─ Delete: NO
```

---

### ❌ Category 3: Old/Merged Scripts (3 files)
These were used for intermediate steps but are now obsolete - **SAFE TO DELETE**

```
❌ merge_datasets.py
   └─ Purpose: Merge train.npz + extracted files
   └─ Status: Already merged! (train_merged_t60_accel.npz created)
   └─ Current use: None (already executed)
   └─ Delete: YES ✗ (final NPZ already exists)

❌ compare_models.py
   └─ Purpose: Compare different model checkpoints
   └─ Current use: None (not in training pipeline)
   └─ Note: Replaced by analyze_confusion_matrix.py (better)
   └─ Delete: YES ✗ (use analyze_confusion_matrix.py instead)

❌ sanity_check.py
   └─ Purpose: Verify setup before training
   └─ Status: Old setup verification (deprecated)
   └─ Current use: Not needed (requirements.txt + Kaggle handles setup)
   └─ Delete: YES ✗ (Kaggle environment is trusted)
```

---

### ❌ Category 4: Old Batch/Shell Scripts (2 files)
These are old test scripts - **SAFE TO DELETE**

```
❌ run_train_t60.bat
   └─ Purpose: Windows batch to run training locally
   └─ Current use: None (will train on Kaggle)
   └─ Note: Local torch_geometric issues, use Kaggle instead
   └─ Delete: YES ✗ (use Kaggle notebook instead)

❌ run_train_t60.sh
   └─ Purpose: Linux shell script for training
   └─ Current use: None (will train on Kaggle)
   └─ Note: Replaced by Kaggle notebook+command
   └─ Delete: YES ✗ (use Kaggle notebook instead)
```

---

### ❌ Category 5: Python Cache Files (3+ files)
Automatic generated cache - **SAFE TO DELETE**

```
❌ __pycache__/ (root directory)
   └─ Size: ~500 KB
   └─ Purpose: Python bytecode cache
   └─ Auto-generated: YES (Python creates automatically)
   └─ Delete: YES ✗ (can regenerate with: rm -rf __pycache__)

❌ tools/__pycache__/
   └─ Size: ~100 KB
   └─ Purpose: Python bytecode cache
   └─ Auto-generated: YES
   └─ Delete: YES ✗

❌ .venv/ (virtual environment)
   └─ Size: ~200+ MB (LARGEST!)
   └─ Purpose: Local virtual environment (not needed for Kaggle)
   └─ Current use: None (Kaggle has its own environment)
   └─ Delete: YES ✗ (Kaggle will use its own venv)
```

---

### ❌ Category 6: Old Model Checkpoints
These are old training outputs - **CHECK IF NEEDED**

```
⚠️  outputs/stgcn_last.pt
   └─ Purpose: Last checkpoint from old training
   └─ Status: Obsolete (newer models in outputs_resume/)
   └─ Size: ~25 MB
   └─ Delete: YES? (unless you want to keep history)

⚠️  final_model/final_model_v2.zip
   └─ Purpose: Zipped old model
   └─ Status: Backup of final_model_v2/
   └─ Size: ~30 MB
   └─ Duplicate of: final_model/final_model_v2/
   └─ Delete: YES ✗ (keep folder, delete .zip - it's redundant)

⚠️  outputs_resume/stgcn_last.pt
   └─ Purpose: Last checkpoint from resumed training
   └─ Status: Could be useful as backup
   └─ Usage: Backup (keep if want history)
   └─ Can delete if: space matters and only need stgcn_best.pt
```

---

### ⚠️ Category 7: Optional Documentation (May be outdated)

```
⚠️  KAGGLE_READY_CHECKLIST.md
   └─ Status: References sanity_check.py (which is old)
   └─ Recommendation: Update or delete sanity_check reference

⚠️  docs/GIAITHICH_EX.txt
   └─ Purpose: Unknown (Vietnamese text file)
   └─ Status: Check if still relevant
   └─ Recommendation: Review or archive
```

---

## 📊 Summary Table

| File/Folder | Type | Size | Status | Action |
|-------------|------|------|--------|--------|
| tmp_*.txt (5) | Temp | ~12 KB | ❌ Delete | YES |
| extract_videos_simple.py | Old code | ~8 KB | ❌ Delete | YES |
| extract_videos_v2.py | Old code | ~6 KB | ❌ Delete | YES |
| merge_datasets.py | Old code | ~4 KB | ❌ Delete | YES |
| compare_models.py | Old code | ~5 KB | ❌ Delete | YES |
| sanity_check.py | Old code | ~4 KB | ❌ Delete | YES |
| run_train_t60.bat | Script | ~0.5 KB | ❌ Delete | YES |
| run_train_t60.sh | Script | ~0.5 KB | ❌ Delete | YES |
| __pycache__/ | Cache | ~600 KB | ❌ Delete | YES |
| .venv/ | Package | 200+ MB | ❌ Keep (LOCAL) | - |
| final_model_v2.zip | Backup | ~30 MB | ⚠️  Delete | MAYBE |
| outputs/stgcn_last.pt | Checkpoint | ~25 MB | ⚠️  Keep? | MAYBE |
| outputs_resume/stgcn_last.pt | Checkpoint | ~25 MB | ⚠️  Keep? | MAYBE |
| **TOTAL SAFE TO DELETE** | | **~60 MB** | | **YES** |

---

## 🗑️ Cleanup Commands

### Delete All Temporary Files (SAFE)
```bash
# Windows (PowerShell)
rm .\tmp_*.txt
rm .\extract_videos_simple.py
rm .\extract_videos_v2.py
rm .\merge_datasets.py
rm .\compare_models.py
rm .\sanity_check.py
rm .\run_train_t60.bat
rm .\run_train_t60.sh
rm -r .\__pycache__
rm -r .\tools\__pycache__

# Or in one command:
rm .\tmp_*.txt, .\extract_videos_*.py, .\merge_*.py, .\compare_*.py, .\sanity_*.py, .\run_train_*.* -Force

# Total freed: ~65 MB (mostly pycache)
```

### Delete Optional Backups
```bash
# If you don't need training history:
rm ".\final_model\final_model_v2.zip"          # Keep folder, delete zip
rm ".\outputs\stgcn_last.pt"                   # Old checkpoint
rm ".\outputs_resume\stgcn_last.pt"            # Last checkpoint (keep best)

# Total freed: ~80 MB
```

### Keep These
```bash
✓ extract_videos_final.py        # Latest version - NEEDED
✓ outputs_resume/stgcn_best.pt   # Best checkpoint - NEEDED
✓ outputs_kaggle/*               # Kaggle training outputs - NEEDED when created
✓ data/processed/train_merged_t60_accel.npz  # Final dataset - NEEDED
✓ All .md documentation files    # For reference - KEEP
```

---

## 🎯 Recommendations

### Immediate Cleanup (SAFE - FREE 65 MB)
```
Delete:
├─ All tmp_*.txt files
├─ extract_videos_simple.py
├─ extract_videos_v2.py
├─ merge_datasets.py
├─ compare_models.py
├─ sanity_check.py
├─ run_train_t60.bat
├─ run_train_t60.sh
└─ __pycache__ directories
```

### Optional Cleanup (FREE ADDITIONAL 80 MB)
```
Delete if you don't need history:
├─ final_model_v2.zip (keep folder, delete zip)
├─ outputs/stgcn_last.pt
└─ outputs_resume/stgcn_last.pt (keep best, delete last)
```

### Structure After Cleanup
```
DL_DEMO/
├── README.md
├── SETUP.md
├── requirements.txt
├── train.py                               ✓ KEEP
├── DATA_STRUCTURE.md                      ✓ KEEP
│
├── models/                                ✓ KEEP
├── dataset/                               ✓ KEEP
├── tools/
│   ├── extract_videos_final.py            ✓ KEEP (latest)
│   ├── analyze_confusion_matrix.py        ✓ KEEP
│   ├── augment_balance.py                 ✓ KEEP
│   ├── convert_sequences.py               ✓ KEEP
│   ├── data_quality.py                    ✓ KEEP
│   ├── demo_webcam.py                     ✓ KEEP
│   ├── infer.py                           ✓ KEEP
│   └── inspect_confusion.py               ✓ KEEP
│
├── data/processed/
│   └── train_merged_t60_accel.npz         ✓ KEEP (final dataset)
│
├── docs/                                  ✓ KEEP
├── [ALL MARKDOWN GUIDES]                  ✓ KEEP (documentation)
│
├── outputs_resume/
│   └── stgcn_best.pt                      ✓ KEEP (best checkpoint)
│
└── outputs_kaggle/                        (will be created on Kaggle)
    ├── stgcn_best.pt
    ├── confusion_matrix.pt
    └── labels.json
```

---

## 🚀 Summary

| Category | Files | Action | Space |
|----------|-------|--------|-------|
| **Temp files** | 5 | DELETE | 12 KB |
| **Old scripts** | 3+2 | DELETE | 15 KB |
| **Cache** | 3+ | DELETE | 600 KB |
| **Old checkpoints** | 2-3 | DELETE? | 80 MB |
| **TOTAL SAFE** | **13** | **DELETE** | **~65 MB** |
| **TOTAL OPTIONAL** | **3** | **DELETE** | **~80 MB** |
| **TOTAL POSSIBLE** | **16** | | **~145 MB** |

---

## ✅ Ready for Kaggle?

After cleanup:
- ✅ All scripts needed for training present
- ✅ Latest dataset ready: train_merged_t60_accel.npz
- ✅ All documentation in place
- ✅ No clutter or obsolete files
- ✅ Clean git history (if committed)

Good to go! 🚀
