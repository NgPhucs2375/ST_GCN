#!/usr/bin/env python3
"""
Project Cleanup Script - Remove unnecessary files
Usage: python cleanup_project.py [--dry-run] [--aggressive]
"""

import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime


class ProjectCleaner:
    """Clean unnecessary files from project"""
    
    def __init__(self, project_root="."):
        self.root = Path(project_root)
        self.deleted_files = []
        self.deleted_dirs = []
        self.freed_space = 0
    
    # Files to always delete (safe)
    SAFE_DELETE_FILES = [
        "tmp_compare_help.txt",
        "tmp_compare_results.txt",
        "tmp_compare_strict_results.txt",
        "tmp_staged_files.txt",
        "tmp_train_help_new.txt",
        "extract_videos_simple.py",
        "extract_videos_v2.py",
        "merge_datasets.py",
        "compare_models.py",
        "sanity_check.py",
        "run_train_t60.bat",
        "run_train_t60.sh",
    ]
    
    # Cache directories to delete
    CACHE_DIRS = [
        "__pycache__",
        "tools/__pycache__",
        ".pytest_cache",
        ".mypy_cache",
    ]
    
    # Optional files (ask first)
    OPTIONAL_DELETE_FILES = [
        "final_model/final_model_v2.zip",
        "outputs/stgcn_last.pt",
        "outputs_resume/stgcn_last.pt",
    ]
    
    # Files/dirs to KEEP
    KEEP_FILES = [
        "extract_videos_final.py",
        "outputs_resume/stgcn_best.pt",
        "data/processed/train_merged_t60_accel.npz",
    ]
    
    def get_file_size(self, path):
        """Get size of file or directory in MB"""
        try:
            if path.is_file():
                return path.stat().st_size / (1024 * 1024)
            else:
                total = 0
                for item in path.rglob("*"):
                    if item.is_file():
                        total += item.stat().st_size
                return total / (1024 * 1024)
        except:
            return 0
    
    def delete_file(self, filepath, dry_run=False):
        """Delete a single file"""
        path = self.root / filepath
        if not path.exists():
            return False
        
        if path.is_file():
            size = self.get_file_size(path)
            if not dry_run:
                path.unlink()
            self.deleted_files.append(str(path))
            self.freed_space += size
            print(f"  ✗ {filepath:<50} ({size:.2f} MB)")
            return True
        return False
    
    def delete_directory(self, dirpath, dry_run=False):
        """Delete a directory recursively"""
        path = self.root / dirpath
        if not path.exists():
            return False
        
        if path.is_dir():
            size = self.get_file_size(path)
            if not dry_run:
                shutil.rmtree(path)
            self.deleted_dirs.append(str(path))
            self.freed_space += size
            print(f"  ✗ {dirpath:<50} ({size:.2f} MB)")
            return True
        return False
    
    def clean_safe(self, dry_run=False):
        """Delete safe-to-delete files"""
        print("\n📋 SAFE TO DELETE (always safe):")
        print("=" * 80)
        
        # Delete single files
        for filename in self.SAFE_DELETE_FILES:
            self.delete_file(filename, dry_run)
        
        # Delete cache directories
        for dirname in self.CACHE_DIRS:
            self.delete_directory(dirname, dry_run)
        
        # Find and delete any remaining __pycache__
        for pycache_dir in self.root.rglob("__pycache__"):
            size = self.get_file_size(pycache_dir)
            if not dry_run:
                shutil.rmtree(pycache_dir)
            self.deleted_dirs.append(str(pycache_dir))
            self.freed_space += size
            print(f"  ✗ {pycache_dir.relative_to(self.root):<50} ({size:.2f} MB)")
    
    def clean_optional(self, dry_run=False, confirm=False):
        """Delete optional files (ask permission)"""
        print("\n❓ OPTIONAL TO DELETE (backups/old checkpoints):")
        print("=" * 80)
        
        for filename in self.OPTIONAL_DELETE_FILES:
            path = self.root / filename
            if path.exists():
                size = self.get_file_size(path)
                
                if confirm:
                    self.delete_file(filename, dry_run)
                else:
                    response = input(f"Delete {filename}? ({size:.2f} MB) [y/N]: ")
                    if response.lower() == 'y':
                        self.delete_file(filename, dry_run)
                    else:
                        print(f"  - {filename:<50} (skipped)")
    
    def clean_venv(self, dry_run=False, confirm=False):
        """Delete virtual environment (largest, usually not needed)"""
        venv_paths = [".venv", "venv", "env"]
        
        print("\n⚠️  VIRTUAL ENVIRONMENT (largest, ~200+ MB):")
        print("=" * 80)
        
        for venv_name in venv_paths:
            path = self.root / venv_name
            if path.exists():
                size = self.get_file_size(path)
                print(f"Found: {venv_name} ({size:.2f} MB)")
                
                if confirm:
                    self.delete_directory(venv_name, dry_run)
                else:
                    response = input(f"Delete {venv_name}? ({size:.2f} MB) [y/N]: ")
                    if response.lower() == 'y':
                        self.delete_directory(venv_name, dry_run)
                    else:
                        print(f"  - {venv_name:<50} (skipped)")
    
    def print_summary(self, dry_run=False, clean_level="safe"):
        """Print cleanup summary"""
        print("\n" + "=" * 80)
        if dry_run:
            print("📊 DRY RUN SUMMARY (no files actually deleted):")
        else:
            print("✅ CLEANUP COMPLETE:")
        print("=" * 80)
        
        print(f"\nFiles deleted: {len(self.deleted_files)}")
        print(f"Directories deleted: {len(self.deleted_dirs)}")
        print(f"Space freed: {self.freed_space:.2f} MB")
        
        if clean_level == "safe":
            print(f"\n📌 Note: Deleted safe files only (~65 MB)")
            print(f"Run with --aggressive to also delete backups/old checkpoints")
        elif clean_level == "aggressive":
            print(f"\n🗑️  Deleted safe + optional, kept best checkpoints")
        
        if dry_run:
            print("\n💾 Run without --dry-run to actually delete files")
        
        print("=" * 80)
    
    def run(self, dry_run=False, aggressive=False, venv=False, auto_confirm=False):
        """Run cleanup"""
        print(f"\n🧹 PROJECT CLEANUP SCRIPT")
        print(f"Project root: {self.root}")
        print(f"Dry run: {dry_run}")
        print(f"Aggressive: {aggressive}")
        print(f"Include venv: {venv}")
        
        # Clean safe files
        self.clean_safe(dry_run)
        
        # Clean optional files
        if aggressive:
            self.clean_optional(dry_run, confirm=auto_confirm)
        
        # Clean venv
        if venv:
            self.clean_venv(dry_run, confirm=auto_confirm)
        
        # Summary
        clean_level = "aggressive" if aggressive else "safe"
        self.print_summary(dry_run, clean_level)


def main():
    parser = argparse.ArgumentParser(
        description="Clean unnecessary files from project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_project.py --dry-run          # See what would be deleted
  python cleanup_project.py                    # Delete safe files only
  python cleanup_project.py --aggressive       # Also delete backups
  python cleanup_project.py --aggressive --venv # Also delete .venv
  python cleanup_project.py -y                 # Auto-confirm, no prompts
        """
    )
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview deletions without actually deleting')
    parser.add_argument('--aggressive', action='store_true',
                       help='Also delete optional files (backups, old checkpoints)')
    parser.add_argument('--venv', action='store_true',
                       help='Also delete virtual environment (.venv, venv, env)')
    parser.add_argument('-y', '--yes', action='store_true', dest='auto_confirm',
                       help='Auto-confirm all deletions, no prompts')
    parser.add_argument('--root', type=str, default='.',
                       help='Project root directory (default: current)')
    
    args = parser.parse_args()
    
    # Confirm before aggressive deletion without dry-run
    if args.aggressive and not args.dry_run and not args.auto_confirm:
        print("\n⚠️  Aggressive mode will delete backup files!")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    cleaner = ProjectCleaner(args.root)
    cleaner.run(
        dry_run=args.dry_run,
        aggressive=args.aggressive,
        venv=args.venv,
        auto_confirm=args.auto_confirm
    )


if __name__ == '__main__':
    main()
