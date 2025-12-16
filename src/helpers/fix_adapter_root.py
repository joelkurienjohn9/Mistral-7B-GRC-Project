#!/usr/bin/env python
# coding: utf-8

"""
Fix Adapter Root Files
======================
This script copies adapter files from the latest checkpoint to the root directory
when the root directory is missing required files (adapter_config.json, adapter_model.safetensors, etc.)

This commonly happens when training is interrupted or save_model() doesn't properly save to root.

Usage:
    # Interactive mode - scan and fix all adapters
    python src/helpers/fix_adapter_root.py

    # Fix a specific adapter
    python src/helpers/fix_adapter_root.py --adapter ./adapters/my_adapter

    # Dry run (show what would be copied without copying)
    python src/helpers/fix_adapter_root.py --dry-run

    # Non-interactive mode (auto-fix all)
    python src/helpers/fix_adapter_root.py --auto-fix
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import json

# Required files for a valid adapter
REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",  # or adapter_model.bin
]

OPTIONAL_FILES = [
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "README.md",
]

def find_latest_checkpoint(adapter_path):
    """Find the latest checkpoint directory in an adapter path."""
    checkpoint_dirs = [d for d in Path(adapter_path).glob("checkpoint-*") if d.is_dir()]
    
    if not checkpoint_dirs:
        return None
    
    # Sort by checkpoint number and get the latest
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
    return checkpoint_dirs[-1]

def check_adapter_validity(adapter_path):
    """Check if adapter root has required files."""
    adapter_path = Path(adapter_path)
    
    # Check for adapter_config.json (critical)
    has_config = (adapter_path / "adapter_config.json").exists()
    
    # Check for model file (either .safetensors or .bin)
    has_model = (
        (adapter_path / "adapter_model.safetensors").exists() or
        (adapter_path / "adapter_model.bin").exists()
    )
    
    return has_config and has_model

def get_checkpoint_files(checkpoint_path):
    """Get list of important files in checkpoint directory."""
    checkpoint_path = Path(checkpoint_path)
    files = []
    
    # Check for required files
    for filename in REQUIRED_FILES:
        file_path = checkpoint_path / filename
        if file_path.exists():
            files.append(filename)
    
    # Also check for .bin version of model
    if "adapter_model.safetensors" not in files:
        bin_path = checkpoint_path / "adapter_model.bin"
        if bin_path.exists():
            files.append("adapter_model.bin")
    
    # Check for optional files
    for filename in OPTIONAL_FILES:
        file_path = checkpoint_path / filename
        if file_path.exists():
            files.append(filename)
    
    return files

def fix_adapter_root(adapter_path, dry_run=False, verbose=True):
    """Fix adapter root by copying files from latest checkpoint.
    
    Args:
        adapter_path: Path to adapter directory
        dry_run: If True, only show what would be copied without copying
        verbose: If True, print detailed information
    
    Returns:
        bool: True if fixed successfully, False otherwise
    """
    adapter_path = Path(adapter_path)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing: {adapter_path.name}")
        print(f"{'='*70}")
    
    # Check if already valid
    if check_adapter_validity(adapter_path):
        if verbose:
            print("✅ Adapter root already has required files - no fix needed")
        return True
    
    if verbose:
        print("⚠️  Adapter root is missing required files")
    
    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(adapter_path)
    
    if latest_checkpoint is None:
        if verbose:
            print("❌ No checkpoints found - cannot fix")
        return False
    
    if verbose:
        print(f"📁 Found latest checkpoint: {latest_checkpoint.name}")
    
    # Get files to copy
    files_to_copy = get_checkpoint_files(latest_checkpoint)
    
    if not files_to_copy:
        if verbose:
            print("❌ No valid files found in checkpoint")
        return False
    
    # Check if checkpoint has required files
    has_config = "adapter_config.json" in files_to_copy
    has_model = any(f in files_to_copy for f in ["adapter_model.safetensors", "adapter_model.bin"])
    
    if not (has_config and has_model):
        if verbose:
            print(f"❌ Checkpoint is missing required files:")
            if not has_config:
                print("   - adapter_config.json")
            if not has_model:
                print("   - adapter_model.safetensors or adapter_model.bin")
        return False
    
    if verbose:
        print(f"\n📋 Files to copy ({len(files_to_copy)}):")
        for filename in files_to_copy:
            src_path = latest_checkpoint / filename
            file_size = src_path.stat().st_size
            size_str = f"{file_size / (1024**2):.2f} MB" if file_size > 1024**2 else f"{file_size / 1024:.2f} KB"
            print(f"   - {filename} ({size_str})")
    
    if dry_run:
        print("\n🔍 DRY RUN - No files will be copied")
        return True
    
    # Copy files
    if verbose:
        print("\n📦 Copying files...")
    
    copied_count = 0
    for filename in files_to_copy:
        src_path = latest_checkpoint / filename
        dst_path = adapter_path / filename
        
        try:
            shutil.copy2(src_path, dst_path)
            if verbose:
                print(f"   ✓ Copied: {filename}")
            copied_count += 1
        except Exception as e:
            if verbose:
                print(f"   ✗ Failed to copy {filename}: {e}")
    
    if verbose:
        print(f"\n✅ Successfully copied {copied_count}/{len(files_to_copy)} files")
    
    # Verify fix
    if check_adapter_validity(adapter_path):
        if verbose:
            print("✅ Adapter root is now valid!")
        return True
    else:
        if verbose:
            print("⚠️  Adapter root still missing some files")
        return False

def scan_adapters_directory(base_dir="./adapters"):
    """Scan adapters directory and return list of adapters needing fixes."""
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        print(f"❌ Adapters directory not found: {base_dir}")
        return []
    
    # Find all adapter directories
    adapter_dirs = [d for d in base_dir.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
    
    if not adapter_dirs:
        print(f"ℹ️  No adapters found in {base_dir}")
        return []
    
    print(f"\n🔍 Scanning {len(adapter_dirs)} adapter(s) in {base_dir}...")
    
    needs_fix = []
    valid = []
    no_checkpoints = []
    
    for adapter_dir in adapter_dirs:
        if check_adapter_validity(adapter_dir):
            valid.append(adapter_dir)
        else:
            # Check if it has checkpoints
            if find_latest_checkpoint(adapter_dir):
                needs_fix.append(adapter_dir)
            else:
                no_checkpoints.append(adapter_dir)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SCAN RESULTS")
    print(f"{'='*70}")
    print(f"✅ Valid adapters: {len(valid)}")
    if valid:
        for adapter in valid:
            print(f"   - {adapter.name}")
    
    print(f"\n⚠️  Adapters needing fix: {len(needs_fix)}")
    if needs_fix:
        for adapter in needs_fix:
            latest = find_latest_checkpoint(adapter)
            print(f"   - {adapter.name} (has {latest.name})")
    
    print(f"\n❌ Adapters without checkpoints: {len(no_checkpoints)}")
    if no_checkpoints:
        for adapter in no_checkpoints:
            print(f"   - {adapter.name}")
    
    return needs_fix

def main():
    parser = argparse.ArgumentParser(
        description="Fix adapter root directories by copying files from latest checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - scan and select adapters to fix
  python src/helpers/fix_adapter_root.py
  
  # Fix a specific adapter
  python src/helpers/fix_adapter_root.py --adapter ./adapters/my_adapter
  
  # Dry run (preview without copying)
  python src/helpers/fix_adapter_root.py --dry-run
  
  # Auto-fix all adapters needing fixes
  python src/helpers/fix_adapter_root.py --auto-fix
  
  # Scan custom directory
  python src/helpers/fix_adapter_root.py --adapters-dir ./my_adapters
        """
    )
    
    parser.add_argument(
        "--adapter",
        type=str,
        help="Path to specific adapter to fix"
    )
    
    parser.add_argument(
        "--adapters-dir",
        type=str,
        default="./adapters",
        help="Base directory containing adapters (default: ./adapters)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes"
    )
    
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically fix all adapters without prompting"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize output (only show errors)"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if verbose:
        print("="*70)
        print("🔧 Adapter Root Fix Utility")
        print("="*70)
    
    # Fix specific adapter
    if args.adapter:
        adapter_path = Path(args.adapter)
        
        if not adapter_path.exists():
            print(f"❌ Adapter not found: {adapter_path}")
            return 1
        
        success = fix_adapter_root(adapter_path, dry_run=args.dry_run, verbose=verbose)
        return 0 if success else 1
    
    # Scan and fix adapters directory
    needs_fix = scan_adapters_directory(args.adapters_dir)
    
    if not needs_fix:
        if verbose:
            print("\n✅ All adapters are valid - nothing to fix!")
        return 0
    
    # Auto-fix mode
    if args.auto_fix:
        print(f"\n🔧 Auto-fixing {len(needs_fix)} adapter(s)...")
        
        success_count = 0
        for adapter_dir in needs_fix:
            if fix_adapter_root(adapter_dir, dry_run=args.dry_run, verbose=verbose):
                success_count += 1
        
        print(f"\n{'='*70}")
        print(f"✅ Successfully fixed {success_count}/{len(needs_fix)} adapter(s)")
        print(f"{'='*70}")
        
        return 0 if success_count == len(needs_fix) else 1
    
    # Interactive mode
    if not args.dry_run:
        print(f"\n❓ Fix {len(needs_fix)} adapter(s)?")
        print("   1. Fix all")
        print("   2. Select which ones to fix")
        print("   3. Cancel")
        
        choice = input("\nYour choice (1-3): ").strip()
        
        if choice == "1":
            print(f"\n🔧 Fixing {len(needs_fix)} adapter(s)...")
            success_count = 0
            for adapter_dir in needs_fix:
                if fix_adapter_root(adapter_dir, dry_run=False, verbose=verbose):
                    success_count += 1
            
            print(f"\n{'='*70}")
            print(f"✅ Successfully fixed {success_count}/{len(needs_fix)} adapter(s)")
            print(f"{'='*70}")
            
        elif choice == "2":
            print("\n📝 Select adapters to fix (comma-separated numbers):")
            for i, adapter in enumerate(needs_fix, 1):
                print(f"   {i}. {adapter.name}")
            
            selection = input("\nEnter numbers (e.g., 1,2,3): ").strip()
            
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',') if x.strip()]
                selected = [needs_fix[i] for i in indices if 0 <= i < len(needs_fix)]
                
                if selected:
                    print(f"\n🔧 Fixing {len(selected)} adapter(s)...")
                    success_count = 0
                    for adapter_dir in selected:
                        if fix_adapter_root(adapter_dir, dry_run=False, verbose=verbose):
                            success_count += 1
                    
                    print(f"\n{'='*70}")
                    print(f"✅ Successfully fixed {success_count}/{len(selected)} adapter(s)")
                    print(f"{'='*70}")
                else:
                    print("⚠️  No valid adapters selected")
            
            except (ValueError, IndexError) as e:
                print(f"❌ Invalid selection: {e}")
                return 1
        
        else:
            print("❌ Cancelled")
            return 0
    else:
        print("\n🔍 DRY RUN complete - use without --dry-run to apply fixes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

