#!/usr/bin/env python3
"""
Script to organize logs by date.

This script takes existing run folders (formatted as YYYY-MM-DD_HH-MM-SS_description)
and organizes them into date-based folders (YYYY-MM-DD) for better organization.
"""

import os
import shutil
import re
from pathlib import Path
from collections import defaultdict
import argparse

def extract_date_from_folder(folder_name):
    """
    Extract date from folder name with format: YYYY-MM-DD_HH-MM-SS_description
    Returns the date part (YYYY-MM-DD) or None if format doesn't match.
    """
    # Pattern to match YYYY-MM-DD at the start of folder name
    pattern = r'^(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}'
    match = re.match(pattern, folder_name)
    return match.group(1) if match else None

def check_folder_identical(folder1, folder2):
    """
    Check if two folders have identical contents.
    Returns True if they appear to be the same run (based on size and file count).
    """
    try:
        # Quick check: compare total size and file count
        def get_folder_stats(folder_path):
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
                        file_count += 1
            return total_size, file_count
        
        stats1 = get_folder_stats(folder1)
        stats2 = get_folder_stats(folder2)
        
        return stats1 == stats2
    except Exception:
        return False

def organize_logs_by_date(logs_path, dry_run=False, handle_duplicates="skip"):
    """
    Organize log folders by date.
    
    Args:
        logs_path: Path to the logs directory containing run folders
        dry_run: If True, only print what would be done without actually moving files
        handle_duplicates: How to handle duplicate folders ("skip", "remove-duplicates", "rename")
    """
    logs_path = Path(logs_path)
    
    if not logs_path.exists():
        print(f"Error: Logs path {logs_path} does not exist")
        return
    
    # Get all directories in the logs path
    run_folders = [item for item in logs_path.iterdir() if item.is_dir()]
    
    if not run_folders:
        print(f"No folders found in {logs_path}")
        return
    
    # Group folders by date
    date_groups = defaultdict(list)
    unmatched_folders = []
    
    for folder in run_folders:
        date = extract_date_from_folder(folder.name)
        if date:
            date_groups[date].append(folder)
        else:
            unmatched_folders.append(folder)
    
    print(f"Found {len(run_folders)} folders total")
    print(f"Matched {sum(len(folders) for folders in date_groups.values())} folders to dates")
    print(f"Unmatched folders: {len(unmatched_folders)}")
    
    if unmatched_folders:
        print("\nUnmatched folders:")
        for folder in unmatched_folders:
            print(f"  {folder.name}")
    
    print(f"\nDate groups found:")
    for date, folders in sorted(date_groups.items()):
        print(f"  {date}: {len(folders)} runs")
    
    if dry_run:
        print("\n=== DRY RUN - No files will be moved ===")
    
    total_moved = 0
    total_duplicates_removed = 0
    total_skipped = 0
    
    # Create date folders and move run folders
    for date, folders in sorted(date_groups.items()):
        date_folder = logs_path / date
        
        if dry_run:
            print(f"\nWould create folder: {date_folder}")
        else:
            # Create date folder if it doesn't exist
            date_folder.mkdir(exist_ok=True)
            print(f"\nProcessing date folder: {date_folder}")
        
        # Move run folders into date folder
        for folder in folders:
            destination = date_folder / folder.name
            
            if destination.exists():
                if handle_duplicates == "skip":
                    if not dry_run:
                        print(f"  Warning: {destination} already exists, skipping {folder.name}")
                    total_skipped += 1
                    continue
                elif handle_duplicates == "remove-duplicates":
                    if dry_run:
                        print(f"  Would check if {folder.name} is duplicate and remove if identical")
                    else:
                        if check_folder_identical(folder, destination):
                            try:
                                shutil.rmtree(str(folder))
                                print(f"  Removed duplicate: {folder.name}")
                                total_duplicates_removed += 1
                                continue
                            except Exception as e:
                                print(f"  Error removing duplicate {folder.name}: {e}")
                                total_skipped += 1
                                continue
                        else:
                            print(f"  Warning: {destination} exists but differs from {folder.name}, skipping")
                            total_skipped += 1
                            continue
                elif handle_duplicates == "rename":
                    # Find a new name with suffix
                    counter = 1
                    new_destination = destination
                    while new_destination.exists():
                        new_destination = date_folder / f"{folder.name}_copy{counter}"
                        counter += 1
                    destination = new_destination
                    if dry_run:
                        print(f"  Would move: {folder.name} -> {destination.name}")
                    # Fall through to move operation below
                    
            if dry_run and handle_duplicates != "remove-duplicates":
                print(f"  Would move: {folder.name} -> {destination.name}")
            elif not dry_run:
                try:
                    shutil.move(str(folder), str(destination))
                    print(f"  Moved: {folder.name} -> {destination.name}")
                    total_moved += 1
                except Exception as e:
                    print(f"  Error moving {folder.name}: {e}")
                    total_skipped += 1
    
    if not dry_run:
        print(f"\nOrganization complete!")
        print(f"  Moved: {total_moved} folders")
        print(f"  Duplicates removed: {total_duplicates_removed} folders")
        print(f"  Skipped: {total_skipped} folders")
        print(f"Check {logs_path} for date-organized folders.")
    else:
        print(f"\nDry run complete. Use --execute to actually move the folders.")

def main():
    parser = argparse.ArgumentParser(description="Organize log folders by date")
    parser.add_argument("logs_path", nargs="?", 
                       default="logs/rl_games/mars_jumper",
                       help="Path to logs directory (default: logs/rl_games/mars_jumper)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without actually moving files")
    parser.add_argument("--execute", action="store_true",
                       help="Actually execute the organization (default is dry-run)")
    parser.add_argument("--handle-duplicates", choices=["skip", "remove-duplicates", "rename"],
                       default="skip",
                       help="How to handle duplicate folders: skip them, remove duplicates from root, or rename with suffix")
    
    args = parser.parse_args()
    
    # Default to dry-run unless --execute is specified
    dry_run = not args.execute
    
    if dry_run and not args.dry_run:
        print("Running in dry-run mode by default. Use --execute to actually move files.")
        print("Use --dry-run to explicitly run in dry-run mode.\n")
    
    organize_logs_by_date(args.logs_path, dry_run=dry_run, handle_duplicates=args.handle_duplicates)

if __name__ == "__main__":
    main() 