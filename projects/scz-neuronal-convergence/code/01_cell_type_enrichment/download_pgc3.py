#!/usr/bin/env python3
"""
Simple PGC3 SCZ GWAS Download Script
====================================

Purpose: Download PGC3 SCZ GWAS summary statistics from Figshare if not already present.

Usage:
    python download_pgc3.py              # Download to default location
    python download_pgc3.py --check-only # Check if file exists without downloading
    python download_pgc3.py --force      # Force re-download even if present

Expected output:
- File: pgc.scz3_2022_EUR.sumstats.gz
- Expected size: ~200-300 MB compressed
- MD5: Not provided by Figshare, but file should be > 100 MB

Location: /mnt/GLaDOS_pool/Iluvatar/biomarvin/schizo/

Author: Marvin
Date: 2026-04-08
"""

import argparse
import gzip
import os
import shutil
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests is required. Install with: pip install requests")
    sys.exit(1)

# Configuration
PGC3_URL = "https://ndownloader.figshare.com/files/34517828"
DATA_DIR = Path("/mnt/GLaDOS_pool/Iluvatar/biomarvin/schizo")
OUTPUT_FILE = DATA_DIR / "pgc.scz3_2022_EUR.sumstats.gz"
CHUNK_SIZE = 8192
MIN_SIZE = 100 * 1024 * 1024  # 100 MB minimum expected


def download_file(url: str, output_path: Path, timeout: int = 600) -> Path:
    """
    Download file with progress indication.

    Args:
        url: Download URL
        output_path: Local path to save file
        timeout: Request timeout in seconds

    Returns:
        Path to downloaded file
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    # Create directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check available disk space
    try:
        stat = os.statvfs(output_path.parent)
        free_space = stat.f_bavail * stat.f_frsize
        print(f"Available disk space: {free_space / (1024**3):.1f} GB")
    except Exception:
        pass

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total_size = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)

                    # Progress every 10 MB
                    if total_size % (10 * 1024 * 1024) == 0:
                        print(f"  Downloaded: {total_size / (1024**2):.1f} MB")

        print(f"Download complete: {total_size / (1024**2):.1f} MB")

        # Verify size
        if total_size < MIN_SIZE:
            print(f"WARNING: Downloaded file is only {total_size / (1024**2):.1f} MB")
            print("Expected size: > 100 MB")
            print("File may be incomplete or corrupt.")

        return output_path

    except requests.exceptions.Timeout:
        print("ERROR: Download timed out. Try again or check your connection.")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error {e.response.status_code}")
        print("The file may have moved or access may be restricted.")
        raise
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        raise


def check_file(file_path: Path) -> bool:
    """
    Check if PGC3 file exists and is valid.

    Returns:
        True if file exists and appears valid
    """
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    size = file_path.stat().st_size
    print(f"File exists: {file_path}")
    print(f"  Size: {size / (1024**2):.1f} MB")

    if size == 0:
        print("  WARNING: File is 0 bytes (placeholder)")
        return False

    if size < MIN_SIZE:
        print(f"  WARNING: File is smaller than expected (< 100 MB)")
        return False

    # Try to open as gzip to verify format
    try:
        with gzip.open(file_path, 'rt') as f:
            # Read first few lines
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  {line[:100].strip()}")
        print("  Format: Valid gzip/gwas format")
        return True
    except Exception as e:
        print(f"  WARNING: Could not read as gzip file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download PGC3 SCZ GWAS summary statistics")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check if file exists, don't download")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if file exists")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Output path (default: {OUTPUT_FILE})")

    args = parser.parse_args()

    output_path = args.output or OUTPUT_FILE

    if args.check_only:
        print("Checking PGC3 file...")
        is_valid = check_file(output_path)
        sys.exit(0 if is_valid else 1)

    if output_path.exists() and not args.force:
        print(f"File already exists: {output_path}")
        is_valid = check_file(output_path)
        if is_valid:
            print("\nFile is valid. Use --force to re-download.")
            sys.exit(0)
        else:
            print("\nFile appears invalid. Re-downloading...")

    try:
        download_file(PGC3_URL, output_path)
        print("\nVerifying downloaded file...")
        check_file(output_path)
        print("\nDownload complete!")

    except Exception as e:
        print(f"\nFailed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
