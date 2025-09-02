#!/usr/bin/env python3
"""
Check CSV files for names not present in SCB Swedish name lists.
"""

import sys
import csv
import argparse
from pathlib import Path
import pickle
from datetime import datetime
import json
from urllib.request import urlopen
from urllib.error import URLError
from typing import Set, Tuple, List, Dict, Optional


def get_cache_dir():
    """Get or create a cache directory for storing name lists."""
    cache_dir = Path.home() / '.cache' / 'scb_names'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def fetch_json_from_url(url, verbosity=0):
    """Fetch JSON data from a URL."""
    try:
        if verbosity > 0:
            print(f"  Fetching data from {url}...", file=sys.stderr)

        with urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))

        return data
    except URLError as e:
        if verbosity >= 0:
            print(f"  Warning: Failed to fetch from {url}: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        if verbosity >= 0:
            print(f"  Warning: Invalid JSON from {url}: {e}", file=sys.stderr)
        return None


def process_first_names(json_data, verbosity=0):
    """Process first names JSON data according to requirements."""
    if not json_data or 'variables' not in json_data:
        return set()

    try:
        # Get the values array from variables[0]
        values = json_data['variables'][0]['values']

        # Process each name
        processed_names = set()
        for name in values:
            # Strip trailing whitespace
            name = name.rstrip()

            # Remove last M or K character
            if name and name[-1] in ['M', 'K']:
                name = name[:-1].rstrip()

            if name:  # Only add non-empty names
                processed_names.add(name)

        if verbosity > 1:
            print(f"    Processed {len(processed_names)} unique first names", file=sys.stderr)

        return processed_names
    except (KeyError, IndexError, TypeError) as e:
        if verbosity >= 0:
            print(f"  Warning: Error processing first names JSON: {e}", file=sys.stderr)
        return set()


def process_last_names(json_data, verbosity=0):
    """Process last names JSON data according to requirements."""
    if not json_data or 'variables' not in json_data:
        return set()

    try:
        # Get the values array from variables[0]
        values = json_data['variables'][0]['values']

        # Process each name
        processed_names = set()
        for name in values:
            # Strip trailing whitespace
            name = name.rstrip()

            if name:  # Only add non-empty names
                processed_names.add(name)

        if verbosity > 1:
            print(f"    Processed {len(processed_names)} unique last names", file=sys.stderr)

        return processed_names
    except (KeyError, IndexError, TypeError) as e:
        if verbosity >= 0:
            print(f"  Warning: Error processing last names JSON: {e}", file=sys.stderr)
        return set()


def load_or_fetch_name_lists(force_refresh=False, verbosity=0):
    """Load name lists from cache or fetch from SCB API."""
    cache_dir = get_cache_dir()
    cache_file = cache_dir / 'scb_names.pkl'

    # URLs for the APIs
    first_names_url = "https://api.scb.se/OV0104/v1/doris/en/ssd/BE/BE0001/BE0001G/BE0001FNamn10"
    last_names_url = "https://api.scb.se/OV0104/v1/doris/en/ssd/BE/BE0001/BE0001G/BE0001ENamn10"

    # Try to load from cache first
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Check cache version and validity
            if cache_data.get('version') == '1.0':
                first_names = cache_data.get('first_names', set())
                last_names = cache_data.get('last_names', set())

                if verbosity >= 0:
                    print(f"Loaded SCB name lists from cache: {len(first_names)} first names, {len(last_names)} last names", 
                          file=sys.stderr)

                return first_names, last_names
        except (pickle.PickleError, KeyError, EOFError) as e:
            if verbosity >= 0:
                print(f"Warning: Cache file corrupted, will fetch fresh data: {e}", file=sys.stderr)

    # Fetch fresh data
    if verbosity >= 0:
        print("Fetching Swedish name lists from SCB API...", file=sys.stderr)

    # Fetch first names
    first_names_json = fetch_json_from_url(first_names_url, verbosity)
    first_names = process_first_names(first_names_json, verbosity) if first_names_json else set()

    # Fetch last names
    last_names_json = fetch_json_from_url(last_names_url, verbosity)
    last_names = process_last_names(last_names_json, verbosity) if last_names_json else set()

    # Cache the results
    if first_names or last_names:
        try:
            cache_data = {
                'version': '1.0',
                'first_names': first_names,
                'last_names': last_names,
                'timestamp': datetime.now().isoformat()
            }

            # Write to temporary file first, then rename (atomic operation)
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f)
            temp_file.replace(cache_file)

            if verbosity > 0:
                print(f"Cached SCB name lists to {cache_file}", file=sys.stderr)
        except (OSError, pickle.PickleError) as e:
            if verbosity >= 0:
                print(f"Warning: Failed to cache name lists: {e}", file=sys.stderr)

    if verbosity >= 0:
        print(f"Fetched {len(first_names)} first names and {len(last_names)} last names", file=sys.stderr)

    return first_names, last_names


def collect_names_from_csv(
    csv_path: Path, 
    fornamn_field: str, 
    efternamn_field: str,
    check_fornamn: bool,
    check_efternamn: bool,
    verbosity: int
) -> Tuple[Set[str], Set[str]]:
    """Collect unique förnamn and efternamn from a CSV file.

    Returns:
        Tuple of (förnamn_set, efternamn_set)
    """
    fornamn_set = set()
    efternamn_set = set()

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Check if required fields exist
            if reader.fieldnames is None:
                print(f"Error: No header found in {csv_path}", file=sys.stderr)
                return fornamn_set, efternamn_set

            if check_fornamn and fornamn_field not in reader.fieldnames:
                print(f"Error: Field '{fornamn_field}' not found in {csv_path}. Available fields: {', '.join(reader.fieldnames)}", file=sys.stderr)
                return fornamn_set, efternamn_set

            if check_efternamn and efternamn_field not in reader.fieldnames:
                print(f"Error: Field '{efternamn_field}' not found in {csv_path}. Available fields: {', '.join(reader.fieldnames)}", file=sys.stderr)
                return fornamn_set, efternamn_set

            # Process each row
            for row in reader:
                if check_fornamn:
                    fornamn = row.get(fornamn_field, '').strip()
                    if fornamn:
                        fornamn_set.add(fornamn)

                if check_efternamn:
                    efternamn = row.get(efternamn_field, '').strip()
                    if efternamn:
                        efternamn_set.add(efternamn)

    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
    except PermissionError:
        print(f"Error: Permission denied: {csv_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}", file=sys.stderr)

    return fornamn_set, efternamn_set


def main():
    parser = argparse.ArgumentParser(
        description='Check CSV files for names not present in SCB Swedish name lists',
        epilog='The script uses the same cached SCB name lists as the OCR script.')

    parser.add_argument('csv_files', nargs='+', type=Path,
                        help='Input CSV file(s) to check')
    parser.add_argument('--fornamn-field', default='fornamn',
                        help='Name of the first name field in CSV (default: fornamn)')
    parser.add_argument('--efternamn-field', default='efternamn',
                        help='Name of the last name field in CSV (default: efternamn)')

    # Name type selection
    name_group = parser.add_mutually_exclusive_group()
    name_group.add_argument('-f', '--fornamn-only', action='store_true',
                           help='Check only förnamn')
    name_group.add_argument('-e', '--efternamn-only', action='store_true',
                           help='Check only efternamn')

    # Prefix control
    parser.add_argument('--no-prefix', action='store_true',
                        help='Do not show F:/E: prefixes (only allowed when checking one type)')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode (only output non-matching names)')
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Force refresh of SCB name lists from API')
    parser.add_argument('--show-fields', action='store_true',
                        help='Show all CSV fields and exit')

    args = parser.parse_args()

    # Determine what to check
    check_fornamn = not args.efternamn_only
    check_efternamn = not args.fornamn_only

    # Validate prefix option
    if args.no_prefix and check_fornamn and check_efternamn:
        print("Error: --no-prefix can only be used when checking either förnamn or efternamn, not both", file=sys.stderr)
        sys.exit(1)

    # Determine if we should show prefixes
    show_prefix = not args.no_prefix if (check_fornamn != check_efternamn) else True

    # Set verbosity
    if args.quiet:
        verbosity = -1
    elif args.verbose:
        verbosity = 1
    else:
        verbosity = 0

    # If show-fields, just display fields and exit
    if args.show_fields:
        for csv_path in args.csv_files:
            if not csv_path.exists():
                print(f"Error: File not found: {csv_path}", file=sys.stderr)
                continue

            print(f"\nFields in {csv_path}:")
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        for field in reader.fieldnames:
                            print(f"  - {field}")
                    else:
                        print("  No fields found (empty file?)")
            except Exception as e:
                print(f"  Error reading file: {e}")
        sys.exit(0)

    # Load SCB name lists
    first_names_scb, last_names_scb = load_or_fetch_name_lists(
        force_refresh=args.refresh_cache,
        verbosity=verbosity
    )

    if not first_names_scb and not last_names_scb:
        print("Error: Could not load SCB name lists", file=sys.stderr)
        sys.exit(1)

    # Collect all unique names from CSV files
    all_fornamn = set()
    all_efternamn = set()

    for csv_path in args.csv_files:
        if not csv_path.exists():
            print(f"Error: File not found: {csv_path}", file=sys.stderr)
            continue

        if verbosity >= 0:
            print(f"Processing {csv_path}...", file=sys.stderr)

        fornamn, efternamn = collect_names_from_csv(
            csv_path,
            args.fornamn_field,
            args.efternamn_field,
            check_fornamn,
            check_efternamn,
            verbosity
        )

        all_fornamn.update(fornamn)
        all_efternamn.update(efternamn)

    # Find names not in SCB lists
    missing_fornamn = all_fornamn - first_names_scb if check_fornamn else set()
    missing_efternamn = all_efternamn - last_names_scb if check_efternamn else set()

    # Prepare output list with prefixes
    output_names = []

    if check_fornamn:
        for name in missing_fornamn:
            if show_prefix:
                output_names.append(('F', name))
            else:
                output_names.append(('', name))

    if check_efternamn:
        for name in missing_efternamn:
            if show_prefix:
                output_names.append(('E', name))
            else:
                output_names.append(('', name))

    # Sort by name (not prefix)
    output_names.sort(key=lambda x: x[1])

    # Output results
    if verbosity >= 0:
        missing_count = len(missing_fornamn) + len(missing_efternamn)
        if missing_count > 0:
            parts = []
            if check_fornamn and missing_fornamn:
                parts.append(f"{len(missing_fornamn)} förnamn")
            if check_efternamn and missing_efternamn:
                parts.append(f"{len(missing_efternamn)} efternamn")
            print(f"Found {' and '.join(parts)} not in SCB lists", file=sys.stderr)
        else:
            print("All names found in SCB lists", file=sys.stderr)

    # Print the names
    for prefix, name in output_names:
        if prefix:
            print(f"{prefix}:{name}")
        else:
            print(name)


if __name__ == '__main__':
    main()

