#!/usr/bin/env python3
"""
OCR PDF and extract Swedish personal numbers with associated names.
"""

import sys
import os
import re
import subprocess
import tempfile
import argparse
from collections import defaultdict, Counter
from datetime import datetime
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading
import time
import unicodedata
import dataclasses
import unidecode
from typing import List, Dict, Any, Tuple, Optional
import json
import pickle
import hashlib
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

@dataclasses.dataclass
class Config:
    """Configuration parameters for OCR and extraction."""

    # OCR settings
    ocr_languages: str = "swe+eng"
    ocr_optimize_level: int = 0
    jpeg_quality: int = 100
    png_quality: int = 100

    # Personnummer patterns
    personnummer_pattern: str = r'[0-9]{6}\-([0-9]{4}|TF[0-9]{2})'
    use_pnr_patterns: bool = True  # NEW: Enable personnummer pattern hints for OCR
    pnr_patterns_path: str = None  # NEW: Path to temporary patterns file

    # Name lists from SCB (will be populated)
    first_names_set: set = dataclasses.field(default_factory=set)
    last_names_set: set = dataclasses.field(default_factory=set)
    use_scb_names: bool = True
    scb_names_dict_path: str = None  # Path to temporary dictionary file

    # Name processing
    allowed_two_letter_words: set = dataclasses.field(default_factory=lambda: {
        # Original Scandinavian
        'af', 'av', 'de',
        # Asian Surnames/Names - Chinese (Romanized)
        'li', 'lu', 'ma', 'wu', 'xu', 'yu', 'hu', 'he', 'qi', 'yi', 'fu', 'du', 'gu', 'su', 'ni', 'pu', 'xi',
        # Korean
        'yi', 'ko', 'no', 'so', 'ha',
        # Vietnamese  
        'le', 'ly', 'do', 'ho', 'vo', 'vu', 'to', 'an',
        # Name Particles/Prefixes - Dutch/Flemish
        'te', 'op',
        # French
        'du', 'le', 'la',
        # Spanish/Portuguese
        'da', 'do', 'el',
        # Italian
        'di', 'lo',
        # Arabic
        'al', 'el',
        # German
        'zu', 'am', 'im',
        # Irish/Gaelic
        'ni', 'ua'
    })

    # NEW: Allowed three-letter words at the end (common name endings)
    allowed_trailing_three_letter_words: set = dataclasses.field(default_factory=lambda: {
        'son',  # Most common Swedish surname ending (Andersson, Johansson, etc.)
        'sen',  # Danish/Norwegian variant (Hansen, Jensen, etc.)
        'zon',  # Less common variant
        'zen',  # Less common variant
        'dotter', # Though this is longer, including for completeness
        'nen',  # Finnish surname ending (Virtanen, etc.)
        'ven',  # Some names end with this
        'man',  # Common ending (Bergman, etc.)
        'din',  # Some names end with this
        'ius',  # Latin-origin names
        'ell',  # Names like Kjell
        'olf',  # Names like Rolf
    })

    # Scoring weights
    scandinavian_bonus: int = 1000
    all_caps_penalty: int = 50
    excessive_length_threshold: int = 15
    excessive_length_penalty: int = 100

    # Name list matching bonuses (UPDATED: Higher than Scandinavian bonus)
    first_name_match_bonus: int = 2000  # Increased from 500
    last_name_match_bonus: int = 2000   # Increased from 500

    # Filter keywords
    filter_keywords: list = dataclasses.field(default_factory=lambda: 
        ['personnr', 'personnummer', 'namn', 'name', 'number'])

    # Name cleaning
    unwanted_chars_pattern: str = r'[|\\/\[\]{}()<>@#$%^&*+=~`"_;:!?\n\r\t0-9]'

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

def load_or_fetch_name_lists(use_scb_names=True, force_refresh=False, verbosity=0):
    """Load name lists from cache or fetch from SCB API."""
    if not use_scb_names:  # RENAMED from use_json_names
        if verbosity > 0:
            print("  SCB name lists disabled by user", file=sys.stderr)
        return set(), set()

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
                    print(f"  Loaded SCB name lists from cache: {len(first_names)} first names, {len(last_names)} last names", 
                          file=sys.stderr)

                return first_names, last_names
        except (pickle.PickleError, KeyError, EOFError) as e:
            if verbosity >= 0:
                print(f"  Warning: Cache file corrupted, will fetch fresh data: {e}", file=sys.stderr)

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
                print(f"  Cached SCB name lists to {cache_file}", file=sys.stderr)
        except (OSError, pickle.PickleError) as e:
            if verbosity >= 0:
                print(f"  Warning: Failed to cache name lists: {e}", file=sys.stderr)

    if verbosity >= 0:
        print(f"  Fetched {len(first_names)} first names and {len(last_names)} last names", file=sys.stderr)

    return first_names, last_names

def create_scb_names_dictionary(first_names_set, last_names_set, verbosity=0):
    """Create a temporary dictionary file with all SCB names and common Swedish words for ocrmypdf.

    Returns:
        Path to the temporary dictionary file, or None if creation fails
    """
    if not first_names_set and not last_names_set:
        if verbosity > 0:
            print("  No SCB names available for dictionary", file=sys.stderr)
        return None

    try:
        # Create a temporary file for the dictionary
        import tempfile
        fd, dict_path = tempfile.mkstemp(suffix='.txt', prefix='scb_names_dict_')

        # Use names as-is from SCB (already properly capitalized)
        # Only deduplicate if there's overlap between first/last names
        all_names = first_names_set | last_names_set

        # Write names to file (one word per line)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for name in sorted(all_names):
                # Only include valid names (no spaces, reasonable length)
                if name and ' ' not in name and len(name) <= 30:
                    f.write(name + '\n')

            # Add common Swedish document words
            swedish_words = [
                'Personnr',
                'Personnummer',
                'Namn',
                'Efternamn',
                'Förnamn',
                'Födelsedatum',
                'Adress',
                'Telefon',
                'Datum',
                'Underskrift'
            ]

            for word in swedish_words:
                f.write(word + '\n')

        if verbosity > 0:
            total_names = len(all_names) + len(swedish_words)
            print(f"  Created SCB names dictionary with {total_names:,} entries "
                  f"({len(first_names_set):,} first + {len(last_names_set):,} last + {len(swedish_words)} document words) for OCR enhancement", 
                  file=sys.stderr)

        return dict_path

    except (OSError, IOError) as e:
        if verbosity >= 0:
            print(f"  Warning: Failed to create SCB names dictionary: {e}", file=sys.stderr)
        return None

def count_words_for_cleaning(text):
    """
    Count words for cleaning logic where hyphens are considered word characters.
    "Anna-Maria" = 1 word, "Anna Maria" = 2 words
    """
    if not text:
        return 0

    # Replace hyphens with a placeholder character that won't be split
    text_modified = text.replace('-', '_')

    # Split on spaces and count
    words = text_modified.split()
    return len(words)

def create_personnummer_patterns_file(verbosity=0):
    """Create a temporary patterns file to help Tesseract recognize personnummer formats.

    Returns:
        Path to the temporary patterns file, or None if creation fails
    """
    try:
        # Create a temporary file for the patterns
        import tempfile
        fd, patterns_path = tempfile.mkstemp(suffix='.txt', prefix='pnr_patterns_')

        # Define Tesseract patterns for personnummer
        # Format: YYMMDD-XXXX or YYMMDD-TFXX
        patterns = [
            # Standard format: 6 digits, hyphen, 4 digits
            r'\d\d\d\d\d\d-\d\d\d\d',
            # TF format: 6 digits, hyphen, TF, 2 digits  
            r'\d\d\d\d\d\d-TF\d\d',
            # Also include without hyphen (sometimes OCR misses it)
            r'\d\d\d\d\d\d\d\d\d\d',
            # With space instead of hyphen (common OCR error)
            r'\d\d\d\d\d\d \d\d\d\d',
            # variants
            r'\d\d \d\d \d\d  - \d\d\d\d',
            r'\d\d\d\d\d\d  - \d\d\d\d',
            # Partial patterns to help with segments
            r'\d\d\d\d\d\d',  # Birth date part
            r'-\d\d\d\d',     # Last 4 digits with hyphen
            r'-TF\d\d',       # TF suffix with hyphen
            r'- \d\d \d\d',
            r'- TF \d\d',
            # variants
            r'\d\d\d\d',
            r'TF\d\d',
            r'\d\d \d\d',
            r'TF \d\d',
        ]

        # Write patterns to file (one per line)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for pattern in patterns:
                f.write(pattern + '\n')

        if verbosity > 1:
            print(f"  Created personnummer patterns file with {len(patterns)} patterns for OCR", 
                  file=sys.stderr)

        return patterns_path

    except (OSError, IOError) as e:
        if verbosity >= 0:
            print(f"  Warning: Failed to create personnummer patterns file: {e}", file=sys.stderr)
        return None


def check_and_swap_names(efternamn, fornamn, first_names_set, last_names_set, verbosity=0):
    """Check if names should be swapped based on name list membership and 'son' ending.

    Will NOT swap if both names appear in both lists UNLESS efternamn ends with 'son'.
    """
    if not efternamn or not fornamn:
        return efternamn, fornamn, False

    # Check where each name appears
    fornamn_in_first = fornamn in first_names_set
    fornamn_in_last = fornamn in last_names_set
    efternamn_in_first = efternamn in first_names_set
    efternamn_in_last = efternamn in last_names_set

    # NEW: Check for "son" ending bias
    fornamn_ends_with_son = fornamn.lower().endswith('son')

    # If fornamn ends with "son" and both names are valid in both lists,
    # we should swap because "son" is a strong indicator of a last name
    if fornamn_ends_with_son and (fornamn_in_first and fornamn_in_last and 
                                   efternamn_in_first and efternamn_in_last):
        if verbosity > 1:
            print(f"    SWAPPING: '{efternamn}, {fornamn}' → '{fornamn}, {efternamn}' "
                  f"(fornamn ends with 'son' - strong last name indicator)", file=sys.stderr)
        return fornamn, efternamn, True

    # Original check: if BOTH names appear in BOTH lists, don't swap
    # (unless overridden by 'son' ending above)
    if (fornamn_in_first and fornamn_in_last and 
        efternamn_in_first and efternamn_in_last):
        if verbosity > 1:
            print(f"    NO SWAP: '{efternamn}, {fornamn}' - both names appear in both lists "
                  f"(both arrangements valid)", file=sys.stderr)
        return efternamn, fornamn, False

    # Original swap condition:
    # fornamn exists in last_names_set AND efternamn exists in first_names_set
    if fornamn_in_last and efternamn_in_first:
        if verbosity > 1:
            print(f"    SWAPPING: '{efternamn}, {fornamn}' → '{fornamn}, {efternamn}' "
                  f"(fornamn found in last names, efternamn found in first names)", file=sys.stderr)
        return fornamn, efternamn, True

    return efternamn, fornamn, False


# Set OMP_THREAD_LIMIT globally if not already set
if 'OMP_THREAD_LIMIT' not in os.environ:
    os.environ['OMP_THREAD_LIMIT'] = '1'
    print("Setting OMP_THREAD_LIMIT=1 for optimal parallel OCR performance", file=sys.stderr)

class ProgressTracker:
    """Thread-safe progress tracker for parallel OCR jobs."""
    def __init__(self, total, verbosity):
        self.total = total
        self.completed = 0
        self.lock = threading.Lock()
        self.verbosity = verbosity
        self.start_time = time.time()

    def update(self, page_num):
        with self.lock:
            self.completed += 1
            if self.verbosity >= 0:
                elapsed = time.time() - self.start_time
                rate = self.completed / elapsed if elapsed > 0 else 0
                remaining = (self.total - self.completed) / rate if rate > 0 else 0

                # Clear line and print progress
                print(f"\r  OCR Progress: {self.completed}/{self.total} pages "
                      f"({self.completed*100//self.total}%) - "
                      f"~{remaining:.0f}s remaining    ", 
                      end='', file=sys.stderr)

                if self.completed == self.total:
                    print(f"\n  OCR completed in {elapsed:.1f} seconds", file=sys.stderr)

def get_optimal_workers(total_pages, verbosity):
    """Determine optimal number of worker processes for OCR."""
    # Get CPU count
    cpu_count = multiprocessing.cpu_count()

    # Check if OMP_THREAD_LIMIT is set
    omp_threads = os.environ.get('OMP_THREAD_LIMIT', '1')

    try:
        threads_per_process = int(omp_threads)
    except ValueError:
        threads_per_process = 1

    if threads_per_process == 1:
        # Each OCR process uses only 1 thread, so we can use more workers
        # Leave some cores for system overhead
        optimal = max(1, min(cpu_count - 2, 16))

        # For very high core counts, be slightly conservative
        if cpu_count > 16:
            optimal = max(1, int(cpu_count * 0.75))
    else:
        # OCR processes might use multiple threads
        # Be more conservative with worker count
        optimal = max(1, cpu_count // (threads_per_process * 2))

    # Don't use more workers than pages
    optimal = min(optimal, total_pages)

    # Minimum of 1 worker
    optimal = max(1, optimal)

    if verbosity > 0:
        print(f"  System has {cpu_count} CPU cores", file=sys.stderr)
        print(f"  OMP_THREAD_LIMIT={omp_threads} (threads per OCR process)", file=sys.stderr)
        print(f"  Using {optimal} parallel OCR workers", file=sys.stderr)

        if threads_per_process > 1:
            print(f"  Tip: Set OMP_THREAD_LIMIT=1 for better parallelization ({cpu_count - 2} workers instead of {optimal})", 
                  file=sys.stderr)

    return optimal

def normalize_to_base(text):
    """Remove diacritics and convert to base characters for comparison.

    Examples:
    - "Bergström" -> "Bergstrom"
    - "Ågren" -> "Agren"
    - "Müller" -> "Muller"
    """
    # Normalize to NFD (decomposed form) then filter out combining marks
    nfd = unicodedata.normalize('NFD', text)
    base = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    return base.lower()

def has_scandinavian_chars(text):
    """Check if text contains Scandinavian special characters."""
    scandinavian = set('åäöÅÄÖ')
    return any(char in scandinavian for char in text)

def count_scandinavian_chars(text):
    """Count the number of Scandinavian special characters in text."""
    scandinavian = set('åäöÅÄÖ')
    return sum(1 for char in text if char in scandinavian)

def is_all_caps(text):
    """Check if text is all uppercase (ignoring non-letter characters)."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 0 and all(c.isupper() for c in letters)

def has_problematic_two_letter_words(efternamn, fornamn):
    """Check if names contain two-letter words that are not af, av, de (case insensitive)."""
    allowed_two_letter = {'af', 'av', 'de'}

    # Check all words in both names
    all_words = (efternamn + ' ' + fornamn).split()

    for word in all_words:
        if len(word) == 2 and word.lower() not in allowed_two_letter:
            return True

    return False

def count_digit_differences(pnr1, pnr2):
    """Count the number of differing digits between two personnummer.

    Returns:
        (count, position) where count is the number of differences
        and position is the index of the difference (if exactly one difference)
    """
    # Remove hyphens for comparison
    clean1 = pnr1.replace('-', '')
    clean2 = pnr2.replace('-', '')

    # Must be same length
    if len(clean1) != len(clean2):
        return (999, -1)  # Different lengths, not comparable

    differences = []
    for i, (c1, c2) in enumerate(zip(clean1, clean2)):
        if c1 != c2:
            differences.append(i)

    if len(differences) == 1:
        return (1, differences[0])
    else:
        return (len(differences), -1)


def merge_short_words_with_nearest_shortest(words, config, verbosity=0):
    """Merge single letter words and problematic two-letter words with nearest shortest neighbor."""
    if len(words) <= 1:
        return words

    words_copy = words[:]
    original_words = words[:]  # For logging

    # Keep track of words to merge (index, reason)
    to_merge = []

    for i, word in enumerate(words_copy):
        if len(word) == 1:
            to_merge.append((i, f"single letter '{word}'"))
        elif len(word) == 2:
            # Clean the word for comparison (remove apostrophes and other punctuation for the check)
            # But we need to be careful - O' should not be considered as just 'O'
            clean_word_for_check = word.lower().replace("'", "").replace("-", "").replace(".", "")
            if clean_word_for_check not in config.allowed_two_letter_words and len(clean_word_for_check) == 2:
                to_merge.append((i, f"two letter '{word}' (not in allowed list)"))

    if not to_merge:
        return words_copy

    # Process merges (work backwards to preserve indices)
    for i, reason in reversed(to_merge):
        if i >= len(words_copy):  # Skip if already merged
            continue

        word_to_merge = words_copy[i]

        # Find the immediate neighbors (before and after)
        candidates = []

        # Word before (if exists and not a short word)
        if i > 0 and len(words_copy[i-1]) > 2:
            candidates.append((i-1, words_copy[i-1], len(words_copy[i-1])))

        # Word after (if exists and not a short word) 
        if i < len(words_copy)-1 and len(words_copy[i+1]) > 2:
            candidates.append((i+1, words_copy[i+1], len(words_copy[i+1])))

        if not candidates:
            # No valid neighbors to merge with, skip
            if verbosity > 1:
                print(f"    Word merging: {reason} - no valid neighbors to merge with", file=sys.stderr)
            continue

        # Choose the shortest candidate (if tie, prefer the earlier one)
        best_target_idx, best_target_word, best_length = min(candidates, key=lambda x: (x[2], x[0]))

        # Merge: if target is before merge word, append; otherwise prepend
        if best_target_idx < i:
            merged = best_target_word + word_to_merge
        else:
            merged = word_to_merge + best_target_word

        if verbosity > 1:
            position = "before" if best_target_idx < i else "after" 
            other_candidates = [f"'{c[1]}' (len {c[2]})" for c in candidates if c[0] != best_target_idx]
            other_info = f" vs {', '.join(other_candidates)}" if other_candidates else ""
            print(f"    Word merging: {reason} merged with shortest neighbor '{best_target_word}' (len {best_length}, {position}){other_info} → '{merged}'", 
                  file=sys.stderr)

        # Replace target word with merged word and remove merge word
        words_copy[best_target_idx] = merged
        words_copy.pop(i)

    # Log overall change if any merges happened
    if verbosity > 1 and to_merge:
        print(f"    Word merging result: {' '.join(original_words)} → {' '.join(words_copy)}", 
              file=sys.stderr)

    return words_copy

def select_best_name_variant(name_counter, pnr, config, verbosity):
    """Select the best name variant with bias towards SCB list matches over Scandinavian characters."""
    if len(name_counter) == 1:
        return list(name_counter.keys())[0]

    # Group variants by their base form (without diacritics)
    base_groups = defaultdict(list)
    for name, count in name_counter.items():
        base_efter = normalize_to_base(name[0])
        base_for = normalize_to_base(name[1])
        base_key = (base_efter, base_for)
        base_groups[base_key].append((name, count))

    # Score all variants
    all_variants = []
    for group_variants in base_groups.values():
        all_variants.extend(group_variants)

    scored_variants = []
    for name, count in all_variants:
        efternamn, fornamn = name
        full_name = efternamn + fornamn

        # Base score from occurrence count
        score = count

        # SCB name list matching bonuses - HIGHER priority than Scandinavian
        if config.use_scb_names:
            if fornamn in config.first_names_set:
                score += config.first_name_match_bonus  # Now 2000
                if verbosity > 1:
                    print(f"    SCB name bonus: '{fornamn}' found in first names (+{config.first_name_match_bonus})", 
                          file=sys.stderr)

            if efternamn in config.last_names_set:
                score += config.last_name_match_bonus  # Now 2000
                if verbosity > 1:
                    print(f"    SCB name bonus: '{efternamn}' found in last names (+{config.last_name_match_bonus})", 
                          file=sys.stderr)

        # Scandinavian character bonus (now lower priority than SCB lists)
        scand_count = count_scandinavian_chars(full_name)
        score += scand_count * config.scandinavian_bonus  # 1000 per character

        # ALL CAPS penalty
        if is_all_caps(efternamn) or is_all_caps(fornamn):
            score -= config.all_caps_penalty

        # Excessive length penalty
        if len(efternamn) > config.excessive_length_threshold:
            score -= config.excessive_length_penalty
        if len(fornamn) > config.excessive_length_threshold:
            score -= config.excessive_length_penalty

        scored_variants.append((name, count, score))

    # Sort by score (highest first)
    scored_variants.sort(key=lambda x: x[2], reverse=True)
    selected = scored_variants[0][0]

    if verbosity > 0 and len(scored_variants) > 1:
        print(f"  {pnr}: Selected from {len(scored_variants)} variants:", file=sys.stderr)
        for name, count, score in scored_variants[:3]:  # Show top 3
            marker = " ← SELECTED" if name == selected else ""
            print(f"    {name[0]}, {name[1]} (occurs {count}x, score={score}){marker}", 
                  file=sys.stderr)

    return selected

def clean_name(name, config):
    """Clean name by removing unusual characters but keeping international characters."""
    if not name:
        return ""

    # Remove unwanted characters using config pattern
    cleaned = re.sub(config.unwanted_chars_pattern, '', name)

    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Remove spaces around hyphens and apostrophes
    cleaned = re.sub(r'\s*-\s*', '-', cleaned)
    cleaned = re.sub(r"\s*'\s*", "'", cleaned)

    # Strip leading and trailing whitespace
    cleaned = cleaned.strip()

    # Strip leading and trailing punctuation (except legitimate name enders)
    # Keep trailing period (for Jr. Sr. etc)
    cleaned = re.sub(r'^[^\w\s]+', '', cleaned)  # Remove leading punctuation
    cleaned = re.sub(r'[^\w\s.]+$', '', cleaned)  # Remove trailing punct except period

    return cleaned

def validate_personnummer_date(pnr, verbosity=0):
    """Validate the date portion (YYMMDD) of a Swedish personnummer.

    Early rejection: ONLY for months 20-99 (likely organizational numbers)
    Late rejection: All other invalid dates (allows autocorrection)
    TF numbers: Must have valid month (01-12) and day (01-31)

    Args:
        pnr: personnummer string (with or without hyphen)
        verbosity: verbosity level for detailed logging

    Returns:
        tuple: (is_valid, rejection_reason, is_samordningsnummer, is_early_rejection)
    """
    # Remove hyphen and ensure we have at least 6 digits
    clean_pnr = pnr.replace('-', '')
    if len(clean_pnr) < 6:
        return (False, "too short (less than 6 digits)", False, False)

    try:
        # Extract date components - YYMMDD (always from the beginning)
        year_str = clean_pnr[:2]   # YY (00-99, always valid)
        month_str = clean_pnr[2:4] # MM 
        day_str = clean_pnr[4:6]   # DD

        year = int(year_str)   # 00-99, always valid for format
        month = int(month_str) 
        day = int(day_str)     

        # Check if this is a TF number (has "TF" after the hyphen)
        is_tf = 'TF' in pnr

        # EARLY REJECTION: Only for months 20-99 (organizational numbers) - NOT for TF numbers
        if not is_tf and month >= 20:
            return (False, f"invalid month {month_str} (>= 20 - organizational number?)", False, True)

        # Determine if this is a samordningsnummer (days 61-91) - only for non-TF
        is_samordningsnummer = (61 <= day <= 91) and not is_tf

        # LATE REJECTIONS (allow name extraction and autocorrection):

        # Validate month (00, 13-19 are invalid but could be autocorrected)
        if month < 1:
            reason = f"invalid month {month_str} for TF number (< 01)" if is_tf else f"invalid month {month_str} (< 01 - could be OCR error)"
            return (False, reason, False, False)
        elif month > 12:
            # Months 13-19 could potentially be autocorrected
            if verbosity > 1 and 13 <= month <= 19:
                print(f"    Month {month_str} (13-19 range) - could be autocorrected later", file=sys.stderr)
            reason = f"invalid month {month_str} for TF number (> 12)" if is_tf else f"invalid month {month_str} (13-19 range - possible OCR error)"
            return (False, reason, False, False)

        # Validate day
        if is_tf:
            # TF numbers: must have normal days (01-31), NOT samordningsnummer range
            if day < 1 or day > 31:
                return (False, f"invalid day {day_str} for TF number (must be 01-31)", False, False)

            # Additional validation: check if day makes sense for the month
            if month == 2 and day > 29:  # February
                return (False, f"invalid day {day} for February in TF number", False, False)
            elif month in [4, 6, 9, 11] and day > 30:  # April, June, September, November
                month_names = {4: "April", 6: "June", 9: "September", 11: "November"}
                return (False, f"invalid day {day} for {month_names[month]} in TF number", False, False)
        else:
            # For non-TF numbers, NO EARLY REJECTION for any day value (00-99)
            # Valid: 01-31 (normal) or 61-91 (samordningsnummer)
            # Invalid but correctable: 00, 32-60, 92-99

            if day == 0:
                return (False, f"invalid day {day_str} (00 - could be OCR error)", False, False)
            elif 32 <= day <= 60:
                return (False, f"invalid day {day_str} (32-60 range - could be OCR error)", False, False)
            elif day > 91:
                return (False, f"invalid day {day_str} (> 91 - could be OCR error)", False, False)

            # Additional validation: check if day makes sense for the month (for valid day ranges)
            if (1 <= day <= 31) or (61 <= day <= 91):
                actual_day = day if day <= 31 else day - 60  # Samordningsnummer: subtract 60

                if month == 2 and actual_day > 29:  # February
                    return (False, f"invalid day {actual_day} for February", is_samordningsnummer, False)
                elif month in [4, 6, 9, 11] and actual_day > 30:  # April, June, September, November
                    month_names = {4: "April", 6: "June", 9: "September", 11: "November"}
                    return (False, f"invalid day {actual_day} for {month_names[month]}", is_samordningsnummer, False)

        # If all checks pass, it's valid
        if is_samordningsnummer and verbosity > 1:
            print(f"    Valid samordningsnummer date: {year_str}-{month_str}-{day_str} (actual day: {day - 60})", 
                  file=sys.stderr)

        return (True, None, is_samordningsnummer, False)

    except ValueError:
        # Non-numeric characters in date portion
        return (False, "non-numeric characters in date portion", False, False)

def luhn_check(number_str):
    """Validate Swedish personnummer using Luhn algorithm."""
    if len(number_str) != 10:
        return False

    try:
        digits = [int(d) for d in number_str]
        checksum = 0
        for i, digit in enumerate(digits[:-1]):
            if i % 2 == 0:
                doubled = digit * 2
                checksum += doubled if doubled < 10 else doubled - 9
            else:
                checksum += digit
        return (10 - (checksum % 10)) % 10 == digits[-1]
    except ValueError:
        return False

def format_personnummer(pnr):
    """Format personnummer with century prefix."""
    if not pnr:  # Handle empty personnummer
        return ""

    # Remove hyphen
    pnr_clean = pnr.replace('-', '')

    # Extract year part (first 2 digits)
    year_part = int(pnr_clean[:2])
    current_year = datetime.now().year

    # Determine century based on 95-year rule
    # This applies to both regular and TF numbers
    if year_part <= (current_year % 100):
        century_year = 2000 + year_part
    else:
        century_year = 1900 + year_part

    # Check if age would be > 95
    age = current_year - century_year
    if age > 95:
        century_year += 100

    return str(century_year)[:2] + pnr_clean

def clean_name_candidate(text, config, verbosity=0):
    """Clean name candidate with new iterative cleaning logic.

    Rules:
    - Preprocess to remove "Eleven" and "intygas att" patterns from the left
    - Only clean if > 2 words (hyphen counts as word character)
    - One/two-letter words: can clean from both leading and trailing
    - Three-letter words: can clean from trailing only
    - Stop cleaning from an end when unclearable word is found
    - Never clean SCB names or allowed words
    """
    if not text or not text.strip():
        return ""

    original_text = text
    text = text.strip()

    # === NEW PREPROCESSING STEP ===
    # Remove "Eleven" and "intygas att" patterns from the left side

    # Pattern 1: "Eleven" - case insensitive, with flexible whitespace
    eleven_pattern = re.compile(r'^.*?\bEleven\b\s*', re.IGNORECASE)
    match = eleven_pattern.search(text)
    if match:
        text = text[match.end():].strip()
        if verbosity > 1:
            print(f"    Preprocessing: Removed 'Eleven' pattern from '{original_text}' → '{text}'", file=sys.stderr)

    # Pattern 2: "intygas att" - case insensitive, with flexible whitespace between words
    intygas_pattern = re.compile(r'^.*?\bintygas\s+att\b\s*', re.IGNORECASE)
    match = intygas_pattern.search(text)
    if match:
        text = text[match.end():].strip()
        if verbosity > 1:
            print(f"    Preprocessing: Removed 'intygas att' pattern from '{original_text}' → '{text}'", file=sys.stderr)

    # If preprocessing removed everything, return empty
    if not text:
        if verbosity > 1:
            print(f"    Preprocessing removed entire string", file=sys.stderr)
        return ""

    text_after_preprocessing = text
    # === END OF NEW PREPROCESSING STEP ===

    # Get SCB names for checking
    scb_names = set()
    if config.use_scb_names:
        scb_names = set(n.lower() for n in config.first_names_set) | set(n.lower() for n in config.last_names_set)

    # Allowed words that should NOT be cleaned
    allowed_one_letter_words = {}

    def can_clean_word(word, position):
        """Check if a word can be cleaned based on all rules."""
        word_lower = word.lower()
        word_len = len(word)

        # Never clean words 4+ letters
        if word_len > 3:
            return False

        # Never clean SCB names
        if word_lower in scb_names:
            return False

        # Never clean allowed one-letter words
        if word_len == 1 and word_lower in allowed_one_letter_words:
            return False

        # Never clean allowed two-letter words
        if word_len == 2 and word_lower in config.allowed_two_letter_words:
            return False

        # Three-letter words can only be cleaned from trailing position
        if word_len == 3:
            if position == 'leading':
                return False
            # Check if it's in allowed trailing three-letter words
            if word_lower in config.allowed_trailing_three_letter_words:
                return False

        return True

    # Iterative cleaning process
    trailing_blocked = False
    leading_blocked = False

    while True:
        # Check word count with hyphen as word character
        if count_words_for_cleaning(text) <= 2:
            if verbosity > 1:
                print(f"    Stopping cleaning: only {count_words_for_cleaning(text)} words remaining", file=sys.stderr)
            break

        # If both ends are blocked, stop
        if trailing_blocked and leading_blocked:
            if verbosity > 1:
                print(f"    Stopping cleaning: both ends blocked", file=sys.stderr)
            break

        cleaned_this_iteration = False

        # Try cleaning trailing word first
        if not trailing_blocked:
            words = text.split()
            if words:
                last_word = words[-1]
                if can_clean_word(last_word, 'trailing'):
                    text = ' '.join(words[:-1])
                    cleaned_this_iteration = True
                    if verbosity > 1:
                        print(f"    Cleaned trailing '{last_word}' → '{text}'", file=sys.stderr)
                else:
                    trailing_blocked = True
                    if verbosity > 1:
                        print(f"    Cannot clean trailing '{last_word}' - blocking trailing", file=sys.stderr)

        # Check word count again
        if count_words_for_cleaning(text) <= 2:
            break

        # Try cleaning leading word
        if not leading_blocked:
            words = text.split()
            if words:
                first_word = words[0]
                if can_clean_word(first_word, 'leading'):
                    text = ' '.join(words[1:])
                    cleaned_this_iteration = True
                    if verbosity > 1:
                        print(f"    Cleaned leading '{first_word}' → '{text}'", file=sys.stderr)
                else:
                    leading_blocked = True
                    if verbosity > 1:
                        print(f"    Cannot clean leading '{first_word}' - blocking leading", file=sys.stderr)

        # If nothing was cleaned in this iteration, stop
        if not cleaned_this_iteration:
            break

    if verbosity > 1 and text != text_after_preprocessing:
        print(f"    Final cleaning result: '{text_after_preprocessing}' → '{text}'", file=sys.stderr)

    return text.strip()


def personnummer_sort_key(pnr):
    """Generate sort key for personnummer using 95-year rule.

    Returns a tuple that sorts correctly:
    - Century (20xx before 19xx for same YY)
    - Full personnummer for final ordering
    """
    if not pnr:
        return (9999, "")  # Empty sorts last

    # Remove hyphen for processing
    clean_pnr = pnr.replace('-', '')

    # Extract year (first 2 digits)
    try:
        year_part = int(clean_pnr[:2])
    except (ValueError, IndexError):
        return (9999, pnr)  # Invalid format sorts last

    current_year = datetime.now().year
    current_century_year = current_year % 100

    # Apply 95-year rule
    if year_part <= current_century_year:
        # Could be 2000s
        century_year = 2000 + year_part
    else:
        # Must be 1900s
        century_year = 1900 + year_part

    # Check if age would be > 95
    age = current_year - century_year
    if age > 95:
        century_year += 100

    # Return sort key: (century_year, full_pnr_as_string)
    return (century_year, clean_pnr)


def parse_name_from_text(text, config, verbosity=0):
    """Parse efternamn and förnamn from text, handling comma cleaning and splitting.

    Process:
    1. Apply word cleaning
    2. Handle commas (keep rightmost if multiple)
    3. Split on comma or space
    """
    if not text or not text.strip():
        return None, None

    # Apply the new cleaning FIRST
    text = clean_name_candidate(text, config, verbosity)

    if not text:
        return None, None

    # Remove leading/trailing commas
    text = text.strip(',').strip()

    # Handle multiple internal commas - keep only the rightmost one
    comma_count = text.count(',')
    if comma_count > 1:
        if verbosity > 1:
            print(f"    Found {comma_count} commas, keeping only rightmost", file=sys.stderr)
        # Find the rightmost comma and remove all others
        parts = text.split(',')
        # Keep only the last comma by joining all but last with space, then add comma before last
        text = ' '.join(parts[:-1]) + ',' + parts[-1]
        if verbosity > 1:
            print(f"    After comma cleaning: '{text}'", file=sys.stderr)

    # Check if comma-separated format
    if ',' in text:
        parts = re.split(r',\s*', text)
        if len(parts) >= 2:
            efternamn_raw = clean_name(parts[0], config)
            fornamn_raw = clean_name(' '.join(parts[1:]), config)

            # Apply short word merging separately to each part
            efternamn_words = efternamn_raw.split() if efternamn_raw else []
            fornamn_words = fornamn_raw.split() if fornamn_raw else []

            if verbosity > 1 and (efternamn_words or fornamn_words):
                print(f"    Word merging phase (after cleaning):", file=sys.stderr)

            # Process each part separately
            processed_efternamn_words = merge_short_words_with_nearest_shortest(efternamn_words, config, verbosity)
            processed_fornamn_words = merge_short_words_with_nearest_shortest(fornamn_words, config, verbosity)

            efternamn = ' '.join(processed_efternamn_words) if processed_efternamn_words else ""
            fornamn = ' '.join(processed_fornamn_words) if processed_fornamn_words else ""

            # Log if cleaning or merging changed the names (in very verbose mode)
            if verbosity > 1:
                original_efter = parts[0].strip()
                original_for = ' '.join(parts[1:]).strip()

                # Check if anything changed
                efter_changed = (original_efter != efternamn_raw) or (efternamn_raw != efternamn)
                for_changed = (original_for != fornamn_raw) or (fornamn_raw != fornamn)

                if efter_changed or for_changed:
                    changes = []
                    if original_efter != efternamn_raw:
                        changes.append(f"efternamn cleaned: '{original_efter}' → '{efternamn_raw}'")
                    if efternamn_raw != efternamn:
                        changes.append(f"efternamn merged: '{efternamn_raw}' → '{efternamn}'")
                    if original_for != fornamn_raw:
                        changes.append(f"förnamn cleaned: '{original_for}' → '{fornamn_raw}'")
                    if fornamn_raw != fornamn:
                        changes.append(f"förnamn merged: '{fornamn_raw}' → '{fornamn}'")

                    print(f"    Name processing (comma-separated): {'; '.join(changes)}", file=sys.stderr)

            return efternamn, fornamn
        return None, None

    else:
        # Space-separated format - use heuristic splitting
        cleaned_text = clean_name(text, config)

        if not cleaned_text:
            return None, None

        words = cleaned_text.split()
        if len(words) >= 2:
            if verbosity > 1:
                print(f"    Word merging phase (after cleaning):", file=sys.stderr)

            # Process short words by merging them with nearest shortest words
            processed_words = merge_short_words_with_nearest_shortest(words, config, verbosity)

            if len(processed_words) >= 2:
                # Heuristic: typically "Förnamn Efternamn" in Swedish
                # But check against SCB lists if available
                if config.use_scb_names and (config.first_names_set or config.last_names_set):
                    # Check if last word is a known surname
                    if processed_words[-1] in config.last_names_set:
                        efternamn = processed_words[-1]
                        fornamn = ' '.join(processed_words[:-1])
                    # Check if first word is a known first name
                    elif processed_words[0] in config.first_names_set:
                        fornamn = processed_words[0]
                        efternamn = ' '.join(processed_words[1:])
                    else:
                        # Default: assume last word is surname
                        efternamn = processed_words[-1]
                        fornamn = ' '.join(processed_words[:-1])
                else:
                    # No SCB lists, use default heuristic
                    efternamn = processed_words[-1]
                    fornamn = ' '.join(processed_words[:-1])

                if verbosity > 1:
                    if processed_words != words:
                        print(f"    Name processing (space-separated): '{text}' → cleaned: '{cleaned_text}' → processed: '{' '.join(processed_words)}' → '{efternamn}, {fornamn}'", 
                              file=sys.stderr)
                    else:
                        print(f"    Name processing (space-separated): '{text}' → cleaned: '{cleaned_text}' → '{efternamn}, {fornamn}'", 
                              file=sys.stderr)

                return efternamn, fornamn
            elif len(processed_words) == 1:
                # After processing, only one word left - can't split into efternamn/fornamn
                if verbosity > 1:
                    print(f"    Name processing (space-separated): '{text}' → only one word after processing: '{processed_words[0]}'", 
                          file=sys.stderr)
                return None, None
        elif len(words) == 1:
            # Only one word - check if it's a known name
            word = words[0]
            if config.use_scb_names:
                if word in config.first_names_set:
                    return "", word
                elif word in config.last_names_set:
                    return word, ""
            return None, None
        return None, None

def _process_inheritance_with_hole_filling(
    current_page_entries: Dict[int, List[Dict]], 
    total_pages: int, 
    inherit: bool,
    disable_hole_filling: bool,
    verbosity: int
) -> Tuple[List[Dict], Dict[str, List], int, int]:
    """Process inheritance logic with hole-filling for identical surrounding personnummer.

    New feature: If pages without personnummer are surrounded by pages with identical 
    personnummer, fill the hole with that personnummer.
    """
    final_data = []
    pnr_to_names = defaultdict(list)
    last_valid_entry = None
    pages_with_pnr = 0
    pages_without_pnr = 0

    # First pass: collect all page data and identify holes
    page_data = {}
    for page_num in range(1, total_pages + 1):
        if page_num in current_page_entries:
            page_entries = current_page_entries[page_num]
            if len(page_entries) != 1:
                raise RuntimeError(f"Internal error: Expected exactly 1 entry for page {page_num}, got {len(page_entries)}")

            entry = page_entries[0]
            page_data[page_num] = entry

            if entry['personnummer']:
                pages_with_pnr += 1
            else:
                pages_without_pnr += 1
        else:
            page_data[page_num] = None
            pages_without_pnr += 1

    # Second pass: apply inheritance and hole-filling
    for page_num in range(1, total_pages + 1):
        entry = page_data.get(page_num)

        if entry and entry['personnummer']:
            # Page has personnummer - use it
            pnr = entry['personnummer']

            # Track name combinations
            name_combo = (entry['efternamn'], entry['fornamn'])
            if name_combo != ("", ""):
                pnr_to_names[pnr].append(name_combo)

            # Add to final data
            final_data.append({
                'page': page_num,
                'personnummer': pnr,
                'efternamn': entry['efternamn'],
                'fornamn': entry['fornamn'],
                'luhn_valid': entry['luhn_valid'],
                'was_corrected': entry.get('was_corrected', False)
            })

            # Update last valid entry for regular inheritance
            last_valid_entry = {
                'personnummer': pnr,
                'efternamn': entry['efternamn'],
                'fornamn': entry['fornamn']
            }
        else:
            # No personnummer on this page - check for hole-filling opportunity
            filled = False

            if not disable_hole_filling:
                # Look for identical personnummer before and after this hole
                # Find the start and end of the current hole
                hole_start = page_num
                hole_end = page_num

                # Find consecutive pages without personnummer
                while hole_end < total_pages:
                    next_entry = page_data.get(hole_end + 1)
                    if next_entry and next_entry['personnummer']:
                        break
                    hole_end += 1

                # Find personnummer before the hole
                pnr_before = None
                for p in range(hole_start - 1, 0, -1):
                    prev_entry = page_data.get(p)
                    if prev_entry and prev_entry['personnummer']:
                        pnr_before = prev_entry['personnummer']
                        break

                # Find personnummer after the hole
                pnr_after = None
                for p in range(hole_end + 1, total_pages + 1):
                    next_entry = page_data.get(p)
                    if next_entry and next_entry['personnummer']:
                        pnr_after = next_entry['personnummer']
                        break

                # Check if we should fill the hole
                if pnr_before and pnr_after and pnr_before == pnr_after:
                    # Fill with the identical surrounding personnummer
                    if verbosity > 1:
                        if hole_start == hole_end:
                            print(f"--- Page {page_num} ---", file=sys.stderr)
                            print(f"  Hole-filling: No personnummer found, but surrounded by identical {pnr_before} - filling hole", 
                                  file=sys.stderr)
                        else:
                            print(f"--- Pages {hole_start}-{hole_end} ---", file=sys.stderr)
                            print(f"  Hole-filling: No personnummer found on {hole_end - hole_start + 1} consecutive pages, "
                                  f"but surrounded by identical {pnr_before} - filling hole", file=sys.stderr)

                    # Use the names from the surrounding entry (prefer the one before)
                    prev_entry = page_data.get(hole_start - 1)
                    if prev_entry:
                        final_data.append({
                            'page': page_num,
                            'personnummer': pnr_before,
                            'efternamn': prev_entry['efternamn'],
                            'fornamn': prev_entry['fornamn'],
                            'luhn_valid': True,
                            'was_corrected': False,
                            'hole_filled': True
                        })
                        filled = True

            # If not filled by hole-filling, use regular inheritance
            if not filled:
                if inherit and last_valid_entry:
                    if verbosity > 1:
                        print(f"--- Page {page_num} ---", file=sys.stderr)
                        print(f"  No personnummer found, inheriting {last_valid_entry['personnummer']} "
                              f"({last_valid_entry['efternamn']}, {last_valid_entry['fornamn']})", 
                              file=sys.stderr)
                    final_data.append({
                        'page': page_num,
                        'personnummer': last_valid_entry['personnummer'],
                        'efternamn': last_valid_entry['efternamn'],
                        'fornamn': last_valid_entry['fornamn'],
                        'luhn_valid': True,
                        'was_corrected': False
                    })
                else:
                    if verbosity > 1:
                        print(f"--- Page {page_num} ---", file=sys.stderr)
                        if not inherit:
                            print(f"  No personnummer found, inheritance disabled - empty row", file=sys.stderr)
                        else:
                            print(f"  No personnummer found, no previous entry to inherit - empty row", file=sys.stderr)
                    final_data.append({
                        'page': page_num,
                        'personnummer': "",
                        'efternamn': "",
                        'fornamn': "",
                        'luhn_valid': False,
                        'was_corrected': False
                    })

    return final_data, pnr_to_names, pages_with_pnr, pages_without_pnr

def should_filter_line(line, config):
    """Check if a line should be filtered out when searching for names."""
    line_lower = line.lower()
    return any(keyword in line_lower for keyword in config.filter_keywords)


def extract_special_personnummer(line, config, verbosity=0):
    """Extract personnummer from a line with flexible formatting.

    Requirements:
    - Line must have at least 8 digits total (minimum for TF numbers: 6+2)
    - Regular numbers need 10 digits (6+4), TF numbers need 8 digits (6+2) plus T and F
    - Find hyphen(s) that are next to each other or separated by spaces only
    - Search outward from hyphen to find digits
    - RIGHT of hyphen: either 4 digits OR T,F,2digits
    - LEFT of hyphen: 6 digits
    - Between last left digit and hyphen: only non-digit, non-hyphen chars
    - Between hyphen and first right element: only non-digit, non-hyphen (non-T for TF) chars
    - No hyphens allowed between digits on either side
    - If multiple matches, return the RIGHTMOST one

    Returns:
        tuple: (cleaned_pnr, original_substring, start_pos, end_pos) or None
    """
    # Apply unidecode to handle em dashes and other special characters
    decoded_line = unidecode.unidecode(line)  # FIXED: was unidecode(line)

    # QUICK REJECTION: Count total digits and check for minimum requirement
    digit_count = sum(1 for c in decoded_line if c.isdigit())

    # Minimum 8 digits needed (for TF numbers: 6 before + 2 after)
    # Regular numbers need 10 (6 before + 4 after)
    if digit_count < 8:
        return None  # Quick rejection - not enough digits

    # Find all hyphen sequences (one or more hyphens, possibly separated by spaces)
    hyphen_pattern = r'-(?:\s*-)*'
    hyphen_matches = list(re.finditer(hyphen_pattern, decoded_line))

    if not hyphen_matches:
        return None

    # Process from rightmost to leftmost hyphen sequence
    for hyphen_match in reversed(hyphen_matches):
        hyphen_start = hyphen_match.start()
        hyphen_end = hyphen_match.end()

        # === SEARCH RIGHT FROM HYPHEN ===
        right_part = decoded_line[hyphen_end:]
        right_is_tf = False
        right_end_pos = hyphen_end
        tf_digits = []
        right_digits = []

        # Look for first significant character (digit or T for TF number)
        first_right_char_pos = -1
        for i, char in enumerate(right_part):
            if char == '-':
                # Another hyphen found before any digit/T - invalid
                break
            if char.isdigit() or char.upper() == 'T':
                first_right_char_pos = i
                break

        if first_right_char_pos == -1:
            continue  # No digits or T found, try next hyphen

        # Check if it's a TF number (starts with T)
        if right_part[first_right_char_pos].upper() == 'T':
            t_pos = first_right_char_pos
            f_pos = -1

            # Find F after T (no hyphens allowed)
            for i in range(t_pos + 1, len(right_part)):
                if right_part[i] == '-':
                    break  # Hyphen found, invalid TF
                if right_part[i].upper() == 'F':
                    f_pos = i
                    break

            if f_pos > t_pos:
                # Find 2 digits after F (no hyphens allowed)
                for i in range(f_pos + 1, len(right_part)):
                    if right_part[i] == '-':
                        break  # Hyphen found, stop
                    if right_part[i].isdigit():
                        tf_digits.append(right_part[i])
                        if len(tf_digits) == 2:
                            right_is_tf = True
                            right_end_pos = hyphen_end + i + 1
                            break

        # If not TF, collect 4 digits
        if not right_is_tf and right_part[first_right_char_pos].isdigit():
            for i in range(first_right_char_pos, len(right_part)):
                if right_part[i] == '-':
                    break  # Hyphen found, stop
                if right_part[i].isdigit():
                    right_digits.append(right_part[i])
                    if len(right_digits) == 4:
                        right_end_pos = hyphen_end + i + 1
                        break

        # Check if we got valid right side
        if not (right_is_tf or len(right_digits) == 4):
            continue  # Invalid right side, try next hyphen

        # === SEARCH LEFT FROM HYPHEN ===
        left_part = decoded_line[:hyphen_start]
        left_digits = []
        left_start_pos = 0

        # Find last digit position (searching backwards)
        last_digit_pos = -1
        for i in range(len(left_part) - 1, -1, -1):
            if left_part[i].isdigit():
                last_digit_pos = i
                break

        if last_digit_pos == -1:
            continue  # No digits on left side

        # Check that between last digit and hyphen there are no digits or hyphens
        between_last_digit_and_hyphen = left_part[last_digit_pos + 1:]
        if any(c.isdigit() or c == '-' for c in between_last_digit_and_hyphen):
            continue  # Invalid characters between last digit and hyphen

        # Collect 6 digits going backwards (no hyphens allowed between them)
        for i in range(last_digit_pos, -1, -1):
            char = left_part[i]
            if char == '-':
                # Hyphen found while collecting digits
                if len(left_digits) < 6:
                    break  # Not enough digits
            elif char.isdigit():
                left_digits.insert(0, char)
                if len(left_digits) == 6:
                    left_start_pos = i
                    break

        # Check if we got 6 digits on the left
        if len(left_digits) != 6:
            continue

        # === CONSTRUCT RESULT ===
        if right_is_tf:
            cleaned_pnr = ''.join(left_digits) + '-TF' + ''.join(tf_digits)
        else:
            cleaned_pnr = ''.join(left_digits) + '-' + ''.join(right_digits)

        # Extract the original substring for logging
        original_substring = line[left_start_pos:right_end_pos]

        if verbosity > 1:
            print(f"    Special extraction: '{line.strip()}' → found '{original_substring}' → cleaned '{cleaned_pnr}'", file=sys.stderr)

        return (cleaned_pnr, original_substring, left_start_pos, right_end_pos)

    return None


def select_closest_personnummer(results, page_num, last_captured_pnr, verbosity):
    """Select the personnummer closest to (but larger than) the last captured one.

    Args:
        results: List of personnummer results from current page
        page_num: Page number for logging
        last_captured_pnr: Last valid personnummer from preceding pages
        verbosity: Verbosity level

    Returns:
        Single result (the closest) or original results if only one
    """
    if len(results) <= 1:
        return results

    # Multiple personnummer found - need to select one
    if verbosity >= 0:
        print(f"\n  Multiple personnummer on page {page_num} - selecting closest to reference:", file=sys.stderr)
        if last_captured_pnr:
            print(f"    Reference (last captured): {last_captured_pnr}", file=sys.stderr)
        else:
            print(f"    No reference available - selecting first valid one", file=sys.stderr)

        # Show all candidates with their validity status
        for result in results:
            status_parts = []
            if 'TF' in result['personnummer']:
                if result.get('date_valid'):
                    status_parts.append("valid TF number")
                else:
                    status_parts.append(f"invalid TF number ({result.get('rejection_reason', 'unknown')})")
            elif result.get('date_valid'):
                if result['luhn_valid']:
                    status_parts.append("fully valid")
                else:
                    status_parts.append("valid date, invalid Luhn")
            else:
                status_parts.append(f"invalid date ({result.get('rejection_reason', 'unknown')})")

            if result.get('is_samordningsnummer'):
                status_parts.append("samordningsnummer")

            print(f"    Candidate: {result['personnummer']} ({', '.join(status_parts)})", file=sys.stderr)

    # If no reference, prefer fully valid ones
    if not last_captured_pnr:
        # Try to find a fully valid one (valid date AND luhn, or valid TF)
        for result in results:
            is_valid_tf = 'TF' in result['personnummer'] and result.get('date_valid')
            is_valid_regular = result.get('date_valid') and result['luhn_valid'] and 'TF' not in result['personnummer']
            if is_valid_tf or is_valid_regular:
                if verbosity >= 0:
                    print(f"    SELECTED: {result['personnummer']} (first fully valid)", file=sys.stderr)
                return [result]
        # Otherwise just take first one
        if verbosity >= 0:
            print(f"    SELECTED: {results[0]['personnummer']} (first overall)", file=sys.stderr)
        return [results[0]]

    # Convert personnummer to comparable numbers (remove hyphen)
    reference_num = int(last_captured_pnr.replace('-', '').replace('TF', '99'))  # TF becomes 99 for comparison

    candidates = []
    for result in results:
        pnr_clean = result['personnummer'].replace('-', '').replace('TF', '99')
        pnr_num = int(pnr_clean)

        if pnr_num > reference_num:  # Only consider larger personnummer
            difference = pnr_num - reference_num
            # Add validity score for sorting (prefer valid ones)
            validity_score = 0
            if result.get('date_valid'):
                validity_score += 2
            if result.get('luhn_valid') or 'TF' in result['personnummer']:
                validity_score += 1
            candidates.append((result, difference, pnr_num, validity_score))
            if verbosity >= 0:
                print(f"    {result['personnummer']} → {pnr_num} (difference: +{difference}, validity score: {validity_score}) ✓", file=sys.stderr)
        else:
            if verbosity >= 0:
                print(f"    {result['personnummer']} → {pnr_num} (smaller than reference, rejected)", file=sys.stderr)

    if not candidates:
        # No personnummer larger than reference - take first valid or first overall
        for result in results:
            is_valid_tf = 'TF' in result['personnummer'] and result.get('date_valid')
            is_valid_regular = result.get('date_valid') and result['luhn_valid'] and 'TF' not in result['personnummer']
            if is_valid_tf or is_valid_regular:
                if verbosity >= 0:
                    print(f"    SELECTED: {result['personnummer']} (no larger candidates, first fully valid)", file=sys.stderr)
                return [result]
        if verbosity >= 0:
            print(f"    SELECTED: {results[0]['personnummer']} (no larger candidates, first overall)", file=sys.stderr)
        return [results[0]]

    # Sort by: validity score (desc), then difference (asc)
    candidates.sort(key=lambda x: (-x[3], x[1]))
    selected_result, difference, selected_num, validity_score = candidates[0]

    if verbosity >= 0:
        print(f"    SELECTED: {selected_result['personnummer']} (closest with difference +{difference}, validity score {validity_score})", file=sys.stderr)

    return [selected_result]


def find_all_personnummer_on_page(text: str, page_num: int, config: Config, verbosity: int) -> List[Dict[str, Any]]:
    """Find all personnummer on a page.

    Early rejects only months 20-99. All other invalid dates are processed for 
    potential autocorrection.
    """
    lines = text.split('\n')

    if verbosity > 1:
        print(f"\n--- Page {page_num} ---", file=sys.stderr)

    # STEP 1: Try normal regex matching first
    results, early_rejected = _find_personnummer_with_regex(lines, config, verbosity)

    # STEP 2: Only try special extraction if no personnummer found (valid or invalid-but-correctable)
    if not results:
        if verbosity > 1:
            print(f"  No personnummer found with normal regex, trying special extraction...", file=sys.stderr)

        special_results, special_rejected = _find_personnummer_with_special_extraction(lines, config, verbosity)
        results.extend(special_results)
        early_rejected.extend(special_rejected)

    # Summary logging
    if verbosity > 1:
        total_early_rejected = len(early_rejected)
        total_late_rejected = sum(1 for r in results if not r['date_valid'])
        total_valid = sum(1 for r in results if r['date_valid'])
        total_tf = sum(1 for r in results if 'TF' in r['personnummer'])
        total_samordning = sum(1 for r in results if r.get('is_samordningsnummer', False))

        parts = []
        if total_early_rejected > 0:
            parts.append(f"{total_early_rejected} early rejected (months 20-99)")
        if total_late_rejected > 0:
            parts.append(f"{total_late_rejected} invalid but correctable")
        if total_valid > 0:
            parts.append(f"{total_valid} valid")
        if total_tf > 0:
            parts.append(f"{total_tf} TF numbers")
        if total_samordning > 0:
            parts.append(f"{total_samordning} samordningsnummer")

        if parts:
            print(f"  Page summary: {', '.join(parts)}", file=sys.stderr)
        else:
            print(f"  Page summary: No personnummer found", file=sys.stderr)

    return results


def _extract_names_from_same_line(
    line: str, 
    pnr: str, 
    config: Config, 
    verbosity: int, 
    match_info: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """Extract names from the same line as personnummer.

    Returns:
        Tuple of (efternamn, fornamn) or ("", "") if not found
    """
    # Find personnummer position - use match info if available
    pnr_pos = -1

    if match_info and 'match_start' in match_info:
        pnr_pos = match_info['match_start']
        if verbosity > 1:
            extraction_type = match_info.get('extraction_type', 'unknown')
            print(f"    Using stored match position from {extraction_type} extraction: pos {pnr_pos}", file=sys.stderr)
    else:
        # Fallback to searching
        pnr_pos = line.find(pnr)
        if pnr_pos == -1:
            # Try to find pattern match
            match = re.search(pnr.replace('-', r'\-'), line)
            if match:
                pnr_pos = match.start()

    # Extract prefix (everything before personnummer)
    prefix = line[:pnr_pos].strip() if pnr_pos > 0 else ""

    if prefix:
        efternamn, fornamn = parse_name_from_text(prefix, config, verbosity)
        if verbosity > 1 and efternamn:
            format_type = "comma" if ',' in prefix else "space"
            print(f"    Found name on same line ({format_type}): {efternamn}, {fornamn}", file=sys.stderr)
        return efternamn, fornamn

    return "", ""


def _search_previous_lines_for_names(
    lines: List[str], 
    start_idx: int, 
    config: Config, 
    verbosity: int
) -> Tuple[str, str]:
    """Search previous lines for names.

    Returns:
        Tuple of (efternamn, fornamn) or ("", "") if not found
    """
    pattern = config.personnummer_pattern
    found_lines = []
    found_single_word = False

    for prev_idx in range(start_idx - 1, -1, -1):
        prev_line = lines[prev_idx].strip()
        if not prev_line:
            continue

        # Skip filtered lines
        if should_filter_line(prev_line, config):
            if verbosity > 1:
                print(f"    Filtered line: '{prev_line}'", file=sys.stderr)
            continue

        # Skip lines with personnummer
        if re.search(pattern, prev_line) or extract_special_personnummer(prev_line, config):
            continue

        # Check for comma-separated format
        if ',' in prev_line:
            found_lines.append(prev_line)
            break
        else:
            # Check word count for space-separated format
            words = clean_name(prev_line, config).split()
            if len(words) == 1 and words[0]:
                found_lines.append(prev_line)
                if found_single_word:
                    break
                found_single_word = True
                continue
            elif len(words) > 1:
                found_lines.append(prev_line)
                break

    # Parse found lines
    if found_lines:
        if len(found_lines) > 1:
            combined_line = ' '.join(reversed(found_lines))
            if verbosity > 1:
                print(f"    Combined {len(found_lines)} lines: '{combined_line}'", file=sys.stderr)
        else:
            combined_line = found_lines[0]

        efternamn, fornamn = parse_name_from_text(combined_line, config, verbosity)

        if efternamn and verbosity > 1:
            lines_back = start_idx - lines.index(found_lines[-1])
            print(f"    Found name {lines_back} lines above: {efternamn}, {fornamn}", file=sys.stderr)

        return efternamn, fornamn

    return "", ""


def _find_personnummer_with_regex(
    lines: List[str], 
    config: Config, 
    verbosity: int
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Find personnummer using regular expression matching.

    Returns:
        Tuple of (results, rejected_with_reasons)
    """
    results = []
    rejected_with_reasons = []
    pattern = config.personnummer_pattern

    for line_idx, line in enumerate(lines):
        for match in re.finditer(r'\S*' + pattern + r'\S*', line):
            full_match = match.group(0)
            cleaned_pnr = re.search(pattern, full_match).group(0)

            # Validate date portion
            date_valid, rejection_reason, is_samordningsnummer, is_early_rejection = validate_personnummer_date(cleaned_pnr, verbosity)

            if is_early_rejection:
                # EARLY REJECTION - no name extraction, no autocorrection
                rejected_with_reasons.append((cleaned_pnr, rejection_reason))
                if verbosity > 1:
                    print(f"  EARLY REJECTED: {cleaned_pnr} ({rejection_reason} - no name extraction)", file=sys.stderr)
            else:
                # Either valid or late rejection - proceed with name extraction
                luhn_valid = True
                if 'TF' not in cleaned_pnr:
                    pnr_no_hyphen = cleaned_pnr.replace('-', '')
                    luhn_valid = luhn_check(pnr_no_hyphen)

                results.append({
                    'personnummer': cleaned_pnr,
                    'line_idx': line_idx,
                    'luhn_valid': luhn_valid,
                    'date_valid': date_valid,
                    'luhn_check_valid': luhn_valid,
                    'is_samordningsnummer': is_samordningsnummer,
                    'extraction_type': 'normal',
                    'original_match': full_match,
                    'match_start': match.start(),
                    'match_end': match.end(),
                    'rejection_reason': rejection_reason if not date_valid else None
                })

                if verbosity > 1:
                    if 'TF' in cleaned_pnr:
                        if date_valid:
                            validity_str = "(valid TF number)"
                        else:
                            validity_str = f"(invalid TF number: {rejection_reason} - will attempt autocorrection)"
                    elif not date_valid:
                        validity_str = f"(invalid date: {rejection_reason} - will attempt name extraction and autocorrection)"
                    elif is_samordningsnummer:
                        if not luhn_valid:
                            validity_str = "(valid samordningsnummer date, invalid Luhn)"
                        else:
                            validity_str = "(valid samordningsnummer)"
                    elif not luhn_valid:
                        validity_str = "(valid date, invalid Luhn)"
                    else:
                        validity_str = "(valid)"
                    print(f"  Found personnummer: {cleaned_pnr} {validity_str}", file=sys.stderr)

    return results, rejected_with_reasons


def _find_personnummer_with_special_extraction(
    lines: List[str], 
    config: Config, 
    verbosity: int
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Find personnummer using special extraction on all lines.

    Args:
        lines: List of text lines to search
        config: Configuration object
        verbosity: Debug output level

    Returns:
        Tuple of (results, rejected_with_reasons)
    """
    results = []
    rejected_with_reasons = []

    for line_idx, line in enumerate(lines):
        # Try special extraction on this line
        special_result = extract_special_personnummer(line, config, verbosity)

        if special_result:
            cleaned_pnr, original_substring, start_pos, end_pos = special_result

            # Validate the extracted personnummer
            date_valid, rejection_reason, is_samordningsnummer, is_early_rejection = validate_personnummer_date(cleaned_pnr, verbosity)

            if is_early_rejection:
                # EARLY REJECTION - no name extraction
                rejected_with_reasons.append((cleaned_pnr, rejection_reason))
                if verbosity > 1:
                    print(f"  EARLY REJECTED (special): {cleaned_pnr} ({rejection_reason})", file=sys.stderr)
            else:
                # Either valid or late rejection - proceed with result
                luhn_valid = True
                if 'TF' not in cleaned_pnr:
                    pnr_no_hyphen = cleaned_pnr.replace('-', '')
                    luhn_valid = luhn_check(pnr_no_hyphen)

                results.append({
                    'personnummer': cleaned_pnr,
                    'line_idx': line_idx,
                    'luhn_valid': luhn_valid,
                    'date_valid': date_valid,
                    'luhn_check_valid': luhn_valid,
                    'is_samordningsnummer': is_samordningsnummer,
                    'extraction_type': 'special',
                    'original_match': original_substring,
                    'match_start': start_pos,
                    'match_end': end_pos,
                    'rejection_reason': rejection_reason if not date_valid else None
                })

                if verbosity > 1:
                    if 'TF' in cleaned_pnr:
                        if date_valid:
                            validity_str = "(valid TF number)"
                        else:
                            validity_str = f"(invalid TF number: {rejection_reason})"
                    elif not date_valid:
                        validity_str = f"(invalid date: {rejection_reason})"
                    elif is_samordningsnummer:
                        if not luhn_valid:
                            validity_str = "(valid samordningsnummer date, invalid Luhn)"
                        else:
                            validity_str = "(valid samordningsnummer)"
                    elif not luhn_valid:
                        validity_str = "(valid date, invalid Luhn)"
                    else:
                        validity_str = "(valid)"
                    print(f"  Found personnummer (special): {cleaned_pnr} {validity_str}", file=sys.stderr)

    return results, rejected_with_reasons


def _process_inheritance(
    current_page_entries: Dict[int, List[Dict]], 
    total_pages: int, 
    inherit: bool, 
    verbosity: int
) -> Tuple[List[Dict], Dict[str, List], int, int]:
    """Process inheritance logic for pages without personnummer.

    Returns:
        Tuple of (final_data, pnr_to_names, pages_with_pnr, pages_without_pnr)
    """
    final_data = []
    pnr_to_names = defaultdict(list)
    last_valid_entry = None
    pages_with_pnr = 0
    pages_without_pnr = 0

    for page_num in range(1, total_pages + 1):
        if page_num in current_page_entries:
            page_entries = current_page_entries[page_num]

            # Guarantee exactly one entry per page
            if len(page_entries) != 1:
                raise RuntimeError(f"Internal error: Expected exactly 1 entry for page {page_num}, got {len(page_entries)}")

            entry = page_entries[0]
            pages_with_pnr += 1

            pnr = entry['personnummer']

            # Track name combinations for valid personnummer
            if pnr:
                name_combo = (entry['efternamn'], entry['fornamn'])
                if name_combo != ("", ""):
                    pnr_to_names[pnr].append(name_combo)

            # Add to final data
            final_data.append({
                'page': page_num,
                'personnummer': pnr if pnr else "",
                'efternamn': entry['efternamn'],
                'fornamn': entry['fornamn'],
                'luhn_valid': entry['luhn_valid'],
                'was_corrected': entry.get('was_corrected', False)
            })

            # Update last valid entry for inheritance
            if pnr:
                last_valid_entry = {
                    'personnummer': pnr,
                    'efternamn': entry['efternamn'],
                    'fornamn': entry['fornamn']
                }
        else:
            # No entries on this page
            pages_without_pnr += 1
            if inherit and last_valid_entry:
                if verbosity > 1:
                    print(f"--- Page {page_num} ---", file=sys.stderr)
                    print(f"  No personnummer found, inheriting {last_valid_entry['personnummer']} "
                          f"({last_valid_entry['efternamn']}, {last_valid_entry['fornamn']})", 
                          file=sys.stderr)
                final_data.append({
                    'page': page_num,
                    'personnummer': last_valid_entry['personnummer'],
                    'efternamn': last_valid_entry['efternamn'],
                    'fornamn': last_valid_entry['fornamn'],
                    'luhn_valid': True,
                    'was_corrected': False
                })
            else:
                if verbosity > 1:
                    print(f"--- Page {page_num} ---", file=sys.stderr)
                    if not inherit:
                        print(f"  No personnummer found, inheritance disabled - empty row", file=sys.stderr)
                    else:
                        print(f"  No personnummer found, no previous entry to inherit - empty row", file=sys.stderr)
                final_data.append({
                    'page': page_num,
                    'personnummer': "",
                    'efternamn': "",
                    'fornamn': "",
                    'luhn_valid': False,
                    'was_corrected': False
                })

    return final_data, pnr_to_names, pages_with_pnr, pages_without_pnr


def _resolve_name_variants(
    pnr_to_names: Dict[str, List], 
    config: Config, 
    verbosity: int
) -> Dict[str, Tuple[str, str]]:
    """Resolve name variants for each personnummer.

    Returns:
        Dictionary mapping personnummer to final (efternamn, fornamn)
    """
    if verbosity > 0 and pnr_to_names:
        print("\nResolving name variants for each personnummer...", file=sys.stderr)

    pnr_final_names = {}
    scandinavian_preferences = 0
    tied_selections = []

    for pnr, name_list in pnr_to_names.items():
        if name_list:
            name_counter = Counter(name_list)

            # Use the selection algorithm with enhanced scoring
            selected_name = select_best_name_variant(name_counter, pnr, config, verbosity)

            # Check if enhanced scoring was applied
            most_common_by_count = name_counter.most_common(1)[0][0]
            if selected_name != most_common_by_count:
                if has_scandinavian_chars(selected_name[0] + selected_name[1]):
                    scandinavian_preferences += 1

            # Check for ties in original counting
            if len(name_counter) > 1:
                top_count = name_counter.most_common(1)[0][1]
                ties = [(name, count) for name, count in name_counter.items() if count == top_count]
                if len(ties) > 1:
                    tied_selections.append((pnr, selected_name, len(ties)))
            elif verbosity > 1:
                # Only one variant found
                print(f"  {pnr}: {selected_name[0]}, {selected_name[1]} (unanimous)", file=sys.stderr)

            pnr_final_names[pnr] = selected_name

    if scandinavian_preferences > 0 and verbosity >= 0:
        print(f"\nApplied enhanced scoring (Scandinavian/penalty) preference to {scandinavian_preferences} names", file=sys.stderr)

    if tied_selections and verbosity >= 0:
        print(f"\nResolved {len(tied_selections)} ties in name variants:", file=sys.stderr)
        if verbosity > 0:
            for pnr, selected_name, num_ties in tied_selections[:10]:
                print(f"  {pnr}: Selected '{selected_name[0]}, {selected_name[1]}' from {num_ties} equally common variants", 
                      file=sys.stderr)

    return pnr_final_names


def _apply_final_names_to_entries(
    final_data: List[Dict], 
    pnr_final_names: Dict[str, Tuple[str, str]], 
    verbosity: int
) -> Tuple[int, int]:
    """Apply the resolved names to all entries.

    Returns:
        Tuple of (updates_made, names_added)
    """
    if verbosity > 0 and pnr_final_names:
        print("\nApplying selected names to all pages...", file=sys.stderr)

    updates_made = 0
    names_added = 0

    for entry in final_data:
        pnr = entry['personnummer']
        if pnr and pnr in pnr_final_names:
            final_name = pnr_final_names[pnr]

            # Check if we need to add or update names
            if entry['efternamn'] == "" and entry['fornamn'] == "":
                entry['efternamn'] = final_name[0]
                entry['fornamn'] = final_name[1]
                names_added += 1
                if verbosity > 1:
                    print(f"  Page {entry['page']}: Added missing name for {pnr}: '{final_name[0]}, {final_name[1]}'", 
                          file=sys.stderr)
            elif (entry['efternamn'], entry['fornamn']) != final_name:
                old_name = f"{entry['efternamn']}, {entry['fornamn']}"
                entry['efternamn'] = final_name[0]
                entry['fornamn'] = final_name[1]
                updates_made += 1
                if verbosity > 1:
                    print(f"  Page {entry['page']}: Updated {pnr} from '{old_name}' to '{final_name[0]}, {final_name[1]}'", 
                          file=sys.stderr)

    return updates_made, names_added


def extract_names_for_personnummer(
    text: str, 
    selected_pnr: str, 
    selected_line_idx: int, 
    config: Config, 
    verbosity: int, 
    match_info: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """Extract names for a specific selected personnummer with name swapping check."""
    lines = text.split('\n')

    # Step 1: Try same line extraction
    efternamn, fornamn = _extract_names_from_same_line(
        lines[selected_line_idx], selected_pnr, config, verbosity, match_info
    )

    if not efternamn:
        # Step 2: Search previous lines
        efternamn, fornamn = _search_previous_lines_for_names(
            lines, selected_line_idx, config, verbosity
        )

    # Check if names should be swapped (if SCB names enabled)
    if config.use_scb_names and efternamn and fornamn:  # RENAMED from use_json_names
        efternamn, fornamn, swapped = check_and_swap_names(
            efternamn, fornamn, 
            config.first_names_set, config.last_names_set, 
            verbosity
        )

    return efternamn, fornamn


def extract_personnummer_and_names(text, page_num, config, verbosity, include_invalid=False, last_captured_pnr=None):
    """Extract personnummer and names - handling late vs early rejection for name keeping."""

    # Step 1: Find ALL personnummer on page (both valid and invalid)
    all_pnr_results = find_all_personnummer_on_page(text, page_num, config, verbosity)

    if not all_pnr_results:
        if verbosity > 1:
            print(f"  No personnummer found on this page", file=sys.stderr)
        return []

    # Step 2: Filter out early rejections (these won't get names)
    # Early rejections are those with months 20-99 (organizational numbers)
    processable_results = []
    for result in all_pnr_results:
        # Check if this was an early rejection
        if result.get('rejection_reason') and 'organizational number' in result.get('rejection_reason', ''):
            if verbosity > 1:
                print(f"  Skipping early rejected {result['personnummer']} - no name extraction", file=sys.stderr)
            continue
        processable_results.append(result)

    if not processable_results:
        return []

    # Step 3: Select ONE personnummer from processable results
    if len(processable_results) > 1:
        selected_results = select_closest_personnummer(processable_results, page_num, last_captured_pnr, verbosity)
    else:
        selected_results = processable_results
        if verbosity > 1:
            pnr_info = processable_results[0]
            status_parts = []
            if pnr_info.get('date_valid'):
                if pnr_info['luhn_valid']:
                    status_parts.append("valid")
                else:
                    status_parts.append("valid date, invalid Luhn")
            else:
                status_parts.append(f"invalid date: {pnr_info.get('rejection_reason', 'unknown')}")
            if pnr_info.get('is_samordningsnummer'):
                status_parts.append("samordningsnummer")
            status = ", ".join(status_parts)
            print(f"  Using personnummer: {pnr_info['personnummer']} ({status})", file=sys.stderr)

    if not selected_results:
        return []

    selected = selected_results[0]

    # Step 4: Extract names for the selected personnummer
    # Names are extracted even for late rejections (invalid date/Luhn but not organizational)
    match_info = {
        'extraction_type': selected.get('extraction_type', 'normal'),
        'match_start': selected.get('match_start'),
        'match_end': selected.get('match_end'),
        'original_match': selected.get('original_match')
    }

    efternamn, fornamn = extract_names_for_personnummer(
        text, selected['personnummer'], selected['line_idx'], config, verbosity, match_info)

    # Step 5: Determine if personnummer should be kept based on validity and include_invalid flag
    final_personnummer = selected['personnummer']

    # For late rejection: if both names were extracted, keep them even if personnummer is rejected
    is_late_rejection = not selected['date_valid'] and not (
        selected.get('rejection_reason') and 'organizational number' in selected.get('rejection_reason', '')
    )

    # Clear personnummer if invalid and not including invalid, but keep names for late rejection
    if not selected['luhn_valid'] and not include_invalid and 'TF' not in selected['personnummer']:
        if is_late_rejection and efternamn and fornamn:
            if verbosity > 1:
                print(f"  Late rejection of {selected['personnummer']}: keeping extracted names '{efternamn}, {fornamn}'", 
                      file=sys.stderr)
        final_personnummer = ""  # Clear invalid personnummer unless include_invalid is set

    # Step 6: Return complete result
    return [{
        'personnummer': final_personnummer if (include_invalid or selected['luhn_valid'] or 'TF' in selected['personnummer']) else "",
        'efternamn': efternamn,
        'fornamn': fornamn,
        'luhn_valid': selected['luhn_valid'],
        'original_pnr': selected['personnummer'],
        'date_valid': selected['date_valid'],
        'luhn_check_valid': selected['luhn_check_valid'],
        'is_samordningsnummer': selected.get('is_samordningsnummer', False),
        'was_early_rejection': selected.get('rejection_reason') and 'organizational number' in selected.get('rejection_reason', ''),
        'was_late_rejection': is_late_rejection
    }]

def ocr_single_page(args):
    """OCR a single page with improved error handling, SCB names dictionary, and personnummer patterns."""
    page_file, page_num, tmpdir, config = args

    text_file = os.path.join(tmpdir, f"page_{page_num:04d}.txt")

    # Verify input file exists
    if not os.path.exists(page_file):
        return (page_num, "", f"Input PDF page not found: {page_file}")

    # Build OCR command
    ocr_cmd = [
        "ocrmypdf",
        "-q",  # Always quiet for ocrmypdf to avoid interleaved output
        "-l", config.ocr_languages,
        "--force-ocr",
        "--optimize", str(config.ocr_optimize_level),
        "--jpeg-quality", str(config.jpeg_quality),
        "--png-quality", str(config.png_quality),
        "--sidecar", text_file
    ]

    # Add SCB names dictionary if available
    if config.scb_names_dict_path and os.path.exists(config.scb_names_dict_path):
        ocr_cmd.extend(["--user-words", config.scb_names_dict_path])

    # Add personnummer patterns if available
    if config.pnr_patterns_path and os.path.exists(config.pnr_patterns_path):
        ocr_cmd.extend(["--user-patterns", config.pnr_patterns_path])

    # Add input and output files
    ocr_cmd.extend([page_file, "-"])  # Output to stdout (we don't need the PDF)

    # Use safe subprocess run with timeout
    result = safe_subprocess_run(
        ocr_cmd, 
        f"OCR for page {page_num}",
        verbosity=0,  # Keep quiet during parallel processing
        capture_output=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300  # 5 minute timeout per page
    )

    # Handle different error conditions
    if result.returncode == -1:  # Timeout
        return (page_num, "", f"OCR timed out (>5min)")
    elif result.returncode == -2:  # Command not found
        return (page_num, "", f"ocrmypdf command not found - please install ocrmypdf")
    elif result.returncode == -3:  # Unexpected error
        return (page_num, "", f"OCR unexpected error: {result.stderr[:100]}")
    elif result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "Unknown OCR error"
        # Filter out common non-critical warnings
        if error_msg and not any(warning in error_msg.lower() for warning in ["warning", "info:", "note:"]):
            return (page_num, "", f"OCR failed: {error_msg[:100]}")

    # Read OCR output with safe file reading
    text, error = safe_file_read(text_file, f"OCR output for page {page_num}")

    if error:
        return (page_num, "", f"Could not read OCR output: {error}")

    # Clean up temporary text file
    try:
        os.remove(text_file)
    except OSError:
        pass  # Not critical if cleanup fails

    return (page_num, text, None)

def safe_create_tempdir():
    """Safely create temporary directory with error handling."""
    try:
        return tempfile.mkdtemp()
    except OSError as e:
        print(f"Error: Could not create temporary directory: {e}", file=sys.stderr)
        sys.exit(1)


def safe_cleanup_tempdir(tmpdir, verbosity=0):
    """Safely cleanup temporary directory."""
    if tmpdir and os.path.exists(tmpdir):
        try:
            import shutil
            shutil.rmtree(tmpdir)
            if verbosity > 1:
                print(f"    Cleaned up temporary directory", file=sys.stderr)
        except OSError as e:
            if verbosity > 0:
                print(f"    Warning: Could not cleanup temporary directory: {e}", file=sys.stderr)

def validate_input_file(pdf_path, verbosity=0):
    """Validate input PDF file exists and is readable."""
    if not pdf_path:
        print("Error: No PDF file specified", file=sys.stderr)
        return False

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        return False

    if not os.path.isfile(pdf_path):
        print(f"Error: Path is not a file: {pdf_path}", file=sys.stderr)
        return False

    try:
        # Test if file is readable
        with open(pdf_path, 'rb') as f:
            f.read(1024)  # Read first KB to test accessibility
        return True
    except PermissionError:
        print(f"Error: Permission denied reading PDF file: {pdf_path}", file=sys.stderr)
        return False
    except OSError as e:
        print(f"Error: Could not read PDF file: {e}", file=sys.stderr)
        return False

def check_required_commands(verbosity=0):
    """Check if required external commands are available."""
    required_commands = ['pdfinfo', 'pdfseparate', 'ocrmypdf']
    missing_commands = []

    for cmd in required_commands:
        result = safe_subprocess_run(
            [cmd, '--version'], 
            f"Check {cmd} availability",
            verbosity=0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode == -2:  # Command not found
            missing_commands.append(cmd)

    if missing_commands:
        print(f"Error: Required commands not found: {', '.join(missing_commands)}", file=sys.stderr)
        print("Please install the missing packages:", file=sys.stderr)
        if 'pdfinfo' in missing_commands or 'pdfseparate' in missing_commands:
            print("  - poppler-utils (for pdfinfo and pdfseparate)", file=sys.stderr)
        if 'ocrmypdf' in missing_commands:
            print("  - ocrmypdf (for OCR functionality)", file=sys.stderr)
        return False

    if verbosity > 0:
        print("All required external commands are available", file=sys.stderr)

    return True

def report_names_not_in_lists(final_data, config, verbosity):
    """Report names not found in the SCB name lists, including those without personnummer.

    Separates reporting into two sections:
    1. Names WITH personnummer not in SCB lists
    2. Names WITHOUT personnummer not in SCB lists
    """
    if not config.use_scb_names or verbosity < 0:
        return

    # Collect entries with names not in lists - separated by whether they have personnummer
    with_pnr_not_in_lists = []
    without_pnr_not_in_lists = []

    for entry in final_data:
        fornamn = entry['fornamn']
        efternamn = entry['efternamn']

        if not fornamn or not efternamn:
            continue

        fornamn_not_in_list = fornamn not in config.first_names_set
        efternamn_not_in_list = efternamn not in config.last_names_set

        if fornamn_not_in_list or efternamn_not_in_list:
            item = {
                'personnummer': entry.get('personnummer', ''),
                'fornamn': fornamn,
                'efternamn': efternamn,
                'fornamn_not_in_list': fornamn_not_in_list,
                'efternamn_not_in_list': efternamn_not_in_list,
                'page': entry['page']
            }

            if entry.get('personnummer'):
                with_pnr_not_in_lists.append(item)
            else:
                without_pnr_not_in_lists.append(item)

    # Report names WITH personnummer not in lists
    if with_pnr_not_in_lists:
        print(f"\n=== Names Not in SCB Lists (WITH Personnummer) ===", file=sys.stderr)
        print(f"Found {len(with_pnr_not_in_lists)} entries with names not in the official Swedish name lists:", 
              file=sys.stderr)

        # Group by personnummer to avoid duplicates
        by_pnr = {}
        for item in with_pnr_not_in_lists:
            pnr = item['personnummer']
            if pnr not in by_pnr:
                by_pnr[pnr] = item

        # Sort using the 95-year rule
        sorted_items = sorted(by_pnr.items(), key=lambda x: personnummer_sort_key(x[0]))

        for pnr, item in sorted_items:
            issues = []
            if item['fornamn_not_in_list']:
                issues.append(f"förnamn '{item['fornamn']}' not in list")
            if item['efternamn_not_in_list']:
                issues.append(f"efternamn '{item['efternamn']}' not in list")

            print(f"  {pnr}: {item['efternamn']}, {item['fornamn']} - {'; '.join(issues)}", 
                  file=sys.stderr)

    # Report names WITHOUT personnummer not in lists
    if without_pnr_not_in_lists:
        print(f"\n=== Names Not in SCB Lists (WITHOUT Personnummer) ===", file=sys.stderr)
        print(f"Found {len(without_pnr_not_in_lists)} entries without personnummer with names not in lists:", 
              file=sys.stderr)

        # Group by name to show page numbers
        by_name = {}
        for item in without_pnr_not_in_lists:
            name_key = (item['efternamn'], item['fornamn'], item['fornamn_not_in_list'], item['efternamn_not_in_list'])
            if name_key not in by_name:
                by_name[name_key] = []
            by_name[name_key].append(item['page'])

        # Sort by name
        sorted_names = sorted(by_name.items(), key=lambda x: (x[0][0], x[0][1]))

        for (efternamn, fornamn, fornamn_not_in_list, efternamn_not_in_list), pages in sorted_names:
            issues = []
            if fornamn_not_in_list:
                issues.append(f"förnamn '{fornamn}' not in list")
            if efternamn_not_in_list:
                issues.append(f"efternamn '{efternamn}' not in list")

            pages_str = ', '.join(str(p) for p in sorted(pages))
            print(f"  {efternamn}, {fornamn} (pages: {pages_str}) - {'; '.join(issues)}", 
                  file=sys.stderr)


def process_pdf(pdf_path, verbosity, max_workers=None, include_invalid=False, inherit=True, 
                autocorrect=True, use_scb_names=True, use_pnr_patterns=True, disable_hole_filling=False):  # NEW parameter
    """Process PDF file with OCR and extract data."""
    # Validate inputs and environment
    if not validate_input_file(pdf_path, verbosity):
        sys.exit(1)

    if not check_required_commands(verbosity):
        sys.exit(1)

    # Create configuration instance
    config = Config()
    config.use_scb_names = use_scb_names
    config.use_pnr_patterns = use_pnr_patterns

    # Load name lists if enabled
    if use_scb_names:
        config.first_names_set, config.last_names_set = load_or_fetch_name_lists(
            use_scb_names=True,
            force_refresh=False, 
            verbosity=verbosity
        )

    if verbosity >= 0:
        print(f"Processing: {pdf_path}", file=sys.stderr)
        if include_invalid:
            print("  Mode: Including invalid personnummer (even if Luhn check fails)", file=sys.stderr)
        else:
            print("  Mode: Invalid personnummer will have empty personnummer field", file=sys.stderr)

        if autocorrect:
            print("  Autocorrection: ENABLED - will try to fix single-digit OCR errors", file=sys.stderr)
        else:
            print("  Autocorrection: DISABLED", file=sys.stderr)

        if not inherit:
            print("  Inheritance: DISABLED - pages without personnummer will have empty field", file=sys.stderr)
        else:
            print("  Inheritance: ENABLED - pages without personnummer inherit from previous", file=sys.stderr)

        if disable_hole_filling:
            print("  Hole-filling: DISABLED - holes won't be filled even with identical surrounding IDs", file=sys.stderr)
        else:
            print("  Hole-filling: ENABLED - holes filled when surrounded by identical personnummer", file=sys.stderr)

        if use_scb_names:
            if config.first_names_set or config.last_names_set:
                print(f"  SCB name lists: ENABLED - {len(config.first_names_set)} first names, "
                      f"{len(config.last_names_set)} last names (used for OCR enhancement and validation)", 
                      file=sys.stderr)
            else:
                print("  SCB name lists: ENABLED but empty or could not be fetched", file=sys.stderr)
        else:
            print("  SCB name lists: DISABLED", file=sys.stderr)

        if use_pnr_patterns:
            print("  Personnummer patterns: ENABLED - helping OCR recognize ID number formats", file=sys.stderr)
        else:
            print("  Personnummer patterns: DISABLED", file=sys.stderr)

    # Create temporary directory for processing with proper error handling
    tmpdir = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            total_pages = setup_pdf_pages(pdf_path, tmpdir, verbosity)
            page_texts = run_parallel_ocr(tmpdir, total_pages, max_workers, config, verbosity)
            raw_data = extract_all_data(page_texts, config, verbosity, include_invalid)

            if autocorrect:
                raw_data = apply_autocorrection(raw_data, verbosity)

            final_data = apply_inheritance_and_resolve_names(
                raw_data, total_pages, inherit, config, verbosity, include_invalid, disable_hole_filling
            )

            # Report names not in lists if SCB names enabled
            if use_scb_names:
                report_names_not_in_lists(final_data, config, verbosity)

            return final_data

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        if tmpdir:
            safe_cleanup_tempdir(tmpdir, verbosity)
        sys.exit(130)
    except Exception as e:
        print(f"Error: Unexpected error during processing: {e}", file=sys.stderr)
        if verbosity > 0:
            import traceback
            traceback.print_exc()
        if tmpdir:
            safe_cleanup_tempdir(tmpdir, verbosity)
        sys.exit(1)


def safe_subprocess_run(cmd, description, verbosity=0, capture_output=True, text=True, check=False, **kwargs):
    """Safely run subprocess with consistent error handling and logging."""
    try:
        if verbosity > 1:
            print(f"    Running: {' '.join(cmd[:3])}{'...' if len(cmd) > 3 else ''}", file=sys.stderr)

        result = subprocess.run(cmd, capture_output=capture_output, text=text, check=check, **kwargs)

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if verbosity > 0:
                print(f"    {description} failed (exit {result.returncode}): {error_msg[:200]}", file=sys.stderr)

        return result

    except subprocess.TimeoutExpired as e:
        error_msg = f"{description} timed out after {e.timeout} seconds"
        if verbosity > 0:
            print(f"    {error_msg}", file=sys.stderr)
        # Return a mock result object for consistency
        return type('MockResult', (), {
            'returncode': -1, 
            'stdout': '', 
            'stderr': error_msg,
            'args': cmd
        })()

    except FileNotFoundError as e:
        error_msg = f"{description} failed - command not found: {cmd[0]}"
        if verbosity > 0:
            print(f"    {error_msg}", file=sys.stderr)
        return type('MockResult', (), {
            'returncode': -2, 
            'stdout': '', 
            'stderr': error_msg,
            'args': cmd
        })()

    except Exception as e:
        error_msg = f"{description} failed with unexpected error: {str(e)}"
        if verbosity > 0:
            print(f"    {error_msg}", file=sys.stderr)
        return type('MockResult', (), {
            'returncode': -3, 
            'stdout': '', 
            'stderr': error_msg,
            'args': cmd
        })()


def safe_file_read(file_path, description, verbosity=0, encoding='utf-8'):
    """Safely read file with error handling."""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read(), None
    except FileNotFoundError:
        error_msg = f"{description} - file not found: {file_path}"
        if verbosity > 0:
            print(f"    {error_msg}", file=sys.stderr)
        return "", error_msg
    except PermissionError:
        error_msg = f"{description} - permission denied: {file_path}"
        if verbosity > 0:
            print(f"    {error_msg}", file=sys.stderr)
        return "", error_msg
    except UnicodeDecodeError as e:
        error_msg = f"{description} - encoding error: {str(e)}"
        if verbosity > 0:
            print(f"    {error_msg}", file=sys.stderr)
        return "", error_msg
    except Exception as e:
        error_msg = f"{description} - unexpected error: {str(e)}"
        if verbosity > 0:
            print(f"    {error_msg}", file=sys.stderr)
        return "", error_msg

def setup_pdf_pages(pdf_path, tmpdir, verbosity):
    """Split PDF into individual pages and return total page count."""
    if verbosity >= 0:
        print("Splitting PDF into pages...", file=sys.stderr)

    # Get page count with improved error handling
    total_pages = 0
    page_count_cmd = ["pdfinfo", pdf_path]

    result = safe_subprocess_run(
        page_count_cmd, 
        "PDF info extraction", 
        verbosity=verbosity
    )

    if result.returncode == 0:
        try:
            pages_lines = [line for line in result.stdout.split('\n') if 'Pages:' in line]
            if pages_lines:
                total_pages = int(pages_lines[0].split()[-1])
                if verbosity > 1:
                    print(f"    PDF info reports {total_pages} pages", file=sys.stderr)
        except (ValueError, IndexError) as e:
            if verbosity > 0:
                print(f"    Warning: Could not parse page count from pdfinfo output: {e}", file=sys.stderr)
            total_pages = 0
    else:
        if verbosity > 0:
            print(f"    Warning: pdfinfo failed, will count pages after splitting", file=sys.stderr)

    # Split PDF into individual pages with improved error handling
    split_cmd = ["pdfseparate", pdf_path, os.path.join(tmpdir, "page_%04d.pdf")]

    result = safe_subprocess_run(
        split_cmd, 
        "PDF splitting", 
        verbosity=verbosity
    )

    if result.returncode != 0:
        error_msg = f"PDF splitting failed: {result.stderr}"
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)

    # Count actual page files if we couldn't get count before or as verification
    try:
        page_files = sorted([f for f in os.listdir(tmpdir) if f.startswith("page_") and f.endswith(".pdf")])
        actual_pages = len(page_files)

        if total_pages == 0:
            total_pages = actual_pages
            if verbosity > 1:
                print(f"    Counted {total_pages} page files", file=sys.stderr)
        elif total_pages != actual_pages:
            if verbosity > 0:
                print(f"    Warning: pdfinfo reported {total_pages} pages but found {actual_pages} files, using {actual_pages}", file=sys.stderr)
            total_pages = actual_pages

    except OSError as e:
        error_msg = f"Could not list page files: {e}"
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)

    if total_pages == 0:
        print("Error: No pages found in PDF", file=sys.stderr)
        sys.exit(1)

    if verbosity >= 0:
        print(f"Found {total_pages} pages", file=sys.stderr)

    return total_pages


def run_parallel_ocr(tmpdir, total_pages, max_workers, config, verbosity):
    """Run OCR on all pages in parallel with SCB names dictionary and personnummer patterns."""
    # Determine number of workers
    if max_workers is None:
        num_workers = get_optimal_workers(total_pages, verbosity)
    else:
        num_workers = min(max_workers, total_pages)
        if verbosity > 0:
            print(f"  Using user-specified {num_workers} parallel OCR workers", file=sys.stderr)

    # Create SCB names dictionary if enabled
    if config.use_scb_names and (config.first_names_set or config.last_names_set):
        config.scb_names_dict_path = create_scb_names_dictionary(
            config.first_names_set, 
            config.last_names_set, 
            verbosity
        )
        if config.scb_names_dict_path and verbosity >= 0:
            print(f"  Using SCB names dictionary for enhanced OCR recognition", file=sys.stderr)

    # Create personnummer patterns file if enabled
    if config.use_pnr_patterns:
        config.pnr_patterns_path = create_personnummer_patterns_file(verbosity)
        if config.pnr_patterns_path and verbosity >= 0:
            print(f"  Using personnummer pattern hints for enhanced OCR recognition", file=sys.stderr)

    if verbosity >= 0:
        print("Running OCR on pages...", file=sys.stderr)

    # Prepare arguments for parallel processing
    ocr_args = []
    for i in range(1, total_pages + 1):
        page_file = os.path.join(tmpdir, f"page_{i:04d}.pdf")
        ocr_args.append((page_file, i, tmpdir, config))

    # Initialize progress tracker
    progress = ProgressTracker(total_pages, verbosity)

    # Run OCR in parallel
    page_texts = {}
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(ocr_single_page, args): args[1] for args in ocr_args}

        for future in as_completed(futures):
            page_num, text, error = future.result()
            page_texts[page_num] = text

            if error:
                errors.append((page_num, error))

            progress.update(page_num)

    # Clean up temporary files
    if config.scb_names_dict_path and os.path.exists(config.scb_names_dict_path):
        try:
            os.remove(config.scb_names_dict_path)
            if verbosity > 1:
                print(f"  Cleaned up SCB names dictionary file", file=sys.stderr)
        except OSError:
            pass  # Not critical if cleanup fails

    if config.pnr_patterns_path and os.path.exists(config.pnr_patterns_path):
        try:
            os.remove(config.pnr_patterns_path)
            if verbosity > 1:
                print(f"  Cleaned up personnummer patterns file", file=sys.stderr)
        except OSError:
            pass  # Not critical if cleanup fails

    # Report any errors after progress is complete
    if errors and verbosity >= 0:
        print("\nOCR Errors:", file=sys.stderr)
        for page_num, error in errors:
            print(f"  Page {page_num}: {error}", file=sys.stderr)

    return page_texts


def extract_all_data(page_texts, config, verbosity, include_invalid):
    """Extract personnummer and names from all pages - FIRST PASS."""
    if verbosity >= 0:
        print("\nExtracting personnummer and names...", file=sys.stderr)

    all_data = []
    all_valid_entries = []  # For autocorrection lookup
    early_rejected_count = 0  # Not tracked here anymore (happens in find_all_personnummer_on_page)
    late_rejected_count = 0
    samordningsnummer_count = 0
    tf_number_count = 0
    last_captured_pnr = None  # Track last captured personnummer for proximity selection

    for page_num in sorted(page_texts.keys()):
        page_text = page_texts[page_num]
        page_results = extract_personnummer_and_names(page_text, page_num, config, verbosity, include_invalid, last_captured_pnr)

        for result in page_results:
            # Count different types
            if 'TF' in result['personnummer']:
                tf_number_count += 1

            if not result['date_valid']:
                late_rejected_count += 1

            if result.get('is_samordningsnummer'):
                samordningsnummer_count += 1

            # Collect valid entries for autocorrection (must pass both date and Luhn, or be valid TF)
            is_valid_tf = 'TF' in result['personnummer'] and result['date_valid']
            is_valid_regular = result['luhn_valid'] and result['date_valid'] and 'TF' not in result['personnummer']

            if (is_valid_tf or is_valid_regular) and result['efternamn'] and result['fornamn']:
                all_valid_entries.append(result)

            # Update last captured personnummer if this one is valid
            if result['date_valid'] and (result['luhn_valid'] or 'TF' in result['personnummer']):
                last_captured_pnr = result['personnummer']

            all_data.append({
                'page': page_num,
                'personnummer': result['personnummer'],
                'efternamn': result['efternamn'],
                'fornamn': result['fornamn'],
                'luhn_valid': result['luhn_valid'] if 'TF' not in result['personnummer'] else result['date_valid'],  # For TF, validity = date validity
                'original_pnr': result['original_pnr'],
                'date_valid': result['date_valid'],
                'luhn_check_valid': result['luhn_check_valid'],
                'is_samordningsnummer': result.get('is_samordningsnummer', False),
                'all_valid_entries': all_valid_entries  # Pass for autocorrection
            })

    if late_rejected_count > 0 and verbosity >= 0:
        print(f"  Found {late_rejected_count} personnummer with invalid dates/format (will attempt autocorrection)", file=sys.stderr)

    if samordningsnummer_count > 0 and verbosity >= 0:
        print(f"  Found {samordningsnummer_count} samordningsnummer (coordination numbers with day 61-91)", file=sys.stderr)

    if tf_number_count > 0 and verbosity >= 0:
        print(f"  Found {tf_number_count} TF numbers", file=sys.stderr)

    return all_data


def find_autocorrection_candidate(invalid_pnr, efternamn, fornamn, valid_entries, verbosity):
    """
    Find a valid personnummer that differs by exactly one digit from the invalid one.
    Now with detailed debug output when verbosity > 2 (-vvv flag).
    """
    # Check condition 1: Both names must be present
    if not efternamn or not fornamn:
        if verbosity > 2:
            print(f"    Evaluating {invalid_pnr} for {efternamn}, {fornamn}:", file=sys.stderr)
            print(f"      CONDITION 1 FAILED - missing name(s)", file=sys.stderr)
        return None

    # Find all entries with matching names (case-insensitive)
    matching_entries = [
        entry for entry in valid_entries
        if entry['efternamn'].lower() == efternamn.lower()
        and entry['fornamn'].lower() == fornamn.lower()
    ]

    # Check condition 2: Must have at least one matching name entry
    if not matching_entries:
        if verbosity > 2:
            print(f"    Evaluating {invalid_pnr} for {efternamn}, {fornamn}:", file=sys.stderr)
            print(f"      CONDITION 1 PASSED - both names present", file=sys.stderr)
            print(f"      CONDITION 2 FAILED - no matching names found", file=sys.stderr)
        return None

    # Check each matching entry to find one that differs by exactly one digit
    for candidate_entry in matching_entries:
        candidate_pnr = candidate_entry['personnummer']

        if verbosity > 2:
            print(f"    Evaluating {invalid_pnr} for {efternamn}, {fornamn}:", file=sys.stderr)
            print(f"      CONDITION 1 PASSED - both names present", file=sys.stderr)
            print(f"      CONDITION 2 PASSED - found {len(matching_entries)} matching name(s)", file=sys.stderr)
            print(f"      Checking candidate: {candidate_pnr}", file=sys.stderr)

        # Check condition 3: Candidate must be from valid entries (always true here by design)
        if verbosity > 2:
            print(f"        CONDITION 3 PASSED - candidate is from valid entries list", file=sys.stderr)

        # Check condition 4: Exactly one digit difference
        if len(invalid_pnr) != len(candidate_pnr):
            if verbosity > 2:
                print(f"        CONDITION 4 FAILED - different lengths ({len(invalid_pnr)} vs {len(candidate_pnr)})", file=sys.stderr)
            continue

        differences = sum(1 for a, b in zip(invalid_pnr, candidate_pnr) if a != b)

        if differences == 1:
            if verbosity > 2:
                print(f"        CONDITION 4 PASSED - exactly 1 digit difference", file=sys.stderr)
                print(f"        ✓ ALL CONDITIONS MET - autocorrection found!", file=sys.stderr)
            return candidate_pnr
        else:
            if verbosity > 2:
                print(f"        CONDITION 4 FAILED - {differences} digit differences (need exactly 1)", file=sys.stderr)

    return None


def has_single_digit_difference(pnr1, pnr2):
    """Check if two personnummer have exactly one digit difference."""
    if len(pnr1) != len(pnr2):
        return False

    # Remove hyphens for comparison
    clean_pnr1 = pnr1.replace('-', '')
    clean_pnr2 = pnr2.replace('-', '')

    if len(clean_pnr1) != len(clean_pnr2):
        return False

    differences = sum(1 for c1, c2 in zip(clean_pnr1, clean_pnr2) if c1 != c2)
    return differences == 1


def apply_single_correction(entry, valid_entries, verbosity):
    """Apply autocorrection to a single entry if possible."""
    # Use original_pnr for comparison (personnummer might be cleared if invalid)
    pnr_to_check = entry.get('original_pnr', entry['personnummer'])

    if not pnr_to_check:
        if verbosity > 2:
            print(f"    Autocorrect: No personnummer to check for page {entry.get('page', '?')}", 
                  file=sys.stderr)
        return False

    correction_candidate = find_autocorrection_candidate(
        pnr_to_check,  # Use original_pnr instead of potentially empty personnummer
        entry['efternamn'], 
        entry['fornamn'],
        valid_entries,
        verbosity  # Pass verbosity to enable debug output
    )

    if correction_candidate:
        original_pnr = pnr_to_check
        entry['personnummer'] = correction_candidate
        entry['luhn_valid'] = True
        entry['was_corrected'] = True
        entry['original_pnr'] = original_pnr  # Keep the actual original

        if verbosity > 1:
            print(f"    CORRECTED: {original_pnr} → {correction_candidate} for {entry['efternamn']}, {entry['fornamn']}", 
                  file=sys.stderr)

        return True

    return False


def apply_autocorrection(raw_data, verbosity):
    """Apply autocorrection to invalid personnummer with simplified logic."""
    # Extract all valid entries for lookup
    all_valid_entries = raw_data[0]['all_valid_entries'] if raw_data else []

    # Remove all_valid_entries from data items (cleanup)
    for entry in raw_data:
        entry.pop('all_valid_entries', None)

    # Find entries that need autocorrection - check using original_pnr
    invalid_entries = [
        entry for entry in raw_data 
        if not entry['luhn_valid'] 
        and 'TF' not in entry.get('original_pnr', entry['personnummer'])
        and entry.get('original_pnr', entry['personnummer'])  # Must have a personnummer to correct
    ]

    if not invalid_entries:
        if verbosity > 0:
            print("\nAutocorrection: All found personnummer are already valid - no autocorrection needed", file=sys.stderr)
        return raw_data

    # Apply autocorrection
    corrections_made = 0

    if verbosity >= 0:
        print(f"\nApplying autocorrection for {len(invalid_entries)} invalid personnummer...", file=sys.stderr)
        print("  Requirements: (1) Both names present, (2) Exact name match exists, (3) That match is valid, (4) Exactly 1 digit difference", file=sys.stderr)

    for entry in invalid_entries:
        if apply_single_correction(entry, all_valid_entries, verbosity):
            corrections_made += 1

            # Add corrected entry to valid entries for future corrections
            all_valid_entries.append({
                'personnummer': entry['personnummer'],
                'efternamn': entry['efternamn'],
                'fornamn': entry['fornamn'],
                'luhn_valid': True
            })

    # Report results
    if corrections_made > 0 and verbosity >= 0:
        print(f"  SUCCESS: Autocorrected {corrections_made}/{len(invalid_entries)} invalid personnummer", file=sys.stderr)
    elif verbosity >= 0:
        print(f"  No autocorrections applied - none of the {len(invalid_entries)} invalid personnummer met all 4 requirements", file=sys.stderr)

    return raw_data

def get_autocorrection_stats(raw_data):
    """Get statistics about autocorrection candidates and results."""
    total_invalid = sum(1 for entry in raw_data if not entry['luhn_valid'] and 'TF' not in entry['personnummer'])
    total_corrected = sum(1 for entry in raw_data if entry.get('was_corrected', False))

    return {
        'total_invalid': total_invalid,
        'total_corrected': total_corrected,
        'correction_rate': total_corrected / total_invalid if total_invalid > 0 else 0
    }


def validate_autocorrection_requirements(invalid_pnr, efternamn, fornamn):
    """Check if an entry meets basic autocorrection requirements."""
    # Requirement 1: Both names must be present
    if not efternamn or not fornamn:
        return False, "missing names"

    # Skip TF numbers
    if 'TF' in invalid_pnr:
        return False, "TF number"

    # Must be proper length
    clean_pnr = invalid_pnr.replace('-', '')
    if len(clean_pnr) != 10:
        return False, "wrong length"

    return True, "ok"


def apply_inheritance_and_resolve_names(
    raw_data: List[Dict], 
    total_pages: int, 
    inherit: bool,
    config: Config, 
    verbosity: int, 
    include_invalid: bool,
    disable_hole_filling: bool = False  # New parameter
) -> List[Dict[str, Any]]:
    """Apply inheritance (with hole-filling) and resolve name variants to produce final data."""
    # Process invalid personnummer first
    invalid_pnr_count = 0
    for entry in raw_data:
        if not entry['luhn_valid']:
            invalid_pnr_count += 1
            if not include_invalid:
                if verbosity >= 0:
                    print(f"  Page {entry['page']}: {entry['personnummer']} still invalid after autocorrection attempt - "
                          f"personnummer field will be EMPTY", file=sys.stderr)
                entry['personnummer'] = None

    # Group entries by page
    current_page_entries = defaultdict(list)
    for entry in raw_data:
        current_page_entries[entry['page']].append(entry)

    # Step 1: Process inheritance with hole-filling
    final_data, pnr_to_names, pages_with_pnr, pages_without_pnr = _process_inheritance_with_hole_filling(
        current_page_entries, total_pages, inherit, disable_hole_filling, verbosity
    )

    # Log inheritance statistics
    if verbosity >= 0:
        print(f"  Found personnummer on {pages_with_pnr} pages", file=sys.stderr)
        hole_filled_count = sum(1 for e in final_data if e.get('hole_filled', False))
        if hole_filled_count > 0:
            print(f"  Hole-filled {hole_filled_count} pages with identical surrounding personnummer", file=sys.stderr)
        if pages_without_pnr > 0:
            if inherit:
                print(f"  {pages_without_pnr} pages inherited personnummer or had empty fields", file=sys.stderr)
            else:
                print(f"  {pages_without_pnr} pages without personnummer will have empty field", file=sys.stderr)
        if invalid_pnr_count > 0:
            if include_invalid:
                print(f"  WARNING: {invalid_pnr_count} personnummer still invalid after autocorrection", file=sys.stderr)
            else:
                print(f"  {invalid_pnr_count} invalid personnummer (after autocorrection) will have empty field", 
                      file=sys.stderr)

    # Step 2: Resolve name variants
    pnr_final_names = _resolve_name_variants(pnr_to_names, config, verbosity)

    # Step 3: Apply final names to all entries
    updates_made, names_added = _apply_final_names_to_entries(final_data, pnr_final_names, verbosity)

    # Log name application statistics
    if verbosity > 0:
        if names_added > 0:
            print(f"  Added {names_added} missing names", file=sys.stderr)
        if updates_made > 0:
            print(f"  Updated {updates_made} name variants to selected version", file=sys.stderr)

    # Final summary
    if verbosity >= 0:
        unique_pnr = len(set(entry['personnummer'] for entry in final_data if entry['personnummer']))
        print(f"\nSummary: {unique_pnr} unique personnummer across {total_pages} pages", file=sys.stderr)

        corrections_made = sum(1 for e in final_data if e.get('was_corrected', False))
        if corrections_made > 0:
            print(f"  AUTOCORRECTED {corrections_made} single-digit OCR errors", file=sys.stderr)

        unique_invalid = len(set(e['personnummer'] for e in final_data if e['personnummer'] and not e['luhn_valid']))
        if unique_invalid > 0 and include_invalid:
            print(f"  INCLUDING {unique_invalid} personnummer with invalid Luhn check", file=sys.stderr)

        empty_pnr_count = sum(1 for e in final_data if not e['personnummer'])
        if empty_pnr_count > 0:
            print(f"  {empty_pnr_count} rows will have empty personnummer field", file=sys.stderr)

    return final_data

def main():
    parser = argparse.ArgumentParser(
        description='OCR PDF and extract Swedish personal numbers',
        epilog='Performance tip: Set OMP_THREAD_LIMIT=1 before running for better parallelization')
    parser.add_argument('pdf_file', help='Input PDF file')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Verbose output (show processing details)')
    parser.add_argument('-vv', '--very-verbose', action='store_true',
                        help='Very verbose output (show all found personnummer)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode (only output CSV)')
    parser.add_argument('-j', '--jobs', type=int, metavar='N',
                        help='Number of parallel OCR jobs (default: auto-detect based on CPU cores)')
    parser.add_argument('--include-invalid', action='store_true',
                        help='Include personnummer that fail Luhn check (default: leave personnummer field empty)')
    parser.add_argument('--no-inheritance', action='store_true',
                        help='Disable inheritance of personnummer from previous pages')
    parser.add_argument('--no-autocorrect', action='store_true',
                        help='Disable automatic correction of single-digit OCR errors')
    parser.add_argument('--no-scb-names', action='store_true',
                        help='Disable using SCB Swedish name lists for OCR enhancement and validation')
    parser.add_argument('--no-pnr-patterns', action='store_true',
                        help='Disable personnummer pattern hints for OCR (may reduce ID number recognition)')
    parser.add_argument('--no-hole-filling', action='store_true',  # NEW
                        help='Disable hole-filling when pages are surrounded by identical personnummer')

    args = parser.parse_args()

    # Set verbosity level
    if args.quiet:
        verbosity = -1
    elif args.very_verbose:
        verbosity = 2
    elif args.verbose:
        verbosity = 1
    else:
        verbosity = 0

    # Check dependencies
    for cmd in ['pdfseparate', 'pdfinfo', 'ocrmypdf']:
        if subprocess.run(['which', cmd], capture_output=True).returncode != 0:
            print(f"Error: {cmd} not found. Please install it first.", file=sys.stderr)
            sys.exit(1)

    # Check if file exists
    if not os.path.exists(args.pdf_file):
        print(f"Error: File {args.pdf_file} not found", file=sys.stderr)
        sys.exit(1)

    # Process the PDF
    data = process_pdf(args.pdf_file, verbosity, args.jobs, args.include_invalid, 
                      not args.no_inheritance, not args.no_autocorrect, 
                      not args.no_scb_names, not args.no_pnr_patterns,
                      args.no_hole_filling)  # NEW parameter

    # Output CSV
    csv_writer = csv.writer(sys.stdout)

    # Write header
    csv_writer.writerow(['filename', 'page', 'personnummer', 'efternamn', 'fornamn'])

    # Write data
    filename = os.path.basename(args.pdf_file)
    for entry in data:
        formatted_pnr = format_personnummer(entry['personnummer'])
        csv_writer.writerow([
            filename,
            entry['page'],
            formatted_pnr,
            entry['efternamn'],
            entry['fornamn']
        ])


if __name__ == '__main__':
    main()

