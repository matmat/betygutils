#!/usr/bin/env python3

import argparse
import csv
import os
import sys
import warnings
import subprocess
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import date
import re

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        print("Error: Please install pypdf or PyPDF2: pip install pypdf", file=sys.stderr)
        sys.exit(1)

try:
    from unidecode import unidecode
except ImportError:
    print("Error: Please install unidecode: pip install unidecode", file=sys.stderr)
    sys.exit(1)

try:
    from jinja2 import Template
except ImportError:
    print("Error: Please install jinja2: pip install jinja2", file=sys.stderr)
    sys.exit(1)


def get_pdf2archive_command():
    """Get the pdf2archive command, checking environment variable and PATH."""
    # Check environment variable first
    env_path = os.environ.get('PDF2ARCHIVE_PATH')
    if env_path:
        pdf2archive_path = Path(env_path)
        if pdf2archive_path.exists() and pdf2archive_path.is_file():
            return str(pdf2archive_path)
        else:
            warnings.warn(f"PDF2ARCHIVE_PATH points to non-existent file: {env_path}")

    # Default to 'pdf2archive' (will search in PATH)
    return 'pdf2archive'


def check_pdf2archive(pdf2archive_cmd=None):
    """Check if pdf2archive is available."""
    if pdf2archive_cmd is None:
        pdf2archive_cmd = get_pdf2archive_command()

    try:
        result = subprocess.run([pdf2archive_cmd, '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def convert_to_pdfa_with_pdf2archive(input_path, output_path, pdf2archive_cmd=None, 
                                     metadata=None, quality=None, clean_metadata=True, 
                                     verbose=False):
    """Convert PDF to PDF/A-1B using external pdf2archive tool."""

    if pdf2archive_cmd is None:
        pdf2archive_cmd = get_pdf2archive_command()

    # Build pdf2archive command
    cmd = [pdf2archive_cmd]

    # Add quality option if specified
    if quality and quality in ['high', 'medium', 'low']:
        cmd.append(f'--quality={quality}')

    # Handle metadata
    if clean_metadata and not metadata:
        # Clean all metadata by default when no metadata is provided
        cmd.append('--cleanmetadata')
    elif metadata:
        # If metadata is provided, add it (don't use --cleanmetadata)
        if 'title' in metadata and metadata['title']:
            cmd.append(f'--title={metadata["title"]}')
        if 'author' in metadata and metadata['author']:
            cmd.append(f'--author={metadata["author"]}')
        if 'subject' in metadata and metadata['subject']:
            cmd.append(f'--subject={metadata["subject"]}')
        if 'keywords' in metadata and metadata['keywords']:
            # Join keywords with commas if it's a list
            if isinstance(metadata['keywords'], list):
                keywords_str = ','.join(metadata['keywords'])
            else:
                keywords_str = str(metadata['keywords'])
            cmd.append(f'--keywords={keywords_str}')
    elif not clean_metadata:
        # User explicitly wants to preserve existing metadata
        # Don't add --cleanmetadata flag
        pass

    # Add input and output paths
    cmd.append(str(input_path))
    cmd.append(str(output_path))

    if verbose:
        print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            if verbose:
                print(f"  PDF/A conversion successful")
            return True
        else:
            if verbose:
                print(f"  PDF/A conversion failed: {result.stderr}")
            return False

    except Exception as e:
        if verbose:
            print(f"  PDF/A conversion error: {e}")
        return False


def clean_filename(text):
    """Clean text for use in filenames using unidecode and replacing spaces with underscores."""
    if not text:
        return ""
    cleaned = unidecode(str(text))
    cleaned = re.sub(r'[^\w\s-]', '', cleaned)  # Remove special chars except spaces and hyphens
    cleaned = re.sub(r'\s+', '_', cleaned)      # Replace spaces with underscores
    return cleaned.strip('_')


def parse_field_spec(field_spec, has_header, headers=None):
    """Parse field specification (name if has_header, number if not)."""
    if has_header:
        if headers is None:
            raise ValueError("Headers required when has_header=True")
        if field_spec not in headers:
            raise ValueError(f"Field '{field_spec}' not found in headers: {headers}")
        return headers.index(field_spec)
    else:
        try:
            field_num = int(field_spec)
            if field_num < 1:
                raise ValueError("Field numbers must start from 1")
            return field_num - 1  # Convert to 0-based index
        except ValueError:
            raise ValueError(f"Invalid field number: {field_spec}")


def extract_pdf_pages(pdf_path, page_numbers):
    """Extract specific pages from a PDF and return a PdfWriter object."""
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        for page_num in page_numbers:
            if page_num < 1 or page_num > len(reader.pages):
                warnings.warn(f"Page {page_num} does not exist in {pdf_path} (has {len(reader.pages)} pages)")
                continue

            writer.add_page(reader.pages[page_num - 1])  # Convert to 0-based

        return writer
    except Exception as e:
        warnings.warn(f"Error reading PDF {pdf_path}: {e}")
        return None


def combine_pdf_pages(pdf_pages, output_path, remove_metadata=True, verbose=False):
    """Combine multiple PDF pages into a single PDF file using pypdf."""

    combined_writer = PdfWriter()

    # Remove metadata from the writer if requested
    if remove_metadata:
        combined_writer.add_metadata({})  # Empty metadata

    for pdf_path, page_num in pdf_pages:
        try:
            reader = PdfReader(pdf_path)
            if page_num <= len(reader.pages):
                combined_writer.add_page(reader.pages[page_num - 1])
                if verbose:
                    print(f"  Added page {page_num} from {pdf_path}")
            else:
                warnings.warn(f"Page {page_num} doesn't exist in {pdf_path}")
        except Exception as e:
            warnings.warn(f"Error reading {pdf_path}: {e}")
            continue

    if not combined_writer.pages:
        return False

    try:
        with open(output_path, 'wb') as f:
            combined_writer.write(f)
        return True
    except Exception as e:
        warnings.warn(f"Error writing combined PDF: {e}")
        return False


def get_default_template_output(template_path):
    """Generate default template output name from template path."""
    template_name = Path(template_path).stem  # Get filename without extension
    return f"{template_name}.xml"


def extract_metadata_fields(row, headers, metadata_fields):
    """Extract metadata fields from a CSV row."""
    metadata = {}

    if not metadata_fields:
        return metadata

    for field in metadata_fields.split(','):
        field = field.strip()
        try:
            if field in headers:
                idx = headers.index(field)
                if idx < len(row):
                    metadata[field] = row[idx]
        except (ValueError, IndexError):
            pass

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Split and combine PDF pages based on CSV metadata with optional PDF/A-1B output",
        epilog="""
JINJA2 TEMPLATE FORMATTING:
  The Jinja2 template file should use field names as variables with double curly braces.

  CSV field examples:
    - With header: {{ filename }}, {{ author }}, {{ title }}, {{ date }}
    - Without header: {{ field_1 }}, {{ field_2 }}, {{ field_3 }}, {{ field_4 }}

  SPECIAL ALWAYS-AVAILABLE FIELDS:
    These fields are automatically populated by the script and can always be used:

    - {{ processing_date }}   : Current date in ISO format (e.g., "2025-01-22")
    - {{ pdf_filename }}      : Generated PDF filename (e.g., "category_author_2023.pdf")

  Template example:
    <metadata>
      <title>{{ title }}</title>
      <author>{{ author }}</author>
      <source_file>{{ filename }}</source_file>
      <source_page>{{ pagenumber }}</source_page>
      <output_pdf>{{ pdf_filename }}</output_pdf>
      <processed_date>{{ processing_date }}</processed_date>
    </metadata>

EXAMPLES:
  # Basic usage with automatic metadata generation
  python script.py data.csv --key-field category --output-dir ./archive --template info.j2

  # With separate input directory for source PDFs
  python script.py data.csv --key-field category --input-dir ./source_pdfs \\
                   --output-dir ./archive --template info.j2

  # With additional naming fields for PDF filenames  
  python script.py data.csv --key-field category --input-dir /path/to/pdfs \\
                   --output-dir ./archive --template metadata.j2 \\
                   --template-output info.xml --naming-fields author,year

  # With PDF/A conversion and metadata extraction
  python script.py data.csv --key-field category --output-dir ./archive \\
                   --template info.j2 --pdfa --pdfa-quality high \\
                   --pdfa-metadata title,author,subject

  # Using custom pdf2archive path
  python script.py data.csv --key-field category --output-dir ./archive \\
                   --template info.j2 --pdfa --pdf2archive-path /usr/local/bin/pdf2archive

  # Or set environment variable:
  export PDF2ARCHIVE_PATH=/usr/local/bin/pdf2archive
  python script.py data.csv --key-field category --output-dir ./archive \\
                   --template info.j2 --pdfa

PDF METADATA:
  By default, all metadata is removed from output PDFs for privacy.
  Use --preserve-metadata to keep existing metadata.
  Use --pdfa-metadata to add specific metadata from CSV fields.

PDF/A-1B COMPLIANCE:
  Uses external pdf2archive tool for PDF/A conversion.
  Install: https://github.com/matteosecli/pdf2archive
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("csv_file", help="CSV file path (use '-' for stdin)")
    parser.add_argument("--key-field", required=True, 
                      help="Field name (with header) or number (without header) to group by")
    parser.add_argument("--input-dir", default=".", 
                      help="Directory containing source PDF files (default: current directory)")
    parser.add_argument("--output-dir", required=True, 
                      help="Root directory for output")
    parser.add_argument("--template", required=True, 
                      help="Jinja2 template file path")
    parser.add_argument("--template-output", 
                      help="Name of output file from template in each key directory (default: <template_basename>.xml)")
    parser.add_argument("--naming-fields", 
                      help="Comma-separated list of additional fields for PDF filenames")
    parser.add_argument("--no-header", action="store_true", 
                      help="CSV file has no header row")
    parser.add_argument("--inherit-keys", action="store_true", 
                      help="Empty keys inherit from nearest valid key above")

    # PDF/A conversion options
    parser.add_argument("--pdfa", action="store_true", 
                      help="Convert output PDFs to PDF/A-1B format using pdf2archive")
    parser.add_argument("--pdf2archive-path",
                      help="Path to pdf2archive executable (default: search in PATH, or use PDF2ARCHIVE_PATH env var)")
    parser.add_argument("--pdfa-quality", choices=['high', 'medium', 'low'],
                      help="PDF/A output quality when using pdf2archive")
    parser.add_argument("--pdfa-metadata", 
                      help="Comma-separated CSV fields to use as PDF/A metadata (title,author,subject,keywords)")

    # Metadata handling options
    parser.add_argument("--preserve-metadata", action="store_true",
                      help="Preserve existing PDF metadata (default: remove all metadata)")

    parser.add_argument("--verbose", action="store_true", 
                      help="Show detailed conversion progress")

    args = parser.parse_args()

    # Set default template output name if not provided
    if not args.template_output:
        args.template_output = get_default_template_output(args.template)

    # Convert input directory to Path object
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Get current date in ISO format for templates
    current_date = date.today().isoformat()

    # Determine pdf2archive command
    pdf2archive_cmd = args.pdf2archive_path if args.pdf2archive_path else get_pdf2archive_command()

    # Check for pdf2archive if PDF/A conversion is requested
    if args.pdfa and not check_pdf2archive(pdf2archive_cmd):
        print(f"Error: pdf2archive is required for PDF/A-1B conversion but not found.", file=sys.stderr)
        if args.pdf2archive_path:
            print(f"Specified path not found: {args.pdf2archive_path}", file=sys.stderr)
        elif os.environ.get('PDF2ARCHIVE_PATH'):
            print(f"PDF2ARCHIVE_PATH not valid: {os.environ.get('PDF2ARCHIVE_PATH')}", file=sys.stderr)
        else:
            print("Install from: https://github.com/matteosecli/pdf2archive", file=sys.stderr)
            print("Or specify path with --pdf2archive-path or PDF2ARCHIVE_PATH environment variable", file=sys.stderr)
        print("Or omit --pdfa to create regular PDFs", file=sys.stderr)
        sys.exit(1)

    # Read CSV file
    if args.csv_file == '-':
        csv_file = sys.stdin
    else:
        try:
            csv_file = open(args.csv_file, 'r', newline='', encoding='utf-8')
        except IOError as e:
            print(f"Error opening CSV file: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        # Read CSV data
        reader = csv.reader(csv_file)
        rows = list(reader)

        if not rows:
            print("Error: CSV file is empty", file=sys.stderr)
            sys.exit(1)

        # Handle headers
        has_header = not args.no_header
        if has_header:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            headers = [f"field_{i+1}" for i in range(len(rows[0]))]
            data_rows = rows

        if len(data_rows) == 0:
            print("Error: No data rows found", file=sys.stderr)
            sys.exit(1)

        # Parse field specifications
        try:
            key_field_idx = parse_field_spec(args.key_field, has_header, headers)

            naming_field_indices = []
            if args.naming_fields:
                for field in args.naming_fields.split(','):
                    field = field.strip()
                    naming_field_indices.append(parse_field_spec(field, has_header, headers))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Process rows with key inheritance if enabled
        processed_rows = []
        last_valid_key = None

        for row in data_rows:
            if len(row) < 2:
                warnings.warn(f"Skipping row with insufficient columns: {row}")
                continue

            # Handle key inheritance
            current_key = row[key_field_idx] if key_field_idx < len(row) else ""

            if args.inherit_keys and not current_key.strip():
                if last_valid_key is not None:
                    row = row.copy()  # Don't modify original
                    if len(row) <= key_field_idx:
                        row.extend([''] * (key_field_idx - len(row) + 1))
                    row[key_field_idx] = last_valid_key
                    current_key = last_valid_key
            elif current_key.strip():
                last_valid_key = current_key.strip()

            if not current_key.strip():
                warnings.warn(f"Skipping row with empty key: {row}")
                continue

            processed_rows.append(row)

        # Group rows by key (stable sort)
        key_groups = defaultdict(list)
        for row in processed_rows:
            key = row[key_field_idx].strip()
            key_groups[key].append(row)

        # Sort keys, but maintain order within each group (stable)
        sorted_keys = sorted(key_groups.keys())

        # Load Jinja2 template
        try:
            with open(args.template, 'r', encoding='utf-8') as f:
                template_content = f.read()
            template = Template(template_content)
        except IOError as e:
            print(f"Error reading template file: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing template: {e}", file=sys.stderr)
            sys.exit(1)

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each key group
        for key in sorted_keys:
            rows_for_key = key_groups[key]

            # Create key subdirectory
            key_dir = output_dir / clean_filename(key)
            key_dir.mkdir(parents=True, exist_ok=True)

            if args.verbose:
                print(f"\nProcessing key: {key}")

            # Collect PDF pages
            pdf_pages = []
            template_data = {}
            pdf_metadata = {}

            for row in rows_for_key:
                if len(row) < 2:
                    continue

                filename = row[0]
                try:
                    page_num = int(row[1])
                except (ValueError, IndexError):
                    warnings.warn(f"Invalid page number in row: {row}")
                    continue

                # Build full path to PDF file using input directory
                pdf_path = input_dir / filename

                # Check if PDF file exists
                if not pdf_path.exists():
                    warnings.warn(f"PDF file not found: {pdf_path}")
                    continue

                pdf_pages.append((str(pdf_path), page_num))

                # Collect template data (use first row's data)
                if not template_data:
                    for i, header in enumerate(headers):
                        template_data[header] = row[i] if i < len(row) else ""

                    # Extract metadata for PDF/A if requested
                    if args.pdfa and args.pdfa_metadata:
                        pdf_metadata = extract_metadata_fields(row, headers, args.pdfa_metadata)

            # Generate PDF filename
            pdf_filename = clean_filename(key)
            if naming_field_indices:
                additional_parts = []
                first_row = rows_for_key[0] if rows_for_key else []
                for idx in naming_field_indices:
                    if idx < len(first_row):
                        additional_parts.append(clean_filename(first_row[idx]))
                if additional_parts:
                    pdf_filename += "_" + "_".join(additional_parts)
            pdf_filename += ".pdf"

            # Add automatic fields to template data
            template_data['processing_date'] = current_date
            template_data['pdf_filename'] = pdf_filename

            # Combine PDF pages
            if pdf_pages:
                pdf_output_path = key_dir / pdf_filename

                if args.pdfa:
                    # Use temporary file for intermediate PDF
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        temp_path = Path(temp_file.name)

                    try:
                        # First combine pages to temporary file
                        # Remove metadata unless user wants to preserve it
                        if combine_pdf_pages(pdf_pages, temp_path, 
                                           remove_metadata=not args.preserve_metadata, 
                                           verbose=args.verbose):
                            # Convert to PDF/A
                            # Clean metadata by default unless user provides metadata or wants to preserve
                            clean_meta = not (args.pdfa_metadata or args.preserve_metadata)

                            if convert_to_pdfa_with_pdf2archive(
                                temp_path, 
                                pdf_output_path,
                                pdf2archive_cmd=pdf2archive_cmd,
                                metadata=pdf_metadata if args.pdfa_metadata else None,
                                quality=args.pdfa_quality,
                                clean_metadata=clean_meta,
                                verbose=args.verbose
                            ):
                                metadata_info = ""
                                if args.pdfa_metadata:
                                    metadata_info = " (with metadata)"
                                elif not args.preserve_metadata:
                                    metadata_info = " (metadata cleaned)"
                                print(f"PDF/A-1B{metadata_info}: {pdf_output_path}")
                            else:
                                # Fall back to regular PDF if conversion fails
                                shutil.move(temp_path, pdf_output_path)
                                warnings.warn(f"PDF/A conversion failed, created regular PDF: {pdf_output_path}")
                        else:
                            warnings.warn(f"Failed to combine pages for {key}")

                    finally:
                        # Clean up temporary file
                        if temp_path.exists():
                            temp_path.unlink()

                else:
                    # Direct combination without PDF/A conversion
                    # Remove metadata unless user wants to preserve it
                    if combine_pdf_pages(pdf_pages, pdf_output_path, 
                                       remove_metadata=not args.preserve_metadata,
                                       verbose=args.verbose):
                        metadata_info = " (metadata removed)" if not args.preserve_metadata else ""
                        print(f"PDF created{metadata_info}: {pdf_output_path}")
                    else:
                        warnings.warn(f"Failed to create PDF for {key}")

            # Generate template file
            try:
                template_output = template.render(**template_data)
                template_path = key_dir / args.template_output
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(template_output)
                if not args.verbose:
                    print(f"Template: {template_path}")
                else:
                    print(f"  Template created: {template_path}")
            except Exception as e:
                warnings.warn(f"Error generating template file for key '{key}': {e}")

    finally:
        if args.csv_file != '-':
            csv_file.close()


if __name__ == "__main__":
    main()

