#!/usr/bin/env python3
"""
XLS/XLSX to CSV Converter for iShares ETF Holdings Files

This script converts iShares ETF holdings files (XML-based .xls format or .xlsx) 
to clean CSV files for further analysis.

Usage:
    python convert_xls_to_csv.py <input_file.xls|xlsx> [output_file.csv]
    python convert_xls_to_csv.py --all-etf-data   # Convert all files in ETF_Data folder
    
    If output_file is not specified, it will be named based on the input file.

Example:
    python convert_xls_to_csv.py STOXX-Europe-600-Oil--Gas-UCITS-ETF-DE_fund.xls
    python convert_xls_to_csv.py ETF_Data/OG_24.xlsx
"""

import re
import sys
import pandas as pd
from pathlib import Path


def parse_ishares_xls(file_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Parse an iShares XLS file (XML SpreadsheetML format) and extract holdings data.
    
    Args:
        file_path: Path to the .xls file
        
    Returns:
        Tuple of (holdings DataFrame, metadata dict)
    """
    # Read the file as binary and decode
    with open(file_path, 'rb') as f:
        content = f.read()
    
    text = content.decode('utf-8', errors='ignore')
    
    # Extract data using regex - get both String and Number types
    cell_pattern = r'<ss:Data ss:Type="(?:String|Number)">([^<]*)</ss:Data>'
    
    # Find all rows
    row_pattern = r'<ss:Row>(.*?)</ss:Row>'
    rows = re.findall(row_pattern, text, re.DOTALL)
    
    data = []
    for row in rows:
        cells = re.findall(cell_pattern, row)
        if cells:
            data.append(cells)
    
    # Extract metadata from first rows
    metadata = {}
    for row in data[:10]:
        if len(row) == 1:
            # Could be date or fund name
            if '-' in row[0] and len(row[0]) == 11:  # Date format like "01-Dec-2025"
                metadata['report_date'] = row[0]
            elif 'UCITS' in row[0] or 'ETF' in row[0]:
                metadata['fund_name'] = row[0]
        elif len(row) == 2:
            key, value = row[0], row[1]
            if 'Inception' in key:
                metadata['inception_date'] = value
            elif 'Holdings as of' in key:
                metadata['holdings_date'] = value
            elif 'Number of Securities' in key:
                metadata['num_securities'] = value
            elif 'Shares Outstanding' in key:
                metadata['shares_outstanding'] = value
    
    # Find header row (contains 'Issuer Ticker')
    header_idx = None
    for i, row in enumerate(data):
        if 'Issuer Ticker' in row and len(row) >= 6:
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError("Could not find header row with 'Issuer Ticker'")
    
    headers = data[header_idx]
    holdings_data = data[header_idx + 1:]
    
    # Filter to only rows that match the header structure (have Asset Class)
    clean_data = []
    for row in holdings_data:
        # Check if this is a valid holdings row (has Asset Class in position 3)
        if len(row) >= 4 and row[3] in ['Equity', 'Cash', 'Fixed Income', 'Derivative']:
            # Pad row to match header length if needed
            while len(row) < len(headers):
                row.append('')
            clean_data.append(row[:len(headers)])
    
    # Create DataFrame
    df = pd.DataFrame(clean_data, columns=headers)
    
    # Clean up HTML entities
    df['Name'] = df['Name'].str.replace('&amp;', '&', regex=False)
    
    # Convert numeric columns
    numeric_cols = ['Market Value', 'Weight (%)', 'Notional Value', 'Nominal', 'Price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, metadata


def parse_xlsx_etf(file_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Parse an iShares XLSX file (standard Excel format) and extract holdings data.
    
    Args:
        file_path: Path to the .xlsx file
        
    Returns:
        Tuple of (holdings DataFrame, metadata dict)
    """
    # Read the raw Excel file
    df_raw = pd.read_excel(file_path, header=None)
    
    # Extract metadata from first rows
    metadata = {}
    metadata['fund_code'] = str(df_raw.iloc[0, 0]) if pd.notna(df_raw.iloc[0, 0]) else ''
    metadata['fund_name'] = str(df_raw.iloc[0, 1]) if pd.notna(df_raw.iloc[0, 1]) else ''
    metadata['report_date'] = str(df_raw.iloc[1, 1]) if pd.notna(df_raw.iloc[1, 1]) else ''
    
    # Find header row (contains 'RIC' and 'Name')
    header_idx = None
    for i in range(min(10, len(df_raw))):
        row_values = df_raw.iloc[i].astype(str).tolist()
        if 'RIC' in row_values and 'Name' in row_values:
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError("Could not find header row with 'RIC' and 'Name'")
    
    # Get headers and clean them
    headers = df_raw.iloc[header_idx].tolist()
    headers = [str(h).strip() if pd.notna(h) else f'Col_{i}' for i, h in enumerate(headers)]
    
    # Get data rows
    df = df_raw.iloc[header_idx + 1:].copy()
    df.columns = headers
    
    # Remove rows where RIC is NaN or empty
    df = df[df['RIC'].notna() & (df['RIC'] != '')]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Convert Weight to numeric (it's stored as decimal, e.g., 0.141473 = 14.1473%)
    if 'Weight' in df.columns:
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
        # Convert to percentage if values are < 1
        if df['Weight'].max() < 1:
            df['Weight (%)'] = df['Weight'] * 100
        else:
            df['Weight (%)'] = df['Weight']
    
    # Convert other numeric columns
    for col in ['No. Shares', 'Change']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean up Name column
    if 'Name' in df.columns:
        df['Name'] = df['Name'].astype(str).str.replace('&amp;', '&', regex=False)
    
    return df, metadata


def convert_xls_to_csv(input_path: str, output_path: str = None, 
                       equity_only: bool = False, verbose: bool = True) -> str:
    """
    Convert an iShares XLS/XLSX file to CSV format.
    
    Args:
        input_path: Path to input .xls or .xlsx file
        output_path: Path for output .csv file (optional, auto-generated if not provided)
        equity_only: If True, only include equity holdings (exclude cash, derivatives)
        verbose: If True, print summary information
        
    Returns:
        Path to the output CSV file
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}.csv"
    
    # Parse based on file extension
    if input_file.suffix.lower() == '.xlsx':
        df, metadata = parse_xlsx_etf(input_path)
    else:
        df, metadata = parse_ishares_xls(input_path)
    
    # Filter to equity only if requested
    if equity_only:
        df = df[df['Asset Class'] == 'Equity'].copy()
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"=" * 60)
        print(f"iShares XLS to CSV Converter")
        print(f"=" * 60)
        print(f"\nInput file:  {input_path}")
        print(f"Output file: {output_path}")
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        print(f"\nHoldings Summary:")
        print(f"  Total rows: {len(df)}")
        if 'Asset Class' in df.columns:
            print(f"  By Asset Class:")
            for asset_class, count in df['Asset Class'].value_counts().items():
                print(f"    - {asset_class}: {count}")
        if 'Sector' in df.columns:
            print(f"  By Sector:")
            for sector, count in df['Sector'].value_counts().items():
                print(f"    - {sector}: {count}")
        if 'Weight (%)' in df.columns:
            total_weight = df['Weight (%)'].sum()
            print(f"  Total Weight: {total_weight:.2f}%")
        print(f"\nTop 5 Holdings:")
        if 'Weight (%)' in df.columns:
            top5 = df.nlargest(5, 'Weight (%)')
            for _, row in top5.iterrows():
                print(f"  {row['Name'][:40]:<40} {row['Weight (%)']:>8.2f}%")
        print(f"\n{'=' * 60}")
        print(f"Successfully saved to: {output_path}")
    
    return str(output_path)


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide an input file path or use --all-etf-data.")
        sys.exit(1)
    
    # Check for batch mode
    if sys.argv[1] == '--all-etf-data':
        etf_data_dir = Path(__file__).parent.parent / 'ETF_Data'
        output_dir = Path(__file__).parent
        
        if not etf_data_dir.exists():
            print(f"Error: ETF_Data directory not found at {etf_data_dir}")
            sys.exit(1)
        
        print(f"Converting all ETF files from: {etf_data_dir}")
        print(f"Output directory: {output_dir}\n")
        
        for xlsx_file in etf_data_dir.glob('*.xlsx'):
            try:
                output_file = output_dir / f"{xlsx_file.stem}.csv"
                convert_xls_to_csv(str(xlsx_file), str(output_file))
                print()
            except Exception as e:
                print(f"Error processing {xlsx_file.name}: {e}\n")
        
        print("Done!")
        sys.exit(0)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        convert_xls_to_csv(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
