#!/usr/bin/env python3
"""
Scan CSV files in this folder and convert columns named 'Weight' that appear to be fractional
(e.g., 0.1234 summing to 1) into percentage values and update the header to 'Weight (%)'.
Saves changes in-place and prints a short diff/preview for each modified file.

Usage: run from the portfolio_exports directory or provide --dir.
"""
import csv
import os
import sys
from decimal import Decimal

TARGET_DIR = os.path.dirname(os.path.abspath(__file__))

def is_fractional_column(values):
    # try to parse numeric values and check if max <= 1.01 (allow small rounding)
    nums = []
    for v in values:
        if v is None or v.strip() == "":
            continue
        try:
            n = float(v)
        except Exception:
            return False
        nums.append(n)
    if not nums:
        return False
    return max(nums) <= 1.01


def convert_file(path):
    changed = False
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return False, None
    header = rows[0]
    if 'Weight (%)' in header:
        return False, None
    if 'Weight' not in header:
        return False, None
    widx = header.index('Weight')
    # extract column values
    col = [row[widx] if len(row) > widx else '' for row in rows[1:]]
    if not is_fractional_column(col):
        return False, None
    # convert values
    new_rows = [list(r) for r in rows]
    new_rows[0][widx] = 'Weight (%)'
    for i in range(1, len(new_rows)):
        r = new_rows[i]
        if len(r) <= widx:
            # pad
            r += [''] * (widx - len(r) + 1)
        val = r[widx].strip()
        if val == '':
            continue
        try:
            fval = Decimal(val)
        except Exception:
            continue
        p = fval * Decimal(100)
        # normalize string: remove trailing zeros but keep at least 4 decimal places if needed
        s = format(p.normalize(), 'f')
        # if the number is an integer, keep one decimal place .0
        if '.' not in s:
            s = s + '.0'
        r[widx] = s
        new_rows[i] = r
    # write back
    bak = path + '.bak'
    os.replace(path, bak)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)
    return True, bak


def main():
    files = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith('.csv')]
    modified = []
    for fn in files:
        if fn.startswith('orig_'):
            continue
        if os.path.isdir(fn):
            continue
        path = os.path.join(TARGET_DIR, fn)
        ok, bak = convert_file(path)
        if ok:
            modified.append((fn, bak))
            print(f"Converted weights to percent in: {fn} (backup: {os.path.basename(bak)})")
    if not modified:
        print("No files modified.")

if __name__ == '__main__':
    main()
