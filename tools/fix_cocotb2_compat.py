#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fix cocotb 2.x compatibility issues in CVDP benchmark dataset files.

Transforms test harnesses written for cocotb 1.x to work with cocotb 2.x.
Creates a NEW modified dataset file (does not overwrite original).

Transformations Applied:
    | cocotb 1.x                              | cocotb 2.x                              |
    |-----------------------------------------|-----------------------------------------|
    | from cocotb.runner import               | from cocotb_tools.runner import         |
    | from cocotb.binary import BinaryValue   | from cocotb.types import LogicArray     |
    | from cocotb.result import TestFailure   | (removed - use AssertionError)          |
    | import cocotb.utils                     | from cocotb import sim_time_utils       |
    | dut.signal[i]                           | dut.signal.value[i]                     |
    | BinaryValue('Z')                        | 'Z'                                     |
    | BinaryValue(...)                        | LogicArray(...)                         |
    | @cocotb.coroutine + async def           | async def (decorator removed)           |
    | raise TestFailure(...)                  | raise AssertionError(...)               |
    | cocotb.utils.get_sim_time()             | sim_time_utils.get_sim_time()           |
    | f"{dut.sig.value:#x}"                   | f"{int(dut.sig.value):#x}"              |
    | dut.s_valid_o.value (boolean context)   | int(dut.s_valid_o.value)                |
"""

import json
import argparse
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# TRANSFORMATION RULES
# =============================================================================

# Import statement transformations (applied first)
IMPORT_TRANSFORMS = [
    # Runner module moved
    (r'from cocotb\.runner import', 'from cocotb_tools.runner import'),
    (r'import cocotb\.runner', 'import cocotb_tools.runner'),
    
    # BinaryValue -> LogicArray
    (r'from cocotb\.binary import BinaryValue', 'from cocotb.types import LogicArray'),
    (r'from cocotb\.binary import', 'from cocotb.types import'),
    (r'import cocotb\.binary', 'import cocotb.types'),
    
    # TestFailure removed - delete import
    (r'from cocotb\.result import TestFailure\n?', ''),
    (r'from cocotb\.result import.*,\s*TestFailure', 'from cocotb.result import'),
    
    # cocotb.utils -> sim_time_utils
    (r'import cocotb\.utils\n', 'from cocotb import sim_time_utils\n'),
    (r'from cocotb\.utils import', 'from cocotb.sim_time_utils import'),
]

# API usage transformations (applied after imports)
API_TRANSFORMS = [
    # Packed signal bit access with .value after subscript: dut.sig[i].value -> dut.sig.value[i]
    # This handles inout/wire signals that look like array access but are packed
    (r'(dut\.gpio)\[(\w+)\]\.value', r'\1.value[\2]'),
    (r'(dut\.s_axis_tvalid)\[(\w+)\]\.value', r'\1.value[\2]'),
    (r'(dut\.s_axis_tlast)\[(\w+)\]\.value', r'\1.value[\2]'),
    
    # LogicObject not subscriptable in 2.x: dut.sig[i] -> dut.sig.value[i]
    # Use negative lookahead (?!\.value) to NOT match when .value already follows (unpacked arrays)
    (r'(dut\.\w+)\[(\d+)\](?!\.value)', r'\1.value[\2]'),
    (r'(dut\.\w+)\[([a-zA-Z_]\w*)\](?!\.value)', r'\1.value[\2]'),
    
    # BinaryValue('Z') for hi-Z -> just 'Z' (must be before BinaryValue -> LogicArray)
    (r"BinaryValue\(['\"]Z['\"]\)", "'Z'"),
    
    # BinaryValue -> LogicArray class name
    (r'\bBinaryValue\b', 'LogicArray'),
    
    # Remove redundant .value[x].value -> .value[x]
    (r'\.value\[(\w+)\]\.value\b', r'.value[\1]'),
    
    # @cocotb.coroutine deprecated - remove when followed by async def
    (r'@cocotb\.coroutine\n(async def)', r'\1'),
    
    # TestFailure -> AssertionError
    (r'raise TestFailure\(', 'raise AssertionError('),
    (r'cocotb\.result\.TestFailure', 'AssertionError'),
    
    # cocotb.utils -> sim_time_utils
    (r'cocotb\.utils\.', 'sim_time_utils.'),
    
    # LogicArray format specifiers need int() wrapper
    (r'\{(dut\.\w+\.value)(:[\#0-9]*[xXbBoOdD][^}]*)\}', r'{int(\1)\2}'),
    (r'\{(dut\.\w+\.value\[\w+\])(:[\#0-9]*[xXbBoOdD][^}]*)\}', r'{int(\1)\2}'),
    
    # .is_resolvable doesn't exist on Logic (single bit from subscript)
    # LogicArray has it, Logic doesn't - use hasattr check for compatibility
    # (?<![.\w]) ensures we match full identifiers not preceded by . or other word chars
    # This avoids matching .value.is_resolvable (which works fine)
    (r'(?<![.\w])(\w+)\.is_resolvable', r"((\1.is_resolvable) if hasattr(\1, 'is_resolvable') else (str(\1) in ('0', '1')))"),
    
    # data_serializer specific: boolean context check with 3 chained signals
    # "if dut.s_valid_o.value and dut.s_ready_i.value and dut.tx_en_i.value:"
    # These Logic objects don't auto-convert to bool in cocotb 2.x
    (r'if (dut\.s_valid_o\.value) and (dut\.s_ready_i\.value) and (dut\.tx_en_i\.value):',
     r'if int(\1) and int(\2) and int(\3):'),
    
    # decode_firstbit specific: boolean context check for Out_Valid
    # "if dut.Out_Valid.value:" -> "if int(dut.Out_Valid.value):"
    (r'if (dut\.Out_Valid\.value):', r'if int(\1):'),
    
    # radix2_div specific: boolean context check for done signal
    # "if dut.done.value:" -> "if int(dut.done.value):"
    (r'if (dut\.done\.value):', r'if int(\1):'),
]


# =============================================================================
# TRANSFORMATION FUNCTIONS
# =============================================================================

def apply_transforms(content: str, transforms: List[Tuple[str, str]]) -> Tuple[str, List[str]]:
    """Apply regex transformations, return (modified_content, list_of_changes)."""
    changes = []
    for pattern, replacement in transforms:
        matches = re.findall(pattern, content)
        if matches:
            count = len(matches) if isinstance(matches[0], str) else len(matches)
            content = re.sub(pattern, replacement, content)
            # Shorter change description
            short_pattern = pattern[:40] + '...' if len(pattern) > 40 else pattern
            changes.append(f"{short_pattern} ({count}x)")
    return content, changes


def add_missing_imports(content: str) -> Tuple[str, List[str]]:
    """Add imports that are needed after API transformations."""
    changes = []
    
    # If sim_time_utils is used but not imported, add it
    if 'sim_time_utils.' in content:
        has_import = ('from cocotb import sim_time_utils' in content or 
                      'from cocotb.sim_time_utils import' in content)
        if not has_import:
            lines = content.split('\n')
            # Find last cocotb import line
            insert_idx = 0
            for i, line in enumerate(lines):
                if 'import cocotb' in line or 'from cocotb' in line:
                    insert_idx = i + 1
            
            if insert_idx == 0:
                # No cocotb imports, find end of imports section
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        insert_idx = i + 1
                    elif stripped and not stripped.startswith('#') and insert_idx > 0:
                        break
            
            lines.insert(insert_idx, 'from cocotb import sim_time_utils')
            content = '\n'.join(lines)
            changes.append("Added: from cocotb import sim_time_utils")
    
    return content, changes


def fix_python_file(content: str) -> Tuple[str, List[str]]:
    """Apply all cocotb 2.x fixes to a Python file."""
    all_changes = []
    
    # 1. Transform imports
    content, changes = apply_transforms(content, IMPORT_TRANSFORMS)
    all_changes.extend(changes)
    
    # 2. Transform API usage
    content, changes = apply_transforms(content, API_TRANSFORMS)
    all_changes.extend(changes)
    
    # 3. Add any missing imports
    content, changes = add_missing_imports(content)
    all_changes.extend(changes)
    
    return content, all_changes


def fix_harness(harness: Dict) -> Tuple[Dict, List[str]]:
    """Fix all Python files in a harness."""
    if 'files' not in harness:
        return harness, []
    
    all_changes = []
    files = harness['files'].copy()
    
    for filename, content in files.items():
        if isinstance(content, str) and filename.endswith('.py'):
            original = content
            content, changes = fix_python_file(content)
            if content != original:
                files[filename] = content
                if changes:
                    all_changes.append(f"  {filename}: {len(changes)} transforms")
    
    harness['files'] = files
    return harness, all_changes


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_dataset(input_path: str, output_path: str, dry_run: bool = False, 
                    modified_only: bool = False, verbose: bool = False) -> Dict:
    """Process a JSONL dataset file."""
    stats = {'total': 0, 'modified': 0, 'changes': [], 'errors': []}
    output_lines = []
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    if modified_only:
        print("Mode:   Modified problems only")
    print()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                if not modified_only:
                    output_lines.append('')
                continue
            
            try:
                problem = json.loads(line)
                stats['total'] += 1
                problem_id = problem.get('id', f'line_{line_num}')
                
                was_modified = False
                if 'harness' in problem:
                    problem['harness'], changes = fix_harness(problem['harness'])
                    if changes:
                        was_modified = True
                        stats['modified'] += 1
                        stats['changes'].append(f"[{problem_id}]")
                        stats['changes'].extend(changes)
                
                if was_modified or not modified_only:
                    output_lines.append(json.dumps(problem, ensure_ascii=False))
                    
            except json.JSONDecodeError as e:
                stats['errors'].append(f"Line {line_num}: JSON error - {e}")
                if not modified_only:
                    output_lines.append(line)
            except Exception as e:
                stats['errors'].append(f"Line {line_num}: {type(e).__name__} - {e}")
                if not modified_only:
                    output_lines.append(line)
    
    # Write output
    if not dry_run:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
            if output_lines:
                f.write('\n')
        print(f"✅ Written to: {output_path}")
    else:
        print(f"[DRY RUN] Would write to: {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Fix cocotb 2.x compatibility issues in CVDP benchmark datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s datasets/input.jsonl
  %(prog)s datasets/input.jsonl -o fixed.jsonl
  %(prog)s datasets/input.jsonl --dry-run -v
  %(prog)s datasets/input.jsonl --modified-only
"""
    )
    parser.add_argument('input', help='Input JSONL file')
    parser.add_argument('-o', '--output', help='Output file (default: input_cocotb2.jsonl)')
    parser.add_argument('-n', '--dry-run', action='store_true', help='Preview without writing')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed changes')
    parser.add_argument('-m', '--modified-only', action='store_true', help='Only output modified problems')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        p = Path(args.input)
        output_path = str(p.parent / f"{p.stem}_cocotb2{p.suffix}")
    
    # Process
    stats = process_dataset(args.input, output_path, args.dry_run, args.modified_only, args.verbose)
    
    # Summary
    print()
    print("=" * 50)
    print(f"Total: {stats['total']} | Modified: {stats['modified']}")
    print("=" * 50)
    
    if args.verbose and stats['changes']:
        print("\nChanges:")
        for change in stats['changes']:
            print(change)
    
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors']:
            print(f"  {error}")


if __name__ == '__main__':
    main()
