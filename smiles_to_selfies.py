#!/usr/bin/env python3
"""
smiles_to_selfies.py
====================
Convert SMILES strings to SELFIES and optionally verify round-trip identity.

SELFIES preserves stereochemistry (e.g. [C@@H1]) and formal charges (e.g. [N+1]).
Decoded SMILES may differ in syntax from the input but should represent the same
molecule (checked via InChIKey when RDKit is available).

Usage:
    python smiles_to_selfies.py                         
    python smiles_to_selfies.py drugs.txt               # single input file
    python smiles_to_selfies.py --csv selfies_table.csv # also write CSV

Requirements:
    pip install selfies rdkit
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import selfies as sf

try:
    from rdkit import Chem
    from rdkit.Chem import inchi

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

BASE = Path(__file__).parent
DEFAULT_INPUTS = [BASE / "drugs1.txt", BASE / "drugs2.txt"]
DEFAULT_OUTPUT = BASE / "selfies.txt"


def smiles_to_selfies(smiles: str) -> str:
    """Convert a SMILES string to SELFIES."""
    return sf.encoder(smiles)


def selfies_to_smiles(selfies: str) -> str:
    """Convert a SELFIES string back to SMILES."""
    return sf.decoder(selfies)


def encode_safe(smiles: str) -> tuple[str | None, str | None]:
    """
    Safely encode SMILES to SELFIES.

    Returns (selfies, error). On success error is None; on failure selfies is None.
    """
    try:
        return smiles_to_selfies(smiles), None
    except Exception as exc:
        return None, str(exc)


def same_molecule(smiles_a: str, smiles_b: str) -> bool | None:
    """Return True if two SMILES represent the same molecule (InChIKey match)."""
    if not HAS_RDKIT:
        return None
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return False
    Chem.AssignStereochemistry(mol_a, force=True, cleanIt=True)
    Chem.AssignStereochemistry(mol_b, force=True, cleanIt=True)
    return inchi.MolToInchiKey(mol_a) == inchi.MolToInchiKey(mol_b)


def convert_record(smiles: str, verify: bool = True) -> dict:
    """Convert one SMILES and optionally verify round-trip with RDKit."""
    selfies, err = encode_safe(smiles)
    record = {
        "smiles": smiles,
        "selfies": selfies,
        "error": err,
        "recovered_smiles": None,
        "roundtrip_ok": None,
    }
    if selfies is None:
        return record

    try:
        recovered = selfies_to_smiles(selfies)
        record["recovered_smiles"] = recovered
        if verify and HAS_RDKIT:
            record["roundtrip_ok"] = same_molecule(smiles, recovered)
    except Exception as exc:
        record["error"] = f"decode: {exc}"

    return record


def encode_batch(smiles_list: list[str]) -> list[str | None]:
    """Safely encode a batch, returning None for unparseable entries."""
    return [encode_safe(smi)[0] for smi in smiles_list]


def parse_drugs_file(path: Path) -> list[dict]:
    """Parse task*.drugs.txt into records with set, name, and smiles."""
    entries: list[dict] = []
    current_set: int | None = None
    drug_num = 0
    source = path.name

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if m := re.match(r"# Set\s+#?(\d+)", line):
            current_set = int(m.group(1))
            drug_num = 0
            continue
        if not line or current_set is None:
            continue

        drug_num += 1
        entries.append(
            {
                "source": source,
                "set": current_set,
                "name": f"set{current_set}_drug{drug_num}",
                "smiles": line,
            }
        )
    return entries


def write_text_report(records: list[dict], path: Path) -> None:
    lines: list[str] = [
        "SELFIES conversion for drug SMILES",
        "=" * 60,
        "",
    ]
    current_header: tuple[str, int] | None = None

    for rec in records:
        header = (rec["source"], rec["set"])
        if header != current_header:
            current_header = header
            lines.append(f"# {rec['source']} — Set {rec['set']}")
            lines.append("")

        lines.append(rec["name"])
        lines.append(f"  SMILES:   {rec['smiles']}")
        if rec["selfies"]:
            lines.append(f"  SELFIES:  {rec['selfies']}")
            if rec["recovered_smiles"]:
                lines.append(f"  Decoded:  {rec['recovered_smiles']}")
            if rec["roundtrip_ok"] is True:
                lines.append("  Round-trip: OK (same InChIKey)")
            elif rec["roundtrip_ok"] is False:
                lines.append("  Round-trip: MISMATCH (different InChIKey)")
        else:
            lines.append(f"  ERROR:    {rec['error']}")
        lines.append("")

    path.write_text("\n".join(lines))


def write_csv(records: list[dict], path: Path) -> None:
    fieldnames = [
        "source",
        "set",
        "name",
        "smiles",
        "selfies",
        "recovered_smiles",
        "roundtrip_ok",
        "error",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def demo() -> None:
    """Run small examples from the documentation."""
    print("=== Simple example (ethanol) ===")
    smiles = "CCO"
    selfies = smiles_to_selfies(smiles)
    recovered = selfies_to_smiles(selfies)
    print(f"SMILES:   {smiles}")
    print(f"SELFIES:  {selfies}")
    print(f"Decoded:  {recovered}")

    print("\n=== Charged / stereochemistry example ===")
    complex_smiles = "CC[N+][C@@](c1ccccc1)C(c2ccc(CC)cc2)C3CCO3"
    rec = convert_record(complex_smiles)
    if rec["selfies"]:
        print("Successful conversion!")
        print(rec["selfies"])
        if rec["roundtrip_ok"] is not None:
            print(f"Round-trip InChIKey match: {rec['roundtrip_ok']}")
    else:
        print(f"Conversion failed: {rec['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert drug SMILES to SELFIES.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input drug list(s) (default: task1.drugs.txt and task2.drugs.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Text report output (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in examples and exit",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip RDKit round-trip verification",
    )
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    inputs = args.inputs or DEFAULT_INPUTS
    inputs = [p if p.is_absolute() else BASE / p for p in inputs]

    all_records: list[dict] = []
    for path in inputs:
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        for entry in parse_drugs_file(path):
            rec = convert_record(entry["smiles"], verify=not args.no_verify)
            all_records.append({**entry, **rec})

    if not all_records:
        print("No compounds to convert.", file=sys.stderr)
        sys.exit(1)

    write_text_report(all_records, args.output)

    ok = sum(1 for r in all_records if r["selfies"])
    failed = sum(1 for r in all_records if not r["selfies"])
    mismatches = sum(1 for r in all_records if r["roundtrip_ok"] is False)

    print(f"Wrote {args.output}")
    print(f"Converted: {ok}/{len(all_records)}")
    if failed:
        print(f"Failed:    {failed}")
    if HAS_RDKIT and not args.no_verify:
        print(f"Round-trip mismatches: {mismatches}")
    elif not HAS_RDKIT:
        print("RDKit not installed — round-trip verification skipped")

    if args.csv:
        write_csv(all_records, args.csv)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
