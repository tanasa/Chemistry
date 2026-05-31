#!/usr/bin/env python3
"""
smiles_to_inchikey.py
=====================
Convert SMILES strings to InChIKey (and full InChI) using RDKit.

Usage:
    python smiles_to_inchikey.py                          
    python smiles_to_inchikey.py drugs.txt                # single file
    python smiles_to_inchikey.py --csv inchikeys_all.csv  # also write CSV

Requirements:
    pip install rdkit
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import inchi

BASE = Path(__file__).parent
DEFAULT_INPUTS = [BASE / "drugs1.txt", BASE / "drugs2.txt"]
DEFAULT_OUTPUT = BASE / "inchikeys_all.txt"


def inchikey_from_smiles(smiles: str) -> tuple[str | None, str | None, int | None, str | None]:
    """
    Convert SMILES to InChIKey, InChI, and formal charge.

    Returns (inchikey, inchi, charge, error).
    On success error is None; on failure inchikey/inchi/charge are None.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None, "invalid SMILES"

        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        return (
            inchi.MolToInchiKey(mol),
            inchi.MolToInchi(mol),
            Chem.GetFormalCharge(mol),
            None,
        )
    except Exception as exc:
        return None, None, None, str(exc)


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


def convert_record(smiles: str) -> dict:
    """Convert one SMILES and return a result dict."""
    ik, inchi_str, charge, err = inchikey_from_smiles(smiles)
    return {
        "smiles": smiles,
        "inchikey": ik,
        "inchi": inchi_str,
        "charge": charge,
        "error": err,
    }


def mark_duplicates(records: list[dict]) -> None:
    """Flag records that share an InChIKey within the same source file and set."""
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for rec in records:
        if rec["inchikey"]:
            groups[(rec["source"], rec["set"], rec["inchikey"])].append(rec)

    for rec in records:
        rec["duplicate"] = "NO"
        rec["same_as"] = "—"

    for (_source, _set_num, _ik), members in groups.items():
        if len(members) <= 1:
            continue
        names = [m["name"] for m in members]
        for rec in members:
            others = [n for n in names if n != rec["name"]]
            rec["duplicate"] = "YES"
            rec["same_as"] = ", ".join(others)


def write_text_report(records: list[dict], path: Path) -> None:
    lines: list[str] = [
        "InChIKeys from SMILES",
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
        if rec["inchikey"]:
            lines.append(f"  InChIKey: {rec['inchikey']}")
            lines.append(f"  InChI:    {rec['inchi']}")
            lines.append(f"  Charge:   {rec['charge']:+d}")
            if rec["duplicate"] == "YES":
                lines.append(f"  Duplicate: YES (same as {rec['same_as']})")
            lines.append(f"  SMILES:   {rec['smiles']}")
        else:
            lines.append(f"  InChIKey: INVALID")
            lines.append(f"  ERROR:    {rec['error']}")
            lines.append(f"  SMILES:   {rec['smiles']}")
        lines.append("")

    path.write_text("\n".join(lines))


def write_csv(records: list[dict], path: Path) -> None:
    fieldnames = [
        "source",
        "set",
        "name",
        "inchikey",
        "inchi",
        "charge",
        "duplicate",
        "same_as",
        "smiles",
        "error",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert drug SMILES to InChIKey.")
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
    args = parser.parse_args()

    inputs = args.inputs or DEFAULT_INPUTS
    inputs = [p if p.is_absolute() else BASE / p for p in inputs]

    all_records: list[dict] = []
    for path in inputs:
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        for entry in parse_drugs_file(path):
            all_records.append({**entry, **convert_record(entry["smiles"])})

    if not all_records:
        print("No compounds to convert.", file=sys.stderr)
        sys.exit(1)

    mark_duplicates(all_records)
    write_text_report(all_records, args.output)

    ok = sum(1 for r in all_records if r["inchikey"])
    failed = sum(1 for r in all_records if not r["inchikey"])
    dupes = sum(1 for r in all_records if r["duplicate"] == "YES")

    print(f"Wrote {args.output}")
    print(f"Converted: {ok}/{len(all_records)}")
    if failed:
        print(f"Failed:    {failed}")
    print(f"Duplicate InChIKeys (within set): {dupes}")

    if args.csv:
        write_csv(all_records, args.csv)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
