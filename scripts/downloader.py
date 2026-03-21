#!/usr/bin/env python3

"""Download and normalize SacreBLEU test sets for evaluation."""

import argparse
import shutil
import sys
from pathlib import Path
from urllib.error import URLError

from sacrebleu import utils as sacrebleu_utils


DEFAULT_TESTSETS = ["wmt14", "wmt17", "wmt20", "wmt22", "wmt23"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download evaluation datasets from SacreBLEU and copy them to a local "
            "directory for evaluation."
        )
    )
    parser.add_argument(
        "--langpair",
        default="en-fr",
        help="Language pair in src-tgt format (default: en-fr).",
    )
    parser.add_argument(
        "--test-sets",
        default=",".join(DEFAULT_TESTSETS),
        help=(
            "Comma-separated list of SacreBLEU test set IDs to prepare "
            f"(default: {','.join(DEFAULT_TESTSETS)})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/eval",
        help="Output directory for prepared datasets (default: datasets/eval).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test sets for --langpair and exit.",
    )
    return parser.parse_args()


def parse_testsets(raw_value):
    values = [entry.strip() for entry in raw_value.split(",") if entry.strip()]
    if not values:
        raise ValueError("No test sets specified. Use --test-sets with at least one value.")
    return values


def copy_file(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def line_count(path):
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def write_manifest(manifest_path, rows):
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write("test_set\tsource\treferences\tsentences\n")
        for row in rows:
            references_str = ",".join(row["references"])
            handle.write(
                f"{row['test_set']}\t{row['source']}\t{references_str}\t{row['sentences']}\n"
            )


def main():
    args = parse_args()

    if "-" not in args.langpair:
        print("error: --langpair must be formatted like src-tgt (for example: en-fr)", file=sys.stderr)
        sys.exit(1)

    available_testsets = sacrebleu_utils.get_available_testsets_for_langpair(args.langpair)

    if args.list:
        if not available_testsets:
            print(f"No SacreBLEU test sets found for language pair: {args.langpair}")
            return
        print(f"Available test sets for {args.langpair}:")
        for testset in available_testsets:
            print(f"- {testset}")
        return

    try:
        selected_testsets = parse_testsets(args.test_sets)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    invalid_testsets = [testset for testset in selected_testsets if testset not in available_testsets]
    if invalid_testsets:
        print(
            "error: unavailable test set(s) for "
            f"{args.langpair}: {', '.join(invalid_testsets)}\n"
            "Run with --list to inspect valid test set IDs.",
            file=sys.stderr,
        )
        sys.exit(1)

    src_lang, tgt_lang = args.langpair.split("-", maxsplit=1)
    output_root = Path(args.output_dir)
    manifest_rows = []

    for testset in selected_testsets:
        try:
            source_path = Path(sacrebleu_utils.get_source_file(testset, args.langpair))
            reference_paths = [Path(path) for path in sacrebleu_utils.get_reference_files(testset, args.langpair)]
        except URLError as exc:
            print(
                f"error: failed to download {testset} for {args.langpair}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        dataset_dir = output_root / testset
        prepared_source = dataset_dir / f"source.{src_lang}.txt"
        prepared_references = [
            dataset_dir / f"reference{index + 1}.{tgt_lang}.txt"
            for index in range(len(reference_paths))
        ]

        copy_file(source_path, prepared_source)
        for src_ref, dst_ref in zip(reference_paths, prepared_references):
            copy_file(src_ref, dst_ref)

        sentence_count = line_count(prepared_source)
        ref_counts = [line_count(ref_path) for ref_path in prepared_references]
        if any(count != sentence_count for count in ref_counts):
            print(
                f"error: line-count mismatch in prepared {testset} files "
                f"(source={sentence_count}, refs={ref_counts})",
                file=sys.stderr,
            )
            sys.exit(1)

        manifest_rows.append(
            {
                "test_set": testset,
                "source": str(prepared_source),
                "references": [str(path) for path in prepared_references],
                "sentences": sentence_count,
            }
        )

        print(f"Prepared {testset}: {sentence_count} sentences")
        print(f"  source: {prepared_source}")
        for path in prepared_references:
            print(f"  reference: {path}")

    manifest_path = output_root / f"manifest.{args.langpair}.tsv"
    write_manifest(manifest_path, manifest_rows)
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
