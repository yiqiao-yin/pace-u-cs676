"""
make_homework.py — generate the student scripts from the answer keys.

INSTRUCTOR TOOL. The `answer/` directory it reads is gitignored and is not
distributed, so running this from a fresh clone will correctly report that it
found no answer keys. Students do not need this file.

    python notebooks/homework/make_homework.py

Each answer script in `answer/` marks its solution with a pair of sentinels:

    # BEGIN SOLUTION: short description
    ...the code students must write...
    # END SOLUTION

This script copies every answer file to the homework folder, drops the `_ans`
suffix, and replaces each marked block with a `NotImplementedError` stub at the
right indentation. Everything else — the data, the metrics, the printing, the
YOUR TASK instructions above the block — is byte-for-byte identical.

Generating rather than hand-writing matters: fix a bug in an answer key and the
homework picks it up on the next run, so the two versions cannot drift apart.
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
ANSWER_DIR = HERE / "answer"

BEGIN = re.compile(r'^(\s*)#\s*BEGIN SOLUTION:?\s*(.*)$')
END = re.compile(r'^\s*#\s*END SOLUTION\s*$')


def blank_out(lines, source_name):
    """Replace every BEGIN/END SOLUTION block with a stub. Returns (lines, count)."""
    out, i, replaced = [], 0, 0

    while i < len(lines):
        m = BEGIN.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        indent, description = m.group(1), (m.group(2).strip() or "the missing code")

        # Find the matching END, so an unbalanced marker is a loud failure
        # rather than a file that silently swallows the rest of itself.
        j = i + 1
        while j < len(lines) and not END.match(lines[j]):
            j += 1
        if j >= len(lines):
            raise SystemExit(f"{source_name}: 'BEGIN SOLUTION' at line {i + 1} has no 'END SOLUTION'")

        out.append(f'{indent}# YOUR CODE HERE — {description}\n')
        out.append(f'{indent}# Delete the raise below once you have written it.\n')
        out.append(f'{indent}raise NotImplementedError(\n')
        out.append(f'{indent}    "Homework: write {description}. "\n')
        out.append(f'{indent}    "See the YOUR TASK box just above for the steps."\n')
        out.append(f'{indent})\n')

        replaced += 1
        i = j + 1

    return out, replaced


def convert(answer_path):
    """Turn one answer file into its homework counterpart."""
    lines = answer_path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines, replaced = blank_out(lines, answer_path.name)
    if replaced == 0:
        raise SystemExit(f"{answer_path.name}: no BEGIN SOLUTION block found — nothing to blank")

    text = "".join(lines)
    # The header says ANSWER KEY; the student copy should not.
    text = text.replace("   [ANSWER KEY]\n", "\n", 1)
    # Reports and figures should not overwrite the answer key's output files.
    text = text.replace("_ans.py", ".py")

    out_path = HERE / answer_path.name.replace("_ans.py", ".py")
    out_path.write_text(text, encoding="utf-8")
    return out_path, replaced


def main():
    answers = sorted(ANSWER_DIR.glob("*_ans.py"))
    if not answers:
        raise SystemExit(f"no *_ans.py files found in {ANSWER_DIR}")

    print(f"Generating student scripts from {len(answers)} answer key(s):\n")
    for a in answers:
        out, n = convert(a)
        print(f"  {a.name:<22} -> {out.name:<20} ({n} block(s) blanked)")
    print(f"\nWrote {len(answers)} file(s) to {HERE}")


if __name__ == "__main__":
    sys.exit(main())
