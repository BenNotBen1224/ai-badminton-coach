"""Append one decision row to decisions.tsv next to this script.

Usage: python log_row.py <phase> <decision> <why> <evidence> <result>
Cells are made single-line; a leading = + - @ is quoted so spreadsheets do not run it.
"""
import datetime
import pathlib
import sys

HEADER = "ts\tphase\tdecision\twhy\tevidence\tresult\n"
LOG = pathlib.Path(__file__).with_name("decisions.tsv")


def clean(cell: str) -> str:
    cell = " ".join(cell.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())
    return "'" + cell if cell[:1] in "=+-@" else cell


def main() -> None:
    if len(sys.argv) != 6:
        sys.exit("usage: log_row.py phase decision why evidence result")
    ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    row = "\t".join([ts] + [clean(a) for a in sys.argv[1:]]) + "\n"
    new = not LOG.exists()
    with LOG.open("a", encoding="utf-8", newline="") as f:
        if new:
            f.write(HEADER)
        f.write(row)
    print("logged:", row.strip()[:160])


if __name__ == "__main__":
    main()
