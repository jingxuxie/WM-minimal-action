"""Compile the manuscript using an installed TeX distribution."""
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent

def main() -> None:
    tex = shutil.which("pdflatex")
    bib = shutil.which("bibtex") or shutil.which("bibtex8")
    if not tex or not bib:
        raise SystemExit("Install a TeX distribution providing pdflatex and bibtex (or bibtex8).")
    commands = [[tex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], [bib, "main"]]
    commands += [[tex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]] * 3
    for i, command in enumerate(commands, 1):
        with (HERE / f"build_{i}.log").open("w") as log:
            result = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"Build failed; inspect paper/build_{i}.log")
    print(HERE / "main.pdf")

if __name__ == "__main__":
    main()
