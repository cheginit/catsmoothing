"""Convert the notebooks in ``docs/examples/`` to Markdown pages next to them.

Zensical renders no notebooks and runs no MkDocs hooks, so ``pixi r docs`` runs this
first. Notebooks are converted with their saved outputs and never executed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from nbconvert.nbconvertapp import main

if Path("docs/examples/index.ipynb").exists():
    sys.exit("docs/examples/index.ipynb would overwrite index.md; rename it.")
notebooks = sorted(str(p) for p in Path("docs/examples").glob("*.ipynb"))
# nbconvert exits non-zero when given no notebooks.
if notebooks:
    main(["--to", "markdown", "--output-dir", "docs/examples", *notebooks])
# The site publishes each .ipynb beside its page; link it under the page title.
for notebook in map(Path, notebooks):
    page = notebook.with_suffix(".md")
    title, _, body = page.read_text(encoding="utf-8").partition("\n")
    link = f"[Download this notebook]({notebook.name}){{ .md-button }}\n"
    page.write_text(f"{title}\n\n{link}{body}", encoding="utf-8")
