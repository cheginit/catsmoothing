"""Hooks for the documentation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mkdocs.structure.files import File, Files

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig

_ROOT = Path(__file__).parent.parent

changelog = _ROOT / "CHANGELOG.md"
contributing = _ROOT / "CONTRIBUTING.md"
readme = _ROOT / "README.md"


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Add root-level markdown files to the documentation site."""
    for path in (changelog, contributing, readme):
        files.append(
            File(
                path=path.name,
                src_dir=str(path.parent),
                dest_dir=str(config.site_dir),
                use_directory_urls=config.use_directory_urls,
            )
        )
    return files
