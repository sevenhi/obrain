# Obrain build, preview and recovery

## Sources and outputs

- Root Markdown and domain directories are the authoritative text sources.
- `diagram-specs/` preserves the existing structured diagram specifications. `diagram-assets.json` records the preservation baseline, with relative paths, sizes and SHA256 hashes.
- `diagrams/*.html` and validation receipts are local assets, still ignored by Git. **A Git checkout alone does not recover those HTML files. Keep the asset archive.** Regenerating a similar diagram from prose is not an exact restoration method.
- `tools/build_site.py` creates an exclusive temporary Markdown staging directory, overrides `docs_dir` and `site_dir` at build time, and discards only that staging directory. It does not read or recreate `.site/` and never clears the existing `site/`.
- The requested output directory must not exist, must be outside the source tree, and must not be an ancestor of the source tree. Existing directories, including empty directories, are rejected rather than cleaned.
- `mkdocs.yml` remains the theme/navigation configuration. Direct `mkdocs build` still uses its local `.site` setting; use the wrapper for reproducible builds.

## Runtime and locked dependencies

The tested baseline is Python 3.11, MkDocs 1.6.1, Material 9.7.7, jieba 0.42.1 and pymdown-extensions 12.3. `requirements-build.txt` pins the complete package set observed in the existing build environment on 2026-11-04, including transitive dependencies. It is a version lock, not a wheel/hash lock or an offline package archive.

Use an existing environment that contains these packages. `OBRAIN_PYTHON` may point to a different compatible Python environment. To create a fresh environment later, explicitly install the locked file (requires package availability and may require network access):

```bash
python3.11 -m venv /path/to/obrain-venv
/path/to/obrain-venv/bin/python -m pip install -r requirements-build.txt
export OBRAIN_PYTHON=/path/to/obrain-venv/bin/python
```

No build command installs dependencies. Invoke the tools with the selected interpreter; `--python` on the build tool selects the MkDocs subprocess interpreter, not the interpreter importing the validation dependencies.

## Check and build

Run from the repository root (the tools themselves do not depend on the working directory):

```bash
export OBRAIN_PYTHON="${OBRAIN_PYTHON:-../.venvs/obrain-mkdocs/bin/python}"
PYTHONDONTWRITEBYTECODE=1 "$OBRAIN_PYTHON" tools/preserve_assets.py check
PYTHONDONTWRITEBYTECODE=1 "$OBRAIN_PYTHON" tools/validate_obrain.py
PYTHONDONTWRITEBYTECODE=1 "$OBRAIN_PYTHON" -m unittest discover -s tools -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 "$OBRAIN_PYTHON" tools/build_site.py --mode portable --output /tmp/obrain-export-new
```

Replace the example output path with a new path each time. The build checks source Markdown, parses frontmatter using PyYAML, requires title/type/tags/source/date on knowledge articles, applies the documented notes/index exemptions, checks 速览 (TL;DR) and source links, checks domain index coverage, validates diagram receipts and the asset manifest when present, then validates built HTML links, anchors and iframes. A failure returns nonzero and does not publish the temporary build to the requested output.

Code examples are rendered as code before link inspection. README links are resolved by MkDocs to homepage/index URLs; built-link checking validates the actual result, not an invented source-to-URL mapping. External URLs are not fetched. Source-link checking does not validate anchors until after MkDocs rendering. Standalone diagram JavaScript is not executed by these checks, and dynamically generated diagram internals are not treated as ordinary content pages.

The build generates a fresh `DIAGRAM-INDEX.md` inside staging, without changing the source copy. MkDocs configuration controls whether this page appears in navigation/search. The source index can be updated separately:

```bash
"$OBRAIN_PYTHON" tools/build_diagram_index.py
"$OBRAIN_PYTHON" tools/build_diagram_index.py --check
"$OBRAIN_PYTHON" tools/build_diagram_index.py --output /tmp/diagram-index.md
"$OBRAIN_PYTHON" tools/build_diagram_index.py --output /tmp/diagram-index.md --check
```

`--check` never writes. The index enumerates every HTML file, not just receipts. Existing diagrams without a summary receipt are explicitly `legacy-unverified`; that label is not a quality pass. Bad status, missing receipt-linked HTML, invalid JSON and mismatched hashes fail. With a preservation manifest, missing baseline assets also fail. Verified receipts prove the recorded HTML version, not current factual correctness or a newly executed browser test.

## Portable export versus local preview

- **Portable** (default): copies the diagram HTML files into the export. No dependency on the source tree's diagrams symlink remains. Keep the whole export directory together; moving only its HTML pages or omitting diagrams breaks it.
- **Preview**: symlinks `output/diagrams` to the source asset directory. Faster and smaller, but deliberately not portable. Removing or moving the source assets breaks that preview.

```bash
"$OBRAIN_PYTHON" tools/build_site.py --mode preview --output /tmp/obrain-preview-new
"$OBRAIN_PYTHON" -m http.server 8000 --bind 127.0.0.1 --directory /tmp/obrain-preview-new
```

If the optional sibling scripts checkout exists, `bash ../scripts/serve_obrain.sh --build` builds and serves an exclusive temporary preview, deleting its own temporary directory after the server exits. `--site DIRECTORY` serves a chosen export; no arguments serves the existing local `site/` without changing it. `OBRAIN_ROOT`, `OBRAIN_PYTHON` and `PORT` are configurable; there are no fixed home-directory paths. The scripts checkout is not required to build or preview with the commands above.

The static server binds localhost only. Stop it with Ctrl-C. No persistent service is installed. To publish, validate a new complete portable directory before switching the reader/server to it. Keep the previous complete export for rollback; do not overwrite a live site's directory in place.

## Restore and protect diagram assets

The current verified preservation archive is outside the knowledge repository, at sibling `archive/obrain-backup-20261104/final/diagram-preservation.tar.gz`, with a `.sha256` companion. It contains the asset manifest, 285 HTML files, 303 stable specifications and 281 summary receipts. The initial archive at `archive/obrain-backup-20261104/diagram-preservation.tar.gz` is the pre-repair version; use it only with its own original manifest. Separate pre-repair snapshots also exist; they are not substitutes for validating the current preservation archive.

**This archive is a durable recovery source, not a disposable cache. Exempt it from any archive/90-day automatic cleanup policy, and copy the archive plus checksum to another device.** Keeping only a local tar beside the checkout does not protect against disk loss. No cloud or remote copy is claimed by this procedure.

After a clean checkout, use a trusted archive and restore into a new, exclusive temporary directory before copying missing baseline files into the checkout. Do not extract an untrusted tar into a live tree. Example verification of the preserved archive without extracting into the source tree:

```bash
"$OBRAIN_PYTHON" tools/preserve_assets.py check-backup --archive ../archive/obrain-backup-20261104/final/diagram-preservation.tar.gz
```

After restoring the documented members (`diagrams/`, `diagram-specs/`, `diagram-assets.json`), run `tools/preserve_assets.py check`, then the validation/build commands above. Restoring old `.archify` work directories is not required for static-site construction. To create another verified asset archive, choose a fresh external destination:

```bash
"$OBRAIN_PYTHON" tools/preserve_assets.py backup --destination /path/to/new-backup-directory
```

Asset updates must deliberately update the preservation baseline and create a new verified archive; do not edit hashes merely to suppress a failed check. The initialization command refuses to overwrite an existing baseline. Stable specifications help future regeneration but do not yet constitute a fully tested deterministic render pipeline for every legacy diagram.

After reviewing changed diagrams and their new validation receipts, the guarded refresh command verifies every member of the previous archive before replacing the current manifest. It retains the old manifest under `diagram-specs/manifests/` and refuses additions/removals of diagram URLs. Then create a new archive in a new destination:

```bash
"$OBRAIN_PYTHON" tools/preserve_assets.py refresh --archive /path/to/previous/diagram-preservation.tar.gz
"$OBRAIN_PYTHON" tools/preserve_assets.py backup --destination /path/to/new-backup-directory
```

Keep each archive together with the matching manifest and checksum. Checking an old archive against the updated manifest should fail; use the old manifest when auditing or restoring that historical version.

## Cleanup and rollback

Do not delete `diagram-specs/` or treat original `.archify` candidates as disposable until the preservation check and backup verification succeed. The sibling cleanup script defaults to no deletion; its `--staging` option is limited to old `.site` after checking assets. New builds do not need `.site`, so cleanup is optional and not part of the build path.

Rollback uses the previous complete portable export and its corresponding verified asset archive. Reverting only source Markdown while retaining changed diagrams may not restore a consistent version. No migration, history rewriting or commit is performed by these build tools.
