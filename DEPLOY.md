# Obrain deployment

## Layout

- Markdown sources live in this repository.
- `mkdocs.yml` uses `docs_dir: .site`.
- `.site/` contains symlinks to the Markdown roots (including `diagrams/`); it is local-only and ignored by Git.
- `site/` is the MkDocs build output; it is local-only and ignored by Git.

## Build and preview

```bash
../scripts/serve_obrain.sh --build
```

This:

1. activates `/home/cambricon/zhanbin/.venvs/obrain-mkdocs`
2. runs `mkdocs build`
3. recreates `site/diagrams -> ../diagrams`
4. serves `site/` at `http://127.0.0.1:8000`

To serve an existing build without rebuilding, omit `--build`.

## Dependencies

- MkDocs and Material are installed in the venv above.
- Chinese search requires `jieba`.
- Diagrams are plain HTML artifacts under `diagrams/`; they are excluded from MkDocs and served directly via the symlink.

## Maintenance

Regenerate the diagram index:

```bash
python3 tools/build_diagram_index.py
```

Clean local-only caches:

```bash
../scripts/clean_obrain.sh
```
