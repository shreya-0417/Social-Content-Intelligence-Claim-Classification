# Contributing

## Licensing note first

This repository is a presentation migration of an all-rights-reserved original
work (see [LICENSE](LICENSE)). The analytical implementation, the dataset and the
original figures were created by Dane Tipene and are **not** licensed for reuse,
redistribution or derivative publication without the copyright owner's written
permission.

That means:

- Issues and written feedback are welcome.
- Code, notebook or documentation changes cannot be merged without the
  copyright owner's permission.
- Do not fork this repository in order to publish a rebranded copy.

## Reporting a problem

Open an issue with:

1. The notebook or file affected.
2. The command you ran and the Python version.
3. The full traceback, if there is one.
4. What you expected to happen.

## Local checks before proposing a change

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# structural validation of every notebook
python -c "import glob, nbformat; [nbformat.validate(nbformat.read(f, as_version=4)) for f in glob.glob('notebooks/*.ipynb')]; print('notebooks valid')"

# re-run the cleaning stage
jupyter nbconvert --to notebook --execute notebooks/05_preprocess_dataset.ipynb \
  --output-dir /tmp --ExecutePreprocessor.timeout=600
```

## Style

- Keep notebook cell output out of commits unless it documents a result.
- Never commit credentials, local absolute paths or `.env` files.
- Keep figures that belong to a notebook inside `assets/` and reference them
  with repository-relative paths.
