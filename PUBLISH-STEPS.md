# Publish Steps — tcad-stabilizer v0.6.0

Everything is prepared. You only do the account steps below — no code
changes needed. Estimated time: about 15 minutes.

## What is already done

- `README.md`, `LICENSE` (MIT), `setup.py`, `pyproject.toml`,
  `MANIFEST.in`, `.gitignore` — all ready in this folder.
- Build verified: wheel + sdist build cleanly, `twine check` PASSED,
  and a fresh `pip install` from the wheel passes the smoke test
  (1D diode solve + 2D MOSFET ramp, both from the README).
- The PyPI name `tcad-stabilizer` did not exist at check time
  (2026-10-26). If `twine upload` ever says the name is taken, rename
  it in `setup.py` and `README.md` and try again.

## Step 1 — Create the GitHub repo (about 3 min)

1. Go to https://github.com/new and sign in.
2. Repository name: `tcad-stabilizer`
3. Description: `Drift-diffusion TCAD device solver (1D/2D/3D diodes and MOSFETs) with a Newton stabilizer that converges on hard bias points`
4. Public. Do **not** tick "Add a README file", "Add .gitignore", or
   "Choose a license" — those files already exist here.
5. Click **Create repository**. GitHub shows you a page with commands —
   use the ones below instead (they match this folder).

## Step 2 — Push this folder to GitHub (about 3 min)

In a terminal, in this folder (`~/workspace/tcad/package`):

```bash
git init
git add .
git status        # check: only source files, README, LICENSE, docs/ — no dist/, no *.egg-info
git commit -m "tcad-stabilizer v0.6.0 — public release"
git branch -M main
git remote add origin https://github.com/<your-github-username>/tcad-stabilizer.git
git push -u origin main
```

Replace `<your-github-username>` with your GitHub username. If git
asks for a password, use a Personal Access Token (GitHub → Settings →
Developer settings → Personal access tokens → Tokens (classic) →
Generate new token, scope `repo`) as the password.

After the push, check the repo page on GitHub: README, plots, and
`tcad_stabilizer/SPEED.md` + `SWEEP.md` should all render.

## Step 3 — Create a PyPI account (about 3 min)

1. Go to https://pypi.org/account/register/ and create an account
   (use the same email as the package: sharmapiyush28965@gmail.com).
2. Verify your email (PyPI sends a confirmation link).
3. Turn on two-factor authentication (PyPI → Account settings →
   Two factor authentication). PyPI requires it before you can upload.

## Step 4 — Create a PyPI API token (about 2 min)

1. PyPI → Account settings → **API tokens** → **Add API token**.
2. Token name: `tcad-stabilizer-upload`. Scope: select
   **Entire account** (simplest) or **Project: tcad-stabilizer** if
   the project already exists.
3. Copy the token immediately — PyPI shows it only once. It starts
   with `pypi-`. Keep it private; never paste it in chat.

## Step 5 — Build and upload (about 4 min)

In a terminal, in this folder:

```bash
pip install --upgrade build twine
python -m build
twine check dist/*
twine upload dist/*
```

When `twine upload` asks for a username, type `__token__`.
When it asks for a password, paste the API token from Step 4.

Expected output: `100%` upload progress, then a link to
https://pypi.org/project/tcad-stabilizer/.

## Step 6 — Verify the public install (about 2 min)

In a fresh terminal (any folder):

```bash
python -m venv /tmp/verify-venv
/tmp/verify-venv/bin/pip install tcad-stabilizer
/tmp/verify-venv/bin/python -c "import tcad_stabilizer; print(tcad_stabilizer.__version__)"
```

You should see `0.6.0`. If it installs and imports, the release is live.

## Optional — make the plots render on PyPI too

The README images use relative paths (`docs/diode_iv.png`), which
render on GitHub but not on PyPI. After the repo exists, replace
them with absolute URLs of this form in `README.md`:

```
https://raw.githubusercontent.com/<your-github-username>/tcad-stabilizer/main/docs/diode_iv.png
```

Do the same for `mosfet_curves.png`, `mosfet_3d_xsec.png`, and
`diode_3d_xsec.png`. Then bump the version (see below) and re-upload.

## Later releases (version bumps)

1. Edit the version in `setup.py` **and**
   `tcad_stabilizer/__init__.py` (both must match, e.g. `0.6.1`).
2. Update `README.md` if anything changed.
3. `git add . && git commit -m "v0.6.1" && git push`
4. `python -m build && twine check dist/* && twine upload dist/*`
   (only the newest version uploads; old versions stay on PyPI).

## If something fails

- `twine upload` says the name is taken: rename in `setup.py` and
  `README.md`, rebuild, upload again.
- `twine upload` says 403: the token is wrong or lacks scope —
  create a new token in Step 4.
- `pip install tcad-stabilizer` says "no matching distribution":
  wait 2–3 minutes (PyPI index lag) and try again.
- Anything else: send the exact error message to Muse.
