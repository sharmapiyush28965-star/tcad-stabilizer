# nanoHUB — submission guide for tcad-stabilizer

Researched 2026-10-26 from nanoHUB's own site, HUBzero documentation,
and peer-reviewed papers by the nanoHUB team. No account was created
and nothing was submitted.

## What nanoHUB is

nanoHUB (nanohub.org) is a free cloud platform for computational
nanotechnology, run by the Network for Computational Nanotechnology
(NCN) at **Purdue University** and funded by the US National Science
Foundation. Simulation tools run in the browser — users need no
download or install. Live figures (2026-10-26): **960+ simulation
tools, 9,800+ resources, 6,000+ contributors, 384,000+ registered
members**. The audience is researchers, educators, and students
worldwide; nanoelectronics is a core area. At the moment of checking,
the most-used tools included Silvaco TCAD (57 users),
PN Junction Lab (32), and the ABACUS semiconductor package (15).

**Fit: good.** Python drift-diffusion device simulators already live
there — the closest precedent is **ADEPT** (1D drift-diffusion,
Poisson + electron/hole continuity,
https://nanohub.org/tools/adept). Our differentiators are 2D/3D
MOSFETs and a modern pure-Python implementation.

## Cost, affiliation, licence

- **Free** to register and publish. No submission fee.
- **No affiliation requirement** — open to anyone, no US institution
  needed.
- Open source is **encouraged but NOT required**. Every tool must be
  open access to *run* (any nanoHUB user can execute it), but the
  source code may stay closed if the developer states a reason.
  (Historically only ~50 of 341 tools were open-source.)
- **Export control — the one legal check:** the author is
  responsible for knowing whether the code has export restrictions.
  A general drift-diffusion solver is normally uncontrolled, but this
  is the owner's determination to make. nanoHUB lets authors restrict
  access (full / US-only / non-D1 nations / specific groups) if needed.

## How to publish a tool (exact process)

1. **Create a free nanoHUB account** at https://nanohub.org/
   (registration is free).
2. **Register the tool** — fill in the tool registration form
   (name, ID, description). Registration creates a code repository
   for the tool (Subversion or Git — chosen at registration).
3. **Upload and develop the code** in that repo, inside nanoHUB's
   in-browser Linux **Workspace** (a virtual desktop with the
   Rappture toolkit and compute access). Standard repo layout:
   `bin/` (executable), `middleware/invoke` (launch script),
   `src/` (source), plus `doc/`, `data/`, `examples/`, `rappture/`.
4. **Test in the nanoHUB environment**, then move through the guided
   Tool Development Pipeline lifecycle (stages on the tool's status
   page).
5. **Approve and publish** — the **developer** approves publication
   ("I approve it" on the status page). There is no formal peer-review
   gate that could be verified; quality control is post-publication
   (user reviews, Q&A, usage stats, citations). HUB Support stages
   versions on request.
6. **Publication outputs:** the tool gets a **DOI** (citable, indexed
   by Web of Science); each new version gets its own DOI. New
   releases can be made at any time.

## What has to be built — our solver does NOT fit as-is

Every nanoHUB tool needs a web-facing interface; a bare command-line
script cannot be published unchanged. Two realistic routes:

### Route A — Rappture (classic; best for "run the solver, show I–V curves")

- Write a `tool.xml` declaring inputs (doping, geometry, biases)
  and outputs (curves, logs, fields). Rappture auto-generates the
  GUI from that XML.
- Add a thin Python driver: read inputs via the Rappture library,
  call the existing solver, return the outputs. The solver core
  stays untouched.
- Add a `middleware/invoke` shell script that launches the tool.
- A starter `tool.xml` is prepared in `nanohub/tool.xml` (inputs:
  doping, mesh, bias; outputs: I–V curve data) — it still needs
  the driver script and testing inside a nanoHUB Workspace.

### Route B — Jupyter notebook / Sim2L (modern; recommended for new tools)

- **Jupyter tool:** the solver behind a notebook UI (the `hublib`
  Python library builds widgets). Notebook Mode (code visible) or
  App Mode (code hidden, runs automatically).
- **Sim2L (newest):** a notebook with a `parameters` cell, the full
  workflow, and all outputs declared in a single YAML cell.
  Published Sim2Ls get a DOI, are API-queryable, and every run's
  results are indexed in the searchable ResultsDB. Most
  metadata-heavy, but the most discoverable.

Either way the solver ships as auxiliary files alongside the
notebook/driver — the numerics do not change.

## What the owner must provide

- One free nanoHUB account; a named development team (only team
  members can edit code/metadata).
- Tool landing-page metadata: abstract/description, input and output
  specifications, references/citations, sources of error and
  caveats, credits, how to use the tool.
- Documentation: at least a first-time user guide (our README.md
  covers most of this) plus references supporting the approach.
- Regression tests: nanoHUB supports developer test suites — our
  existing test cases map directly onto this.
- Licence decision (open vs closed source, with a stated reason if
  closed).

## Blockers and open questions

- **Python dependencies:** how NumPy/SciPy are provisioned for a
  Rappture tool is NOT verified — Jupyter kernels ship common Python
  packages, but for a Rappture tool the mechanism (conda env,
  invoke-script install, or pre-installed modules) needs confirming
  with HUB Support before committing to a route.
- **Compute weight:** very large/memory-intensive jobs are a known
  constraint class; 2D/3D MOSFET runs may need the `submit` path to
  HPC rather than the interactive session. Verify runtime/memory
  limits with HUB Support early.
- **Timeline:** no published review SLA. The process is
  developer-paced, so the timeline is dominated by our own wrapper
  work.
- **Exact "Contribute a tool" entry URL:** the registration form
  lives behind login; the precise click-path on today's nanoHUB needs
  a logged-in check (or an email to HUB Support).

## Recommended next step

Email HUB Support from the nanoHUB site and ask three things:
(a) the current contribute-entry URL, (b) NumPy/SciPy provisioning
for Rappture tools, (c) runtime/memory limits for 3D runs. Then
build the Route A wrapper (`tool.xml` + driver) and test it in a
Workspace.

## Sources (verbatim)

- Homepage / stats: https://nanohub.org/
- Precedent tool ADEPT: https://nanohub.org/tools/adept
- HUBzero tool-development guide: https://theghub.org/tool-development
- Sim2L / FAIR paper (publishing process): https://arxiv.org/pdf/2306.11062
- Sim2L specification: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0264492
- "Cloud computing in nanoHUB" (registration → repo → workspace → approve → publish): https://docs.lib.purdue.edu/ses2014/nanohub/cloudcomp/20/
- Rappture via hublib docs: https://hubzero.github.io/hublib/rappture.html
- nanoHUB overview (Purdue/NCN, audience): https://serc.carleton.edu/msu_nanotech/Nano_Hub.html
- Purdue news (scale, NSF grant): https://engineering.purdue.edu/ECE/News/2020/democratizing-learning-and-research-in-nanotechnology
- Sustainability/licensing/export-control paper: https://account.openresearchsoftware.metajnl.com/index.php/up-j-jors/article/view/jors.bd
