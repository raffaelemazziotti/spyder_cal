# Download page

`index.html` is a static GitHub Pages landing page. It has no JavaScript,
external fonts, build step, or dependencies. `assets/site.css` supplies the
layout and `assets/spyder-cal.png` shows an illustrative session in the real app.
`installation.html` covers Windows/Linux installation, sensor setup, measurement
and export. `api.html` covers Python installation and usage, with links to the
generated `cal_lib.html` and `cal_psy.html` reference. Existing module URLs still
work and their contents have been regenerated from the current source.

## Publishing

The repository already serves GitHub Pages from **main /docs** at
<https://raffaelemazziotti.github.io/spyder_cal/>. Pushing the page to main
updates the site without changing Pages settings.

Executables belong in **GitHub Releases**, rather than the Git history or the
Pages directory. The buttons use `/releases/latest/download/` with these stable
asset names:

| Asset | Purpose |
| --- | --- |
| `Spyder-Cal-windows-x64.zip` | Recommended Windows download: extract once and run `Spyder Cal.exe` with its adjacent files |
| `Spyder-Cal-windows-x64.exe` | Portable Windows launcher |
| `Spyder-Cal-linux-x86_64.AppImage` | Linux x86_64 application |
| `Spyder-Cal-0.1.2-source.zip` | Matching application source for this release |
| `SHA256SUMS.txt` | SHA-256 hashes of the release assets |

The prepared v0.1.2 files are in the ignored `release/publish-0.1.2` directory.
For each release, attach the stable download names above to the new release;
the page will then use the latest published version automatically. Tag the
commit containing the matching application source. Publish a normal release,
not a draft or prerelease, so GitHub's `latest` URLs resolve to it.

The Windows ZIP must include the **whole** app folder, including
`resources/backend`. An `.exe` copied out of that folder cannot run by itself.

## Preview and checks

Open `docs/index.html` locally, or serve `docs` with a static HTTP server. The
download links become available after the first release is published with the
asset names above.

`desktop/test/check-site.cjs` checks desktop/mobile layouts, images, local
links and download URLs using the installed Electron renderer. It writes
preview screenshots to the ignored `desktop/.cache` directory.

`desktop/test/capture-preview.cjs` updates the product screenshot using an
illustrative fixture and the actual app renderer. It performs no USB operations
and does not open a PsychoPy window.

## Refresh the Python reference

In the Python environment used by this release, install `pdoc==15.0.1`, then
generate into a temporary directory so the download homepage is never replaced:

```powershell
$env:PSYCHOPYNOPACKAGES = '1'
$env:MPLBACKEND = 'Agg'
.venv\Scripts\python.exe -m pdoc cal_lib cal_psy --docformat google --favicon assets/favicon.svg --logo assets/favicon.svg --logo-link api.html --footer-text 'Spyder Cal 0.1.2' -o desktop/.cache/api-reference
Copy-Item desktop/.cache/api-reference/cal_lib.html docs/cal_lib.html
Copy-Item desktop/.cache/api-reference/cal_psy.html docs/cal_psy.html
Copy-Item desktop/.cache/api-reference/search.js docs/search.js
```

On Linux, set the two environment variables using `export`, use
`.venv/bin/python`, and copy the same three files with `cp`. Keep the handwritten
installation/API guides in sync with the examples and current signatures.
