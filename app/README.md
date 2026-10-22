# 🖥️ `app/`

The ChargeIQ Streamlit app — live at **[chargeiq.streamlit.app](https://chargeiq.streamlit.app)**. It tells the project as a story in eleven chapters: the problem, then for each track *meet the data → how the model works → try it yourself*, and a conclusion.

## Run it locally

From the repository root, so `.streamlit/config.toml` is picked up exactly as on Streamlit Cloud:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The app reads only the committed files in `../models/` and `../assets/figures/` — none of the datasets.

## Layout

| Path | What it holds |
|:--|:--|
| `streamlit_app.py` | Entry point: page setup, navigation, and a readable error page if a model cannot be loaded |
| `chapters/` | One file per chapter, in reading order: `home`, then `vehicle_*`, `battery_*`, `charging_*` (data, model, predict), then `conclusion` |
| `core/` | Inference, one module per track — validation, prediction, explanations. Never imports Streamlit, so it is tested on its own |
| `content/` | Text taken from the notebooks: each quote with its notebook and cell, plus the example inputs |
| `ui/` | Navigation (`SHOW_UNFINISHED` hides unfinished chapters), shared components, `style.css` |
| `tests/` | 259 tests — see below |
| `requirements.txt` | Exact pins; Streamlit Cloud uses this file, not the root one, because it sits beside the entry point |
| `DEPLOYMENT.md` | How to deploy, verify and troubleshoot |

## Tests

```bash
python -m pytest app/tests -q
```

| File | Checks |
|:--|:--|
| `test_classification.py` · `test_regression.py` · `test_clustering.py` | Each saved model reproduces its notebook's results; input validation; explanations |
| `test_content.py` | Every quote appears word for word in the notebook cell it cites; every example input exists in the data |
| `test_app.py` | Every chapter renders; the numbers on screen match the models; every figure is byte-identical to its notebook output |
| `test_deploy.py` | What a visitor sees: no unfinished chapter or `TODO(team)` box, every import pinned, model files committed |

The regression replay tests skip when the local NASA data is absent; everything else runs anywhere.

## Writing conventions

- Interpretation comes from the team: quote the notebooks through `content/` (the tests check each quote), or add a `ui.team_note(title, text)`. `ui.team_todo(...)` marks text still to be written, and blocks deployment while visible.
- Every number shown must come from a model's metadata, a model itself, or a cited notebook cell.
