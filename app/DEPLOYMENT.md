# Deploying ChargeIQ

The app is deployed on **Streamlit Community Cloud** (free) from this repository.
It needs no secrets, API keys or datasets: everything it shows comes from the
committed files in `models/` and `assets/figures/`.

| Setting | Value |
|:--|:--|
| Repository | `Roots26Hz/ChargeIQ` |
| Branch | `main` |
| Main file path | `app/streamlit_app.py` |
| Python version | **3.13** — Community Cloud defaults to 3.12 |
| Dependencies | `app/requirements.txt` (used automatically, because it sits beside the entrypoint) |
| Theme | `.streamlit/config.toml` (must stay at the repository root) |
| Secrets | none |

*Streamlit details below were checked against docs.streamlit.io in October 2026.*

---

## 1. Before every deploy

From the repository root:

```bash
.venv/bin/python -m pytest app/tests -q    # must pass, including test_deploy.py
git status                                 # the app's changes are committed
git push origin main                       # Community Cloud deploys what is on GitHub
```

`app/tests/test_deploy.py` checks what a visitor will see: unfinished chapters
are hidden (`SHOW_UNFINISHED = False` in `app/ui/navigation.py`), no visible
chapter has a `TODO(team)` box or placeholder, every import is pinned, and the
model files are committed and unmodified.

## 2. One-time setup

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. The repository is **private**, so Community Cloud needs permission to read it:
   **Settings → Linked accounts → Source control → Connect here → Authorize streamlit**.
3. You need **admin** rights on the repository to deploy from it — the
   repository owner has them.

## 3. Deploy

1. **Create app**, and choose to deploy from a GitHub repository.
2. Fill in the repository, branch and main file path from the table above.
3. *Optional:* under **App URL**, choose a memorable subdomain, e.g. `chargeiq`.
4. Open **Advanced settings** and set **Python version → 3.13**. Leave secrets empty.
5. **Deploy.** The first build installs the requirements and takes a few minutes.

> **The Python version cannot be changed after deploying.** If the app was
> deployed with the wrong version, delete it and deploy it again with the same
> subdomain, choosing 3.13 under Advanced settings.

## 4. Make it public

An app deployed from a private repository is **private by default**: only
developers in your workspace can open it. The capstone bonus needs a public URL, so:

**App settings → Sharing → Who can view this app → "This app is public and searchable"**

## 5. Verify

Open the app URL in a **private/incognito window, signed out** — that is what an
examiner sees. Check that:

- Chapter 1 loads, and the sidebar lists only the four finished chapters;
- in *Make a prediction*, choosing **Ford Ranger 2000** shows **PHEV** with a
  warning that it is one of the model's 5 test-set errors;
- choosing **BMW i3 2017 · range 97** shows **PHEV**, and changing the range to 81 shows **BEV**.

## Updating the app

Push to `main`; Community Cloud redeploys automatically within moments. Run the
tests before every push.

## Before an evaluation

Apps with no traffic for **12 hours go to sleep**. A visitor can wake one with
**"Yes, get this app back up!"**, but it takes a moment, so open the app yourself
shortly before it is assessed.

## Troubleshooting

Logs: open the app and click **Manage app** (lower right).

| Symptom | Cause and fix |
|:--|:--|
| "This chapter can't load its model … exported with scikit-learn 1.9.0, but this environment has …" | The installed scikit-learn differs from the one that wrote the model. Check the app was deployed with Python **3.13** (if not, delete and redeploy) and that `app/requirements.txt` still pins `scikit-learn==1.9.0`. |
| "This chapter can't load its model … Missing model file" | `models/classification_model.joblib` or its metadata is not on GitHub. Commit and push it. |
| Build fails while installing requirements | Usually the wrong Python version: the pinned versions are tested on 3.13. Delete and redeploy with 3.13. |
| `ModuleNotFoundError` | A new import was added without pinning it in `app/requirements.txt` (`test_deploy.py` catches this locally). |
| Default fonts and colours | `.streamlit/config.toml` was moved; it must stay at the repository root. |
| Visitors are asked to sign in | The app is still private — see step 4. |
