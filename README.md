# Daily ML Projects

A collection of small, self-contained machine learning projects — with one new project **published automatically every day** via GitHub Actions.

Each project is a standalone folder containing:
- Working Python code
- A short `README.md` explaining the idea, approach, and how to run it
- A `requirements.txt`

## How the daily automation works

- All projects live in [`library/`](library/) (the backlog).
- A scheduled GitHub Actions workflow runs once per day.
- It picks the **next** project from the library, copies it into [`published/`](published/), updates the log, and pushes a commit.
- This creates one meaningful commit (and a green contribution square) per day, fully hands-free.

See [`published/`](published/) for projects released so far.

## Published log

Released projects are tracked in [`published/PUBLISHED_LOG.md`](published/PUBLISHED_LOG.md).

## Run a project locally

```bash
cd published/<project-folder>
pip install -r requirements.txt
python main.py
```

---
Maintained by [Krish-naa](https://github.com/Krish-naa)
