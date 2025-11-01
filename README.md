# cs224_soccer_prediction

This is a Python 3.9 project for cs224W.

## Prerequisites
- Python 3.9.x installed (recommended via pyenv or your system package manager)
- pip available in your environment

## Setup
1. Create and activate a virtual environment (Python 3.9):
   - macOS/Linux:
     ```bash
     python3.9 -m venv .venv
     source .venv/bin/activate
     ```
   - If `python3.9` is not available, install Python 3.9 (e.g., with pyenv) and retry.

2. Verify the Python version:
   ```bash
   python --version  # Should show Python 3.9.x
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data
- **Datasets**: `actions.csv`, `games.csv`, `players.csv`, `teams.csv`.
- **Git tracking**: These files are large and are excluded via `.gitignore`.
- **Download**: Get them from Kaggle: [Soccer Match Event Dataset](https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset).
- **Placement**: Put the four CSVs into the `app/data` directory so the code can find them.

## Run
From the project root:
```bash
python -m app
```

Expected output:
```
Hello World
```

## Notes
- No external dependencies are required.
- Deactivate the environment when done:
  ```bash
  deactivate
  ```

