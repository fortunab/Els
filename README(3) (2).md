# Els

This repository contains experimental scripts for evaluating classification performance, federated learning behavior, cross-institution generalization, explainability, calibration, NAS efficiency, state-of-the-art comparisons, framework comparisons, and ablation studies.

The project can be executed either script by script or all at once using `run\_all.py`.

## Repository

```bash
git clone https://github.com/fortunab/Els.git
cd Els
```

## Requirements

Make sure you have the following installed:

* Python 3.9 or newer
* Git
* pip

## Installation with Virtual Environment

Using a virtual environment is recommended because it isolates the dependencies of this project from other Python projects on your system.

### Windows

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Run All Experiments

To execute all available experiments using `run\_all.py`, run:

### Windows

```powershell
.\\.venv\\Scripts\\python.exe src/run\_all.py
```

### Linux/macOS

```bash
./.venv/bin/python run\_all.py
```

## Run Individual Scripts

Each experiment can also be executed independently.

### Windows

```powershell
.\\.venv\\Scripts\\python.exe src/classification\_performance.py
.\\.venv\\Scripts\\python.exe src/federated\_performance.py
.\\.venv\\Scripts\\python.exe src/cross\_institution\_generalization.py
.\\.venv\\Scripts\\python.exe src/explainability\_analysis.py
.\\.venv\\Scripts\\python.exe src/calibration\_analysis.py
.\\.venv\\Scripts\\python.exe src/nas\_efficiency.py
.\\.venv\\Scripts\\python.exe src/sota\_comparison.py
.\\.venv\\Scripts\\python.exe src/extended\_metrics\_domains.py
.\\.venv\\Scripts\\python.exe src/extended\_sota\_metrics.py
.\\.venv\\Scripts\\python.exe src/framework\_comparison.py
.\\.venv\\Scripts\\python.exe src/ablation\_study.py
```

### Linux/macOS

```bash
./.venv/bin/python classification\_performance.py
./.venv/bin/python federated\_performance.py
./.venv/bin/python cross\_institution\_generalization.py
./.venv/bin/python explainability\_analysis.py
./.venv/bin/python calibration\_analysis.py
./.venv/bin/python nas\_efficiency.py
./.venv/bin/python sota\_comparison.py
./.venv/bin/python extended\_metrics\_domains.py
./.venv/bin/python extended\_sota\_metrics.py
./.venv/bin/python framework\_comparison.py
./.venv/bin/python ablation\_study.py
```

## Optional: Activate the Virtual Environment

Instead of calling the virtual environment Python directly every time, you can activate it.

### Windows CMD

```cmd
.venv\\Scripts\\activate.bat
```

### Windows PowerShell

```powershell
.venv\\Scripts\\Activate.ps1
```

If PowerShell blocks script execution, you can still use the direct commands shown above, such as:

```powershell
.\\.venv\\Scripts\\python.exe classification\_performance.py
```

### Linux/macOS

```bash
source .venv/bin/activate
```

After activation, scripts can be executed simply with:

```bash
python classification\_performance.py
python run\_all.py
```

## Project Scripts

|Script|Purpose|
|-|-|
|`classification\_performance.py`|Evaluates classification performance.|
|`federated\_performance.py`|Evaluates performance in federated learning settings.|
|`cross\_institution\_generalization.py`|Tests generalization across institutions or data sources.|
|`explainability\_analysis.py`|Runs explainability-related analyses.|
|`calibration\_analysis.py`|Evaluates calibration behavior of models.|
|`nas\_efficiency.py`|Analyzes neural architecture search efficiency.|
|`sota\_comparison.py`|Compares results with state-of-the-art methods.|
|`extended\_metrics\_domains.py`|Computes extended metrics across domains.|
|`extended\_sota\_metrics.py`|Computes extended metrics for SOTA comparison.|
|`framework\_comparison.py`|Compares different frameworks or methodological variants.|
|`ablation\_study.py`|Runs ablation experiments.|
|`run\_all.py`|Executes the full experimental workflow.|

## Troubleshooting

### Git is not recognized on Windows

Install Git for Windows, then open a new PowerShell or CMD window and verify:

```powershell
git --version
```

### Missing Python package

If you see an error such as `ModuleNotFoundError`, reinstall dependencies inside the virtual environment:

### Windows

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

### Linux/macOS

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

### PowerShell script execution is disabled

You do not need to activate the environment. Use the direct virtual environment Python command instead:

```powershell
.\\.venv\\Scripts\\python.exe run\_all.py
```

## License

Please refer to the repository license file if available.

## Citation

If this repository supports a scientific publication, please cite the corresponding paper or project once citation information is available.

