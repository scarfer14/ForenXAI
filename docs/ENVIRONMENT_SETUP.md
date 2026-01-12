# ENVIRONMENT SETUP

To set up the development environment for ForenXAI, follow these steps:

1. Navigate to the project directory

```bash
    cd ForenXAI
```

2. Create/activate venv (virtual environment)

Windows (CMD):

```cmd
python -m venv forenxai_env
forenxai_env\Scripts\activate
```

Windows (PowerShell):

```powershell
python -m venv forenxai_env
./forenxai_env/Scripts/Activate.ps1
```

macOS/Linux:

```bash
python -m venv forenxai_env
source forenxai_env/bin/activate
```

3.Activate the virtual environment

```cmd
forenxai_env\Scripts\activate.bat (for Windows CMD)
```

```cmd
forenxai_env\Scripts\Activate.ps1 (for Windows PowerShell)
```

4.Install dependencies

```bash
    pip install --upgrade pip
    pip install -r requirements.txt
```

5.If not used, deactivate the virtual environment

```bash
    deactivate