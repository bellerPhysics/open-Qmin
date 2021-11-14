#!/bin/bash

command -v deactivate # deactivate existing environment if possible
python3="/usr/bin/env python3.9" # or substitute your own path to python<3.10
envname="openViewMin-env"

echo ""
echo "Creating a $($python3 --version) environment for openViewMin in $envname/"
echo "Warning: Python package installation may fail for Python 3.10 (vtk not supported yet). If you have installed Python≥3.10, you may need to provide a path to Python 3.9 (or lower) within configure-openViewMin.sh\"."

$python3 -m venv --clear "$envname"
$python3 -m pip install pandas "PyQt5<5.14" pyvistaqt tqdm
chmod +x ./openViewMin.py

echo ""
echo "Now run"
echo "    source $envname/bin/activate"
echo "Then you can run openViewMin with"
echo "    ./openViewMin.py "
echo "from the command line, or "
echo "    import openViewMin "
echo "from a python interpreter."
echo "To use this Python environment in a Jupyter notebook:"
echo "    python3 -m pip install jupyterlab"
echo "    python3 -m ipykernel install --name="$envname" --user"
echo "    python3 -m jupyterlab"
echo "then select $envname as the kernel and start a Python notebook with "
echo "    import openViewMin"
echo ""
