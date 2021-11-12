#!/bin/bash
python3=python3.9 # or substitute your own path to python<3.10

echo ""
echo "Creating a $($python3 --version) environment for openViewMin."

echo "Warning: Python package installation may fail for Python 3.10 (vtk not supported yet). If you have installed Python 3.10, you may need to provide a path to Python 3.9 (or lower) within configure.sh\"."

$python3 -m venv --system-site-packages --clear "openViewMin-env"

openViewMin-env/bin/$python3 -m pip install pandas "PyQt5<5.14" pyvistaqt tqdm

echo ""
echo "Now run"
echo "    source openViewMin-env/bin/activate"
echo ""
echo "Then you can run \"python3 ./openViewMin.py\" from the command line,"
echo "or \"import openViewMin\" from a python interpreter or Jupyter notebook."
echo ""
