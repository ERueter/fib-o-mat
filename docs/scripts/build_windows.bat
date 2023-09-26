@echo off

SET versions=py38_win32 py38_win64 py39_win32 py39_win64

@RD /S /Q "_skbuild"

for %%a in (%versions%) do (
    echo %%a

    call conda env create --name %%a --file scripts/%%a.yaml
    call conda activate %%a
    call python -c "import sys; print('Python {0:s} on {1:s}'.format(sys.version, sys.platform))"
    call pip install --upgrade -r dev-requirements.txt
    call python setup.py bdist_wheel
    call conda deactivate
    call conda env remove --name %%a

    @RD /S /Q "_skbuild"
)


