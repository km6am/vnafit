@echo off
REM Windows twin of ./run -- launch with an interpreter that has the deps.
setlocal
set HERE=%~dp0
set ENVNAME=vnafit
for %%P in (
  "%USERPROFILE%\miniconda3\envs\%ENVNAME%\python.exe"
  "%USERPROFILE%\anaconda3\envs\%ENVNAME%\python.exe"
  "%USERPROFILE%\miniforge3\envs\%ENVNAME%\python.exe"
  "%HERE%.venv\Scripts\python.exe"
  "%CONDA_PREFIX%\python.exe"
) do (
  if exist %%P (
    %%P -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && ^
    %%P -c "import numpy,scipy,matplotlib,serial,tkinter" >nul 2>&1 && (
      set PYTHONPATH=%HERE%;%PYTHONPATH%
      %%P -m vnafit.gui %*
      exit /b
    )
  )
)
echo No Python ^>= 3.10 found with numpy, scipy, matplotlib, pyserial and tkinter.
echo Create the environment:
echo     conda env create -f "%HERE%environment.yml"
exit /b 1
