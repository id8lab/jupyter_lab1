# Jupyter Tutorials
# 1 Jupyter Mechanics

TO USE the notebook altair_R in Jupyter Lab please do the following:

     upload these 5 files:
        pd_rbase.py
	boxplot.py
	describe.py
	common.py
	docstrings.py

     upload the necessary data files: for example iris.csv

     in the terminal of jupyter_lab:
        pip install rpy2
        pip install pandas_datareader
        pip install pdvega
        pip install qgrid
        pip install colorcet
        pip install stemgraphic
        pip install statutils
        pip install vega_datasets

If you are working your own laptop or PC:
   put the .py files in the same directory as your notebooks
   and install the 'pip' files on your machine.

WARNING: The cells that use 'vgplot' will not work for you: they need a change
	 in the python module vega3 which I can make on my laptop.
	 I can't of course change the public module.
	 This only affects about half a dozen cells. All the rest are ok.

DATA: you will need to change the path to data files( ex iris.csv )
