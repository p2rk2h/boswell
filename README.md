# boswell
Extends Alan Wilhelm’s [botwell/archive/botwell_test.py](https://github.com/alanwilhelm/botwell/tree/main/archive) for fast multi-LLM response collection via OpenRouter. Automatically scores answers using the unbiased (or normalized) Boswell Quotient (accuracy, consistency, speed), assigns A+–F letter grades, and generates ranked, styled result tables exported as high-quality PNGs.  A typical multi-LLM query on OpenRouter run for 10 LLM's cost no more than $2.00 to run. 

Recommended python environment:
python version >= 3.10
pandas version >= 2.1
dataframe_image version >= 0.2,7

Best to create a new virtual environment:
Anaconda:
1) $ conda create -n py310 python=3.10&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#  or higher (at this writiing, 3.10 is most stable for data-science related packages)
2) $ conda activate py310
3) $ conda install pandas numpy matplotlib ipython seaborn ipython jupyter scipy
4) #use pip for PyPI packages:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ pip install dataframe_image selenium

initial Checkin files:
1) boswell_test.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#  slight enhancement of Alan's original botwell_test.py
2) LetterGrade.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#  slight improvement of [LetterGrade checkin](https://github.com/p2rk2h/LetterGrades)  
3) boswellQuotientTable.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#  tabulates original LLM assigned grades, calculates normalized grades, and tabulates with final weighted Boswell Quotient table; all 3 tables are styled and exportable as png files.
