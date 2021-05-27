# PREDICTIVE MAINTENANCE 
[![DOI](https://zenodo.org/badge/364837516.svg)](https://zenodo.org/badge/latestdoi/364837516) [![Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/judithspd/predictive-maintenance/master)

Access to the Binder-compatible repo by clicking the blue badge above. The `environment.yml` file list all Python libraries on which the different notebooks
depend.

---
Baseline study on the development of predictive maintenance techniques using open data. Classic predictive maintenance problems will be studied:
- Classification of the signal as healthy or faulty.
- Failure type classification.
- Prediction of time to failure.

## Data used
Different data obtained from open repositories will be used:
- Failure classification (healthy, inner race or outer race). In this case two different signal types are used:
    - Signals captured under constant speed conditions: for this purpose, we have used data from [CWRU](https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website)
    - Signals captured under variable speed conditions: in this case, we hace use data from [Bearing Vibration Data under Time-varying Rotational Speed Conditions](https://data.mendeley.com/datasets/v43hmbwxpm/1)
- Prediction of time to failure. In this case we have used the data set called _Bearing Data Set_ which is provided by the Center for Intelligent Maintenance Systems (IMS), University of Cincinnati, and is available in the [NASA Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

## Libraries used 
In this analysis we will use Python (version higher than 3.8) and different libraries: 
### Scientific computing:
- [NumPy](https://numpy.org/): Library specialized in numerical analysis. Support for the creation of vectors and matrices, as well as for the use of different mathematical functions.
- [Pandas](https://pandas.pydata.org/): Tool for data analysis and manipulation. Written as an extension of NumPy.
- [SciPy](https://www.scipy.org/): Library with different mathematical tools and algorithms.
### Machine learning and data analysis:
- [Scikit-learn](https://scikit-learn.org/stable/): Library wich contains tools for data analysis, especially used in our case for the use of machine learning models. 
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/): Gradient boosting framework which uses tree-based learning algorithms.
- [Keras](https://keras.io/): Open Source Neural Networks library written in Python.
- [Hyperopt](http://hyperopt.github.io/hyperopt/): Hyper-parameter optimization.
- [Joblib](https://joblib.readthedocs.io/en/latest/): In our case, library used to export models for forecasting.
### Data visualization:
- [Matplotlib](https://matplotlib.org/): Library for creating static, animated and interactive visualizations in Python.
- [Seaborn](https://seaborn.pydata.org/): High-level interface for creating visualizations.
### Time series analysis:
- [PyHHT](https://pyhht.readthedocs.io/en/latest/tutorials.html): Module that implements the Hilbert-Huang Transform (HHT).

## Recommendations for installation:
1. Clone the repository
2. Create a conda or Python virtual environment. We recommend to create a Python virtual environment, as well as not to use a Python version higher than 3.9 because it has not been tested with this project.
```
python3.8 -m venv .venv
```
3. Activate virtual environment:
```
source .venv/Scripts/activate.bat
```
4. Install requirements.txt
```
pip install -r requirements.txt
```
