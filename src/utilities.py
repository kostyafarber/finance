import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# yfinance a module which imports data from Yahoo Finance.
import yfinance as yf
from scipy.stats import norm
import talib as ta
from scipy.optimize import minimize
import os

np.set_printoptions(suppress=True)
# plt.style.use("seaborn-dark-palette")

plt.rcParams['figure.figsize'] = (20.00, 10.00)


# Importing Data
def grab_hedgefund_data(data_filepath):
    """
    :param data_filepath: The directory filepath
    :return: the hedge fund data set with all dates parsed and in decimal format.
    """
    hedge_fund = pd.read_csv(os.path.join(data_filepath, "edhec-hedgefundindices.csv"), parse_dates=True,
                             index_col=0) / 100
    return hedge_fund


def grab_kenfrench_data(data_filepath):
    """

    :param data_filepath: The directory filepath
    :return: Returns the Ken French industry Dataset, formatting and cleaned.
    """
    ken_french = pd.read_csv(os.path.join(data_filepath, "ind30_m_vw_rets.csv"), header=0, parse_dates=True,
                             index_col=0) / 100
    ken_french.index = pd.to_datetime(ken_french.index, format="%Y%m").to_period("M")


def grab_ind_n(data_filepath):
    """

    :param data_filepath: The directory filepath
    :return: returns the number of stocks in each industry over time, nicely formatted.
    """
    ind_n = pd.read_csv(os.path.join(data_filepath, "ind30_m_nfirms.csv"), index_col=0, parse_dates=True, header=0)
    ind_n.index = pd.to_datetime(ind_n.index, format="%Y%m").to_period("M")

    return ind_n


def grab_ind_size(data_filepath):
    """

    :param data_filepath: data_filepath
    :return: Industry Size for over time
    """
    ind_size = pd.read_csv(os.path.join(data_filepath, "ind30_m_size.csv"), index_col=0, parse_dates=True, header=0)

    # format dates
    ind_size.index = pd.to_datetime(ind_size.index, format="%Y%m").to_period("M")

    return ind_size


def grab_stock_close_data(tickers, start, end):
    """

    :param tickers: Takes in a list of stock tickers.
    :param start: the start date (in Y-M-D format)
    :param end: the end date
    :return: Returns a dataframe with the closing prices for the tickers specified.
    """
    if len(tickers) == 1:
        stock = yf.Ticker(tickers[0])
        data = stock.history(start=start, end=end)
        return pd.DataFrame(data["Close"])

    # use yfinance to initialise a Tickers object and grab the stock history for the specified dates.
    stocks = yf.download(tickers=tickers, period=period)

    # grab the closing columns from the dataframe and return the closing prices for each stock.
    columns = [x for x in stocks.columns if "Close" in x]
    stock_close = (stocks[columns]).dropna()

    return stock_close

# Returns
def returns(data):
    """
    Takes in a dataframe in which a close column exists and converts the returns into 1+r format and returns a dataframe 
    with those returns.
    """

    cols = ['Close']
    stock_return = data[cols].pct_change().dropna().rename({"Close": "Return"}, axis=1) + 1
    return stock_return


def annual_return(r, periods_per_year):
    """
    Returns the the annualised return for a dataset. Returns ARE NOT in (1+r) format.
    """

    return (((1 + r).prod()) ** (periods_per_year / r.shape[0])) - 1


def portfolio_return(weights, r):
    """Returns the Portfolio Return, given an dataframe of weights and a return matrix"""

    return weights.T @ r


# Volatility
def annual_vol(r, periods_per_year):
    """
    Returns the annualised volatility, must be in returns format (1+r)
    """
    return r.std() * np.sqrt(periods_per_year)


def portfolio_vol(weights, cov_max):
    """"
    Returns the portfolio volatility. Takes in an array of weights and a covariance matrix
    """
    return (weights.T @ cov_max @ weights) ** 0.5


# Plotting Functions
def plot_returns(data):
    """"Plots the returns from a pandas dataframe"""

    returns(data).cumprod().plot()


# Drawdown
def plot_drawdown(r):
    """Plots the drawdown given a series of returns."""

    wealth_index = 1000 * (1 + r).cumprod()
    previous_peaks = wealth_index.cummax()

    draw_down = (wealth_index - previous_peaks) / previous_peaks

    draw_down.plot()


# VaR
def historicVaR(r, level=5):
    """

    Calculates the historical VaR, givena  set of returns and a percentile.

    """

    if isinstance(r, pd.DataFrame):
        return r.aggregate(historicVaR, level=level)

    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)


def gaussianVaR(r, level=0.05, modified=False):
    """
    Returns the parametric Gaussian VaR for a set of returns.
    :Level is the confidence interval
    :modified if True then returns the modified VaR

    """
    z = norm.ppf(level)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )
    return -(r.mean() + z * r.std(ddof=0))


# Statistics
def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp / sigma_r ** 3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp / sigma_r ** 4


# Optimisers
def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }

    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
                        }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1, return_is_target),
                       bounds=bounds)
    return weights.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)
