from prettytable import PrettyTable
from scipy.stats.distributions import chi2
from scipy.stats import f
import pandas as pd
import numpy as np


def SpanningTest(benchmark_assets, test_assets):

    # benc_asset = pd.read_csv('benchmark_asset.csv', index_col='Date', parse_dates=['Date'])[-25:]
    # test_asset = pd.read_csv('test_asset.csv', index_col='Date', parse_dates=['Date'])[-25:]

    benc_asset = benchmark_assets
    test_asset = test_assets

    T = len(benc_asset)
    K = len(benc_asset.columns)
    N = len(test_asset.columns)

    benc_asset_ret = benc_asset.pct_change()[1:]
    test_asset_ret = test_asset.pct_change()[1:]

    benc_asset_ret.insert(loc=0, column='intercept', value=1)

    Y = test_asset_ret.values
    X = benc_asset_ret.values
    B_hat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
    Summation_hat =  (Y - X.dot(B_hat)).T.dot(Y - X.dot(B_hat)) / T

    A = np.array([
                    [1] + [0]*K,
                    [0] + [-1]*K,
                                 ])

    C = np.array([
                    [0]*N,
                    [1]*N,
                            ])

    Theta_hat = A.dot(B_hat) + C
    G_hat = T * A.dot(np.linalg.inv(X.T.dot(X))).dot(A.T)
    H_hat = Theta_hat.dot(np.linalg.inv(Summation_hat)).dot(Theta_hat.T)

    result = np.linalg.eigvals(H_hat.dot(np.linalg.inv(G_hat)))
    lambda1 = max(result)
    lambda2 = min(result)

    U = 1 / ( ( 1+lambda1 ) * (1+lambda2) )

    F_value = round( (1 / U**0.5 - 1) * ( (T-K-N)/N ) , 3)
    LR = round(T * (np.log(1+lambda1) +  np.log(1+lambda2)), 3)
    LM = round(T * ( lambda1 / (1+lambda1)  + lambda2 / (1+lambda2) ) , 3)
    Wald = round(T * (lambda1 + lambda2), 3)

    F_pvalue = round(f.sf(F_value, 2*N, T-K-N), 3)
    LR_pvalue = round(chi2.sf(LR, 2*N), 3)
    LM_pvalue = round(chi2.sf(Wald, 2*N), 3)
    Wald_pvalue = round(chi2.sf(LM, 2*N), 3)


    output_table = PrettyTable(['F-test', ' LR ', ' LM ', 'Wald'])

    output_table.add_row(['{:.3f}'.format(F_value), '{:.3f}'.format(LR),
                          '{:.3f}'.format(LM), '{:.3f}'.format(Wald)])

    output_table.add_row(['({:.3f})'.format(F_pvalue), '({:.3f})'.format(LR_pvalue), 
                          '({:.3f})'.format(LM_pvalue), '({:.3f})'.format(Wald_pvalue)])

    print(output_table)
