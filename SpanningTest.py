import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as solver
from scipy.stats.distributions import chi2
from functools import reduce
from prettytable import PrettyTable
from scipy.stats import f


class Investment:

    def __init__(self):

        pass


    def EfficientFrontierComparison(self, portfolio1, portfolio2):

        def ori_standard_deviation(weights):
            return np.sqrt(reduce(np.dot, [weights, ori_covariance_matrix, weights.T]))

        def com_standard_deviation(weights):
            return np.sqrt(reduce(np.dot, [weights, com_covariance_matrix, weights.T]))

        df1 = portfolio1
        df2 = portfolio2

        # K assets portfolio
        ori_portfolio_assets_num = len(df1.columns)
        ori_portfolio_ind_returns = df1.pct_change()[1:]
        ori_portfolio_ind_weights = [1/ori_portfolio_assets_num for i in range(ori_portfolio_assets_num)]

        ori_covariance_matrix = ori_portfolio_ind_returns.cov() * 12
        ori_assets_expected_return = ori_portfolio_ind_returns.mean() * 12

        # K + N assets portfolio
        combined_df = pd.concat([df1, df2], axis = 1)
        com_portfolio_assets_num = len(combined_df.columns)
        com_portfolio_ind_returns = combined_df.pct_change()[1:]
        com_portfolio_ind_weights = [1/com_portfolio_assets_num for i in range(com_portfolio_assets_num)]

        com_covariance_matrix = com_portfolio_ind_returns.cov() * 12
        com_assets_expected_return = com_portfolio_ind_returns.mean() * 12

        # K assets m.v.p
        bounds = tuple((0, 1) for x in range(ori_portfolio_assets_num))
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
        ori_minimize_variance = solver.minimize(ori_standard_deviation, x0=ori_portfolio_ind_weights,
                                            constraints=constraints, bounds=bounds)
        ori_mvp_risk = ori_minimize_variance.fun
        ori_mvp_return = sum(ori_minimize_variance.x * ori_assets_expected_return)

        # K + N assets m.v.p
        bounds = tuple((0, 1) for x in range(com_portfolio_assets_num))
        com_minimize_variance = solver.minimize(com_standard_deviation, x0=com_portfolio_ind_weights,
                                            constraints=constraints, bounds=bounds)
        com_mvp_risk = com_minimize_variance.fun
        com_mvp_return = sum(com_minimize_variance.x * com_assets_expected_return)

        # set bound for return will be tried to fit in lambda
        upper_bound = max(ori_assets_expected_return)
        interval = (upper_bound - ori_mvp_return) / 100
        lower_bound = ori_mvp_return - 50*interval
        efficient_frontier_return_range = np.arange(lower_bound, upper_bound, interval)
        
        ori_risk_list = []
        ori_return_list = []
        com_risk_list = []
        com_return_list = []

        # set return goal and minimum the risk
        for ret_goal in efficient_frontier_return_range:

            # original portfolio and combined portfolio should deal with different cases
            for por_ret in [ori_assets_expected_return, com_assets_expected_return]:

                if len(por_ret) == len(ori_assets_expected_return):

                    bounds = tuple((0, 1) for x in range(ori_portfolio_assets_num))
                    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                                   {'type': 'eq', 'fun': lambda x: sum(x * por_ret) - ret_goal}]

                    efficient_frontier = solver.minimize(ori_standard_deviation, x0=ori_portfolio_ind_weights, constraints=constraints, bounds=bounds)
                    ori_risk_list.append(efficient_frontier.fun)
                    ori_return_list.append(ret_goal)

                elif len(por_ret) == len(com_assets_expected_return):

                    bounds = tuple((0, 1) for x in range(com_portfolio_assets_num))
                    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                                   {'type': 'eq', 'fun': lambda x: sum(x * por_ret) - ret_goal}]

                    efficient_frontier = solver.minimize(com_standard_deviation, x0=com_portfolio_ind_weights, constraints=constraints, bounds=bounds)
                    com_risk_list.append(efficient_frontier.fun)
                    com_return_list.append(ret_goal)

        # visualization result
        fig = plt.figure(figsize = (6, 4.5))
        fig.subplots_adjust(top=0.85)
        ax = fig.add_subplot()

        # plot efficient frontier
        ax.plot(ori_risk_list, ori_return_list, linewidth=2, linestyle='--', color='#e05151', label='K assets')
        ax.plot(com_risk_list, com_return_list, linewidth=2, linestyle='--', color='#548cd1', label='K+N assets')

        # plot M.V.P.
        ax.plot(ori_mvp_risk, ori_mvp_return, 'o', color='r', markerfacecolor='r',  markersize=10, label='K assets - M.V.P. ')
        ax.plot(com_mvp_risk, com_mvp_return, 'o', color='b', markerfacecolor='b',  markersize=10, label='K+N assets - M.V.P. ')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('Efficient Frontier Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Risk')
        ax.set_ylabel('Return')
        ax.legend(loc='best')

        plt.show()


    def SpanningTest(self, benchmark_assets, test_assets):

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

        result = [F_value, F_pvalue, LR, LR_pvalue, LM, LM_pvalue, Wald, Wald_pvalue]

        output_table = PrettyTable(['F-test', ' LR ', ' LM ', 'Wald'])

        output_table.add_row(['{:.3f}'.format(F_value), '{:.3f}'.format(LR),
                              '{:.3f}'.format(LM), '{:.3f}'.format(Wald)])

        output_table.add_row(['({:.3f})'.format(F_pvalue), '({:.3f})'.format(LR_pvalue), 
                              '({:.3f})'.format(LM_pvalue), '({:.3f})'.format(Wald_pvalue)])

        print(output_table)

        return result
    
