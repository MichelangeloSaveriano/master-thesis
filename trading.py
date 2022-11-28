import pandas as pd
import numpy as np


def quantiles_std_trading_rule(q, w, gamma, trading_type):
    def q_std_trading(train_spreads, test_spreads):
        _q = q
        rolling_spreads = pd.concat([train_spreads, test_spreads]).rolling(w, min_periods=1).mean()
        train_rolling_spreads = rolling_spreads.loc[train_spreads.index]
        test_rolling_spreads = rolling_spreads.loc[test_spreads.index]
        rolling_spreads_std = rolling_spreads.rolling(len(rolling_spreads), min_periods=1).std().loc[test_spreads.index]
        

        if trading_type == 'both':
            _q /= 2
        lower, upper = np.quantile(test_rolling_spreads,
                                   (_q, 1 - _q), axis=1).reshape((2, -1, 1))

        trading_mask = np.zeros_like(test_rolling_spreads)

        if (trading_type == 'short') or (trading_type == 'both'):
            q_mask = (test_rolling_spreads >= upper) | (q == 1)
            std_mask = (test_spreads - test_rolling_spreads >= gamma * rolling_spreads_std) | (gamma == 0)
            short_mask = q_mask & std_mask
            short_mask /= short_mask.sum(axis=1).values.reshape((-1, 1))
            short_mask = short_mask.fillna(0)
            trading_mask = trading_mask + short_mask

        if (trading_type == 'long') or (trading_type == 'both'):
            q_mask = (test_rolling_spreads <= lower) | (q == 1)
            std_mask = (test_spreads - test_rolling_spreads <= -gamma * rolling_spreads_std) | (gamma == 0)
            long_mask = q_mask & std_mask
            long_mask /= long_mask.sum(axis=1).values.reshape((-1, 1))
            long_mask = long_mask.fillna(0)
            trading_mask = trading_mask + long_mask

        return trading_mask

    return q_std_trading


def compute_config_returns(train_spreads, test_spreads, returns_fwd, trading_method, L_sqrt,
                           columns_trading_mask=None, benchmark=True):
    trading_mask = trading_method(train_spreads, test_spreads)

    A_sqrt = np.diag(np.diag(L_sqrt)) - L_sqrt
    row_sum = A_sqrt.sum(axis=1, keepdims=True)
    row_sum += np.abs(row_sum) < 1e-6
    A_sqrt = A_sqrt / row_sum
    compl_trading_mask = trading_mask @ A_sqrt
    compl_trading_mask.columns = trading_mask.columns

    full_trading_returns = returns_fwd * trading_mask
    compl_full_trading_returns = returns_fwd * compl_trading_mask

    if columns_trading_mask is not None:
        full_trading_returns = full_trading_returns.loc[:, columns_trading_mask]
        compl_full_trading_returns = compl_full_trading_returns.loc[:, columns_trading_mask]

    trading_returns = full_trading_returns.sum(axis=1)
    compl_trading_returns = compl_full_trading_returns.sum(axis=1)

    trading_returns_df = pd.DataFrame({'avg_month_returns': trading_returns,
                                       'compl_avg_month_returns': compl_trading_returns,
                                       'pairs_avg_month_returns': (trading_returns + compl_trading_returns) / 2,
                                       'n_positions_avg': (trading_mask != 0).sum(axis=1),
                                       'n_positions_compl': (compl_trading_mask != 0).sum(axis=1),
                                       'n_positions_pair': ((trading_mask + compl_trading_mask) != 0).sum(axis=1),
                                       }).reset_index()

    if not benchmark:
        return trading_returns_df

    trading_returns_df['benchmark'] = False

    full_benchmark_returns = returns_fwd

    if columns_trading_mask is not None:
        full_benchmark_returns = full_benchmark_returns.loc[:, columns_trading_mask]

    benchmark_returns = full_benchmark_returns.mean(axis=1)

    benchmark_returns_df = pd.DataFrame({'avg_month_returns': benchmark_returns,
                                         'compl_avg_month_returns': benchmark_returns,
                                         'pairs_avg_month_returns': benchmark_returns,
                                         'n_positions_avg': test_spreads.shape[1],
                                         'n_positions_compl': test_spreads.shape[1],
                                         'n_positions_pair': test_spreads.shape[1],
                                         'benchmark': True
                                         }).reset_index()
    return pd.concat([trading_returns_df, benchmark_returns_df])