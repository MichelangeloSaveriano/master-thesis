from sklearn.linear_model import ElasticNet
from trading import *
from spreads import *


def compute_factor_residuals(X_train, factors_train, X_test=None, factors_test=None, alpha_elastic_net=3e-3):
    factor_model = ElasticNet(alpha=alpha_elastic_net, fit_intercept=False).fit(factors_train, X_train)
    train_residuals = X_train - factor_model.predict(factors_train)
    if (X_test is None) or (factors_test is None):
        return train_residuals
    test_residuals = X_test - factor_model.predict(factors_test)
    return train_residuals, test_residuals


def evaluate_split_returns(train_returns, test_returns, residuals_methods, spreads_methods, trading_methods):
    l = []
    test_returns_fwd = test_returns.shift(-1).iloc[:-1]
    for res_name, res_func in residuals_methods.items():
        X_train, X_test = res_func(train_returns, test_returns)

        l1 = []
        for spreads_name, spreads_func in spreads_methods.items():
            spreads_train, spreads_test, L, L_sqrt = compute_spreads(X_train, X_test, spreads_func)
            spreads_test = spreads_test.iloc[:-1]

            l2 = []
            for trading_name, trading_func in trading_methods.items():
                trading_results = compute_config_returns(spreads_train, spreads_test, test_returns_fwd,
                                                         trading_func, L_sqrt)
                trading_results['trading_method'] = trading_name
                l2.append(trading_results)

            df2 = pd.concat(l2)
            df2['spreads_method'] = spreads_name
            l1.append(df2)

        df1 = pd.concat(l1)
        df1['residuals_method'] = res_name
        l.append(df1)
    return pd.concat(l)


def evaluate_split_returns_from_idx(merged_df, factors_columns, idx_tuple,
                                    residuals_method_names, spreads_methods, trading_methods):
    x_idx, y_idx = idx_tuple
    returns_factors_merged_split_train = merged_df.loc[x_idx]
    returns_factors_merged_split_test = merged_df.loc[y_idx]
    # Extract factor columns
    factors_train = returns_factors_merged_split_train[factors_columns]
    factors_test = returns_factors_merged_split_test[factors_columns]
    # Extract return column
    returns_split_train = returns_factors_merged_split_train.drop(columns=factors_columns)
    returns_split_test = returns_factors_merged_split_test.drop(columns=factors_columns)
    # Remove NaN from return columns
    not_na_columns = ~returns_split_train.isna().any(axis=0)
    not_na_columns &= ~returns_split_test.isna().any(axis=0)
    train_returns = returns_split_train.loc[:, not_na_columns]
    test_returns = returns_split_test.loc[:, not_na_columns]

    # Define residuals methods
    residuals_methods = {}
    if 'returns' in residuals_method_names:
        residuals_methods['returns'] = lambda train, test: (train, test)
    if 'residuals' in residuals_method_names:
        residuals_methods['residuals'] = lambda train, test: compute_factor_residuals(train, factors_train,
                                                                                      test, factors_test)

    return evaluate_split_returns(train_returns, test_returns, residuals_methods, spreads_methods, trading_methods)


