from xgboost import XGBRegressor



if __name__=='__main__':
    xgb = XGBRegressor()
    xgb.fit(X_train_scaled, y_train)

    plotModelResults(xgb,
                     X_train=X_train_scaled,
                     X_test=X_test_scaled,
                     plot_intervals=True, plot_anomalies=True)