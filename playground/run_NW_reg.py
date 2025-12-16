import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from nwsig import NWSignatureRegressor, augment_with_time

def local_reg_nw_signature(X, y, test_size=0.2):
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X0_train = np.ones((X_train.shape[0], 1)) * X_train[0, :1]
    X0_test = np.ones((X_test.shape[0], 1)) * X_test[0, :1]
    nw_regressor = NWSignatureRegressor(
        signature_level=3,
        input_type='signatures',
        cv_folds=3,
        bandwidth_grid=np.linspace(0.001, 10, 50),
    )
    nw_regressor.fit(X_train, X0_train, y_train)
    y_pred_nw = nw_regressor.predict(X_test, X0_test)
    mse_nw = mean_squared_error(y_test, y_pred_nw)
    rmse_nw = np.sqrt(mse_nw)
    mae_nw = mean_absolute_error(y_test, y_pred_nw)
    r2_nw = r2_score(y_test, y_pred_nw)
    return nw_regressor, y_pred_nw, mse_nw, rmse_nw, mae_nw, r2_nw

