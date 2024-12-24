from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
# 2a
# 生成數據
def generate_regression_data(n_samples=100, n_features=6):
    X = np.random.uniform(-1.5, 1.5, size=(n_samples, n_features))
    y = (10 * np.sin(2 * np.pi * X[:, 0]) +
         20 * (X[:, 1] - 0.5) ** 2 +
         5 * X[:, 2] + 5 * X[:, 3] +
         0.05 * X[:, 4] + 0.05 * X[:, 5])
    return X, y

# SVM回歸
def tune_svm(X_train, y_train, X_val, y_val):
    best_model = None
    best_mse = float('inf')
    for epsilon in [0, 2, 4, 6, 8]:
        for gamma in [2 ** i for i in range(-5, 6)]:
            svr = SVR(kernel='rbf', C=1.0, epsilon=epsilon, gamma=gamma)
            svr.fit(X_train, y_train)
            y_val_pred = svr.predict(X_val)
            mse = mean_squared_error(y_val, y_val_pred)
            if mse < best_mse:
                best_mse = mse
                best_model = svr
    return best_model

# 主程式
X_train, y_train = generate_regression_data(100)
X_val, y_val = generate_regression_data(100)
X_test, y_test = generate_regression_data(800)

best_svm = tune_svm(X_train, y_train, X_val, y_val)
y_test_pred = best_svm.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_nrms = np.sqrt(test_mse) / np.std(y_test)

print(f"Test MSE: {test_mse:.4f}, Test NRMS: {test_nrms:.4f}")
#2b
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# PPR方法 (使用多項式擬合模擬PPR)
def tune_ppr(X_train, y_train, X_val, y_val):
    best_model = None
    best_mse = float('inf')
    for degree in range(1, 6):  # 假設最多5個投影項
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = model
    return best_model

# 主程式
X_train, y_train = generate_regression_data(100)
X_val, y_val = generate_regression_data(100)
X_test, y_test = generate_regression_data(800)

# PPR最佳模型
best_ppr = tune_ppr(X_train, y_train, X_val, y_val)
y_test_pred_ppr = best_ppr.predict(X_test)
test_mse_ppr = mean_squared_error(y_test, y_test_pred_ppr)

# SVM最佳模型
best_svm = tune_svm(X_train, y_train, X_val, y_val)
y_test_pred_svm = best_svm.predict(X_test)
test_mse_svm = mean_squared_error(y_test, y_test_pred_svm)

# 結果比較
print(f"PPR Test MSE: {test_mse_ppr:.4f}")
print(f"SVM Test MSE: {test_mse_svm:.4f}")
