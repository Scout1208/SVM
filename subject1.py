import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#1a
# 生成數據
def generate_data(n_samples=50):
    X = np.random.rand(n_samples, 20)  # 20維度特徵
    y = np.sign(np.sum(X[:, :10], axis=1) - 5)  # 根據前10個特徵生成標籤
    return X, y

# 評估函數
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return train_acc, val_acc, test_acc

# 主程式
n_repeats = 5
results_knn = []
results_svm = []

for _ in range(n_repeats):
    X, y = generate_data(n_samples=1100)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1050, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1000, random_state=42)

    # KNN
    best_knn_acc = 0
    for k in [1, 3, 5, 7, 9]:
        knn = KNeighborsClassifier(n_neighbors=k)
        _, val_acc, _ = evaluate_model(knn, X_train, y_train, X_val, y_val, X_test, y_test)
        if val_acc > best_knn_acc:
            best_knn_acc = val_acc
            best_knn = knn
    train_acc, val_acc, test_acc = evaluate_model(best_knn, X_train, y_train, X_val, y_val, X_test, y_test)
    results_knn.append((train_acc, val_acc, test_acc))

    # SVM
    best_svm_acc = 0
    for C in [0.1, 1, 10, 100]:
        svm = SVC(kernel='linear', C=C)
        _, val_acc, _ = evaluate_model(svm, X_train, y_train, X_val, y_val, X_test, y_test)
        if val_acc > best_svm_acc:
            best_svm_acc = val_acc
            best_svm = svm
    train_acc, val_acc, test_acc = evaluate_model(best_svm, X_train, y_train, X_val, y_val, X_test, y_test)
    results_svm.append((train_acc, val_acc, test_acc))

print("KNN Results (Train, Validation, Test):", np.mean(results_knn, axis=0))
print("SVM Results (Train, Validation, Test):", np.mean(results_svm, axis=0))
#1b
import matplotlib.pyplot as plt

# 投影分析函數
def plot_histogram_of_projections(svm_model, X, y):
    # 計算樣本點到決策邊界的距離
    projections = svm_model.decision_function(X)
    
    # 將樣本點分為兩類
    class_1 = projections[y == 1]
    class_minus_1 = projections[y == -1]
    
    # 繪製投影值的直方圖
    plt.hist(class_1, bins=20, alpha=0.5, label="Class 1", color="blue")
    plt.hist(class_minus_1, bins=20, alpha=0.5, label="Class -1", color="red")
    plt.axvline(0, color='black', linestyle='dashed', label='Decision Boundary')
    plt.xlabel("Projection Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Projections")
    plt.legend()
    plt.savefig("Histogram of Projections.jpg")
    plt.show()

# 主程式 (使用問題1(a)中最佳SVM模型進行投影分析)
X, y = generate_data(n_samples=1100)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1050, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1000, random_state=42)

# 訓練最佳SVM模型
svm_model = SVC(kernel='linear', C=10)  # 使用問題1(a)中的最佳C值
svm_model.fit(X_train, y_train)

# 投影直方圖
plot_histogram_of_projections(svm_model, X_test, y_test)
