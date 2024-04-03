import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Крок 1: Генеруємо випадковий набір даних
lower_bound = 0
upper_bound = 100
random_data = np.random.randint(lower_bound, upper_bound, size=1000)

# Крок 2: Нормалізуємо значення
normalized_data = (random_data - np.mean(random_data)) / np.std(random_data)

# Крок 3: Розділяємо існуючі записи на навчальну і тестові вибірки
X_train, X_test = train_test_split(normalized_data, test_size=0.2, random_state=42)

# Крок 4: Навчаємо KNN-регресор з різними значеннями K та оцінюємо якість
best_k = None
best_mse = float('inf')
mse_values = []
for k in range(1, 11):
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train.reshape(-1, 1), X_train)  # Навчання на навчальних даних
    predictions = knn_regressor.predict(X_test.reshape(-1, 1))  # Прогнозування на тестових даних
    mse = mean_squared_error(X_test, predictions)  # Розрахунок середньоквадратичної помилки
    mse_values.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_k = k

print(f"Найкраще значення K: {best_k} з MSE = {best_mse}")

# Крок 6: Візуалізація отриманих результатів
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), mse_values, marker='o', linestyle='-', color='b')
plt.title('Залежність MSE від значення K')
plt.xlabel('Значення K')
plt.ylabel('MSE')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
