import numpy as np
from scipy.sparse import coo_matrix

# Размеры матрицы и количество ненулевых элементов
rows, cols = 1000, 1000
nnz = 3000

# Генерация случайных уникальных индексов для ненулевых элементов
row_indices = np.random.randint(0, rows, size=nnz)
col_indices = np.random.randint(0, cols, size=nnz)

# Генерация случайных значений для ненулевых элементов
values = np.random.randn(nnz)

# Создание разреженной матрицы в формате COO
sparse_matrix = coo_matrix((values, (row_indices, col_indices)), shape=(rows, cols))

# Генерация случайного вектора размера 1000
vector = np.random.randn(cols)

# Умножение вектора на матрицу (вектор слева)
# В scipy.sparse умножение вектор * матрица можно сделать через dot
# Но чтобы умножить вектор (1 x cols) на матрицу (cols x rows), транспонируем матрицу
result = vector.dot(sparse_matrix.toarray())  # Преобразуем в dense для демонстрации

# Чтобы использовать sparse умножение без преобразования в dense, транспонируем матрицу и используем dot:
result_sparse = sparse_matrix.transpose().dot(vector)

# print(result_sparse)  # Результат умножения вектор * матрица, размер 1000