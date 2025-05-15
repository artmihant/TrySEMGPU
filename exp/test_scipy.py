import torch

# Размеры матрицы и количество ненулевых элементов
rows, cols = 1000, 1000
nnz = 3000

# Генерация случайных индексов для ненулевых элементов (координаты в формате COO)
indices = torch.randint(0, rows, (1, nnz)), torch.randint(0, cols, (1, nnz))
indices = torch.cat(indices, dim=0)  # shape (2, nnz)

# Генерация случайных значений для ненулевых элементов
values = torch.randn(nnz)

# Создание разреженной матрицы в формате COO
sparse_matrix = torch.sparse_coo_tensor(indices, values, (rows, cols))

# Генерация случайного вектора размера 1000
vector = torch.randn(cols)

# Чтобы умножить вектор на матрицу (вектор слева), преобразуем вектор в 2D (1 x cols)
vector_2d = vector.unsqueeze(0)  # shape (1, cols)

# Умножение: вектор (1 x cols) на матрицу (cols x rows) - для этого транспонируем матрицу
# Но у нас матрица 1000x1000, умножение вектор (1x1000) на матрицу (1000x1000) даёт (1x1000)
# В PyTorch sparse.mm принимает (sparse_matrix, dense_matrix), где sparse_matrix должен быть 2D sparse
# Для умножения вектор * матрица, транспонируем матрицу и умножаем справа:
result = torch.sparse.mm(sparse_matrix.transpose(0, 1), vector_2d.t())

# Приводим результат к 1D вектору
result = result.squeeze()

print(result)