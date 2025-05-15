import numpy as np
from numba import cuda, float32
from numpy.typing import NDArray
from numba.cuda.cudadrv.devicearray import DeviceNDArray

@cuda.jit
def sum_reduce_kernel(d_in: DeviceNDArray, d_out: DeviceNDArray) -> None:
    # Shared память для редукции внутри блока
    sdata = cuda.shared.array(512, dtype=float32)

    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x * 2 + tid

    # Загружаем два элемента на поток (если есть)
    s = 0.0
    if i < d_in.size:
        s += d_in[i]
    if i + cuda.blockDim.x < d_in.size:
        s += d_in[i + cuda.blockDim.x]

    sdata[tid] = s
    cuda.syncthreads()

    # Редукция в shared памяти
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            sdata[tid] += sdata[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # Записываем результат блока в глобальную память
    if tid == 0:
        d_out[cuda.blockIdx.x] = sdata[0]

def gpu_sum(arr: NDArray) -> float:
    threads_per_block = 512
    blocks_per_grid = (arr.size + threads_per_block * 2 - 1) // (threads_per_block * 2)

    d_in = cuda.to_device(arr.astype(np.float32))
    d_out = cuda.device_array(blocks_per_grid, dtype=np.float32)

    sum_reduce_kernel[blocks_per_grid, threads_per_block](d_in, d_out)

    # Если осталось больше одного блока, повторяем редукцию рекурсивно
    while blocks_per_grid > 1:
        temp_in = d_out
        blocks_per_grid = (blocks_per_grid + threads_per_block * 2 - 1) // (threads_per_block * 2)
        d_out = cuda.device_array(blocks_per_grid, dtype=np.float32)
        sum_reduce_kernel[blocks_per_grid, threads_per_block](temp_in, d_out)

    # Копируем результат обратно на CPU
    result = d_out.copy_to_host()
    return result[0]

# Пример использования
if __name__ == "__main__":
    N = 10_000_000
    arr = np.ones(N, dtype=np.float32)  # массив из единиц, сумма должна быть N

    total = gpu_sum(arr)
    print(f"Total sum: {total}")