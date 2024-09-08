import numpy as np
from numpy import int64


def bilinear_interp_baseline(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This is the baseline implementation of bilinear interpolation without vectorization.
    - a is a ND array with shape [N, H1, W1, C], dtype = int64
    - b is a ND array with shape [N, H2, W2, 2], dtype = float64
    - return a ND array with shape [N, H2, W2, C], dtype = int64
    """
    # Get axis size from ndarray shape
    N, H1, W1, C = a.shape  # 也就是知道a的shape，得到输入的batch size，height，width，channel这些信息，作为三重循环的参数
    N1, H2, W2, _ = b.shape
    assert N == N1  # assert是一种“断言”，在程序运行时会实时检查，一旦N!=N1，就会报错

    # Do iteration
    res = np.empty((N, H2, W2, C), dtype=int64)  # 创建一个空的ndarray来存放结果
    for n in range(N):
        for i in range(H2):
            for j in range(W2):
                x, y = b[n, i, j]  # b的shape是[N,H2,W2,2]，它存储的是想要从原图中取的位置坐标，其中b[ , , ,0]是x坐标，b[ , , ,1]是y坐标
                x_idx, y_idx = int(np.floor(x)), int(np.floor(y))  # floor是向下取整，也就是去寻找距离(x,y)相邻的整点
                _x, _y = x - x_idx, y - y_idx
                # For simplicity, we assume all x are in [0, H1 - 1), all y are in [0, W1 - 1)
                res[n, i, j] = a[n, x_idx, y_idx] * (1 - _x) * (1 - _y) + a[n, x_idx + 1, y_idx] * _x * (1 - _y) + \
                               a[n, x_idx, y_idx + 1] * (1 - _x) * _y + a[n, x_idx + 1, y_idx + 1] * _x * _y
                # 这个计算完全就在套公式
    return res
