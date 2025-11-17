import numpy as np
import os

def gen_golden_data():
    M = 64
    K = 32
    N = 64*8
    E = 4

    Y = np.zeros((M, N), dtype=np.float16)

    # Group_list: 每个专家的行数分配
    Group_list = np.array([15, 12, 19, 18], dtype=np.int64)
    
    # 生成测试数据
    X = np.random.randint(-8, 7, (M, K), dtype=np.int8)  # 模拟int4范围
    W = np.random.randint(-8, 7, (E, K, N), dtype=np.int8)
    X_Scale = np.random.randn(M).astype(np.float32)
    W_Scale = np.random.randn(E, N).astype(np.float32)
    
    current_row = 0
    for expert_idx in range(E):
        # 获取当前专家的行数
        m_i = Group_list[expert_idx]
        
        if m_i == 0:
            continue
            
        # 从X中截取当前专家的数据
        x_i = X[current_row:current_row + m_i, :]  # shape (m_i, K)
        
        # 获取当前专家的权重
        w_i = W[expert_idx]  # shape (K, N)
        
        # int4 * int4 矩阵乘法，结果为int32
        # 注意：NumPy没有原生int4类型，我们假设输入已经是适当缩放的int8
        # 在实际应用中，可能需要先进行int4到int8的转换
        matmul_int32 = np.dot(x_i.astype(np.int32), w_i.astype(np.int32))
        
        # 转换为float32
        matmul_float32 = matmul_int32.astype(np.float32)
        
        # 应用缩放因子
        x_scale_i = X_Scale[current_row:current_row + m_i]  # shape (m_i,)
        w_scale_i = W_Scale[expert_idx]  # shape (N,)
        
        # 广播缩放因子并应用
        scale_matrix = x_scale_i[:, np.newaxis] * w_scale_i[np.newaxis, :]  # shape (m_i, N)
        scaled_result = matmul_float32 * scale_matrix
        
        # 存储到输出张量
        Y[current_row:current_row + m_i, :] = scaled_result.astype(np.float16)
        
        current_row += m_i

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    X.tofile("./input/x.bin")
    W.tofile("./input/w.bin")
    X_Scale.tofile("./input/x_scale.bin")
    W_Scale.tofile("./input/w_scale.bin")
    Group_list.tofile("./input/group_list.bin")

    Y.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
