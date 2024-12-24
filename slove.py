import math
import numpy as np
from scipy.integrate import quad
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# 读取 JSON 文件
with open("./parameters.json", "r") as file:
    params = json.load(file)

# 提取参数
const_rho = params["const_rho"] # 常量反应性
k_rho = params["k_rho"] # 线性反应性的斜率
amplitude_rho = params["amplitude_rho"] # 周期反应性的振幅
omega_rho = params["omega"] # 周期反应性的周期
start_phi = params["start_phi"] # 周期反应性的起始相位
Delta = params["Delta"] # 中子一代时间
M = params["M"] # 先驱核分组数量
lambda_i = np.array(params["lambda_i"]) # 缓发中子衰变常数
beta_i = np.array(params["beta_i"]) # 每个群的缓发中子份额
beta = sum(beta_i) # 总缓发中子份额
Q = params["Q"] # 中子源项
h = params["h"] # 时间步长
T_max = params["T_max"] # 最大模拟时间
N_0 = params["N_0"]

B1 = 1 / 6
a_i = 1 / (1 + h * lambda_i / 2 + pow(h, 2) * np.power(lambda_i, 2) / 12)

h_over_Delta = h / (2 * Delta)
h2_over_Delta = pow(h, 2) * B1 / (2 * Delta)
sum_h = np.sum(a_i * beta_i * h) / (2 * Delta)
sum_h2 = np.sum(a_i * beta_i * pow(h, 2) * B1) / (2 * Delta)
sum_h_with_lambda = np.sum(a_i * beta_i * h * lambda_i) / (2 * Delta)
sum_h2_with_lambda = np.sum(a_i * beta_i * pow(h, 2) * B1 * lambda_i) / (2 * Delta)
sum_h_with_lambda2 = np.sum(a_i * beta_i * h * np.power(lambda_i, 2)) / (2 * Delta)
sum_h2_with_lambda2 = np.sum(a_i * beta_i * pow(h, 2) * B1 * np.power(lambda_i, 2)) / (2 * Delta)
sum_h_with_lambda3 = np.sum(a_i * beta_i * h * np.power(lambda_i, 3)) / (2 * Delta)
sum_h2_with_lambda3 = np.sum(a_i * beta_i * pow(h, 2) * B1 * np.power(lambda_i, 3)) / (2 * Delta)

def plot_results(N, C_i, time_steps):
    plt.figure(figsize=(12, 6))

    # 绘制中子密度 N
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, N, label='Neutron Density (N)')
    plt.xlabel('Time')
    plt.ylabel('Density')
    plt.title('Neutron Density Over Time')
    plt.legend()

    # 绘制先驱核浓度 C_i
    plt.subplot(1, 2, 2)
    for i in range(C_i.shape[1]):
        plt.plot(time_steps, C_i[:, i], label=f'Precursor Concentration (C_{i+1})')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Precursor Concentrations Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

def getRho(Time = 0):
    # 初始化 Rho 为零
    Rho = np.zeros_like(Time, dtype=float)  # 生成与 Time 同形状的零数组
    
    if const_rho != 0:
        Rho += const_rho
    
    if k_rho != 0:
        Rho += k_rho * Time
    
    if amplitude_rho != 0:
        # 使用 numpy 的向量化操作来计算余弦项
        Rho += amplitude_rho * np.cos(omega_rho * Time + start_phi)
    
    return Rho

def getGradRho(Time = 0):
    # 初始化 Rho 为零
    Rho = np.zeros_like(Time, dtype=float)  # 生成与 Time 同形状的零数组

    if k_rho != 0:
        Rho += k_rho

    if amplitude_rho != 0:
        # 使用 numpy 的向量化操作来计算余弦项
        Rho += -amplitude_rho * omega_rho * np.sin(omega_rho * Time + start_phi)

    return Rho

# 计算D(tau)的函数
def D(tau):
    return (getRho(tau) - beta) / Delta

def query_N_at_time(t, time_steps, N):
    """
    查询时间 t 对应的 N 值。返回最接近 t 的时间步的 N 值。
    
    参数:
    t (float): 需要查询的时间点
    time_steps (numpy.ndarray): 时间步长数组
    N (numpy.ndarray): 对应时间步的中子密度值数组
    
    返回:
    float: 时间点 t 对应的 N 值
    """
    # 找到离 t 最近的时间步
    idx = np.abs(time_steps - t).argmin()
    print(time_steps[idx])
    print(N[idx])

def get_next_step(N_n, C_i_n, t_n, t_n1):
    Rho_tn = getRho(t_n)
    Rho_tn1 = getRho(t_n1)
    RhoGrad_tn = getGradRho(t_n)
    RhoGrad_tn1 = getGradRho(t_n1)
    D_tn = D(t_n)
    D_tn1 = D(t_n1)
    HIC_i = np.sum(a_i * (1 - h * lambda_i / 2) * C_i_n)
    HIC_iWithLambda = np.sum(a_i * lambda_i * (1 - h * lambda_i / 2) * C_i_n)
    H2C_iWithLambda2 = np.sum(a_i * np.power(lambda_i, 2) * pow(h, 2) * B1 * C_i_n) / 2
    H2C_iWithLambda3 = np.sum(a_i * np.power(lambda_i, 3) * pow(h, 2) * B1 * C_i_n) / 2
    LambdaC_i = np.sum(lambda_i * C_i_n)
    C_iSum = np.sum(C_i_n)
    N_tn1 = (
        (
            (h_over_Delta * Rho_tn + 1 + h2_over_Delta * RhoGrad_tn - sum_h)
            + (h2_over_Delta * Rho_tn - sum_h2) * D_tn
            + sum_h2_with_lambda 
            + (sum_h_with_lambda + sum_h2_with_lambda * D_tn - sum_h2_with_lambda2) * (- h2_over_Delta * Rho_tn1 + sum_h2) / (1 + sum_h2_with_lambda)
        ) * N_n
        + (
            (Q * h - HIC_i - H2C_iWithLambda2 + C_iSum)
            + (h2_over_Delta * Rho_tn - sum_h2) * (LambdaC_i + Q)
            + (- h2_over_Delta * Rho_tn1 + sum_h2) / (1 + sum_h2_with_lambda) * (HIC_iWithLambda + H2C_iWithLambda3 + sum_h2_with_lambda * LambdaC_i + (1 + sum_h2_with_lambda) * Q)
        )
    )   
    N_tn1 /= (
        (h_over_Delta * Rho_tn1 - 1 - h2_over_Delta * RhoGrad_tn1 - sum_h - sum_h2_with_lambda)
        + (- h2_over_Delta * Rho_tn1 + sum_h2) * (D_tn1 + sum_h_with_lambda + sum_h2_with_lambda2) / (1 + sum_h2_with_lambda)
    )
    NGrad_tn1 = (
        (D_tn1 + sum_h_with_lambda + sum_h2_with_lambda2) * N_tn1
        + (sum_h_with_lambda + D_tn * sum_h2_with_lambda - sum_h2_with_lambda2) * N_n
        + (HIC_iWithLambda + H2C_iWithLambda3 + sum_h2_with_lambda * LambdaC_i + (1 + sum_h2_with_lambda) * Q)
    )
    NGrad_tn1 /= 1 + sum_h2_with_lambda
    C_i_tn1 = (
        (
            a_i * (h * beta_i+ pow(h, 2) * B1 * lambda_i * beta_i) / (2 * Delta) * N_tn1
            - a_i * pow(h, 2) * B1 * beta_i / (2 * Delta) * NGrad_tn1
        ) 
        +
        (
            a_i * (h * beta_i + pow(h, 2) * B1 * beta_i * D_tn - pow(h, 2) * B1 * lambda_i * beta_i) / (2 * Delta) * N_n
            + a_i * (1 - h * lambda_i / 2 + pow(h, 2) * B1 * np.power(lambda_i, 2) / 2) * C_i_n
            + a_i * pow(h, 2) * B1 * beta_i / (2 * Delta) * (LambdaC_i + Q)
        )
    )
    return N_tn1, C_i_tn1
    
def main():
    # 生成时间序列
    time_steps = np.arange(0, T_max, h)
    num_steps = len(time_steps)
    # 初始化变量
    N = np.zeros(num_steps)  # 中子密度
    C_i = np.zeros((num_steps, M))  # 先驱核浓度
    N[0] = N_0
    C_i[0, :] = N_0 * beta_i / (Delta * lambda_i)
    print("开始计算")
    for i in tqdm(range(num_steps - 1)):
        N[i + 1], C_i[i + 1, :] = get_next_step(N[i], C_i[i, :], time_steps[i], time_steps[i + 1])
    print("计算完成")
    # print(N)
    # print(C_i)
    # 绘制结果
    # query_N_at_time(0.4, time_steps, N)
    plot_results(N, C_i, time_steps)


if __name__ == "__main__":
    main()