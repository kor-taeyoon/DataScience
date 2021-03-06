import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def avg_filter(k, x_meas, x_avg):
    alpha = (k - 1) / k
    x_avg = alpha * x_avg + (1 - alpha) * x_meas
    
    return x_avg

def get_volt():
    v = np.random.normal(0, 4)  # v: 잡음 평균(노이즈)
    volt_mean = 14.4            # volt_mean: 측정하는 전압의 평균(기준값)
    volt_meas = volt_mean + v   # volt_meas: 식별가능한 전압 평균 [V] (기준값 + 잡음)

    return volt_meas

# Input parameters. (측정시간, 측정간격)
time_end = 10
dt = 0.2

# 값을 저장할 공간(array) 생성
time = np.arange(0, time_end, dt)
n_samples = len(time)
x_meas_save = np.zeros(n_samples)
x_avg_save = np.zeros(n_samples)

x_avg = 0
for i in range(n_samples):
    k = i + 1
    x_meas = get_volt()
    x_avg = avg_filter(k, x_meas, x_avg)
 
    x_meas_save[i] = x_meas
    x_avg_save[i] = x_avg

x_meas_save

plt.plot(time, x_meas_save, 'r*', label='Measured')
plt.plot(time, x_avg_save, 'b-', label='Average')
plt.legend(loc='upper left')
plt.title('Measured Voltages v.s. Average Filter Values')
plt.xlabel('Time [sec]')
plt.ylabel('Volt [V]')
plt.show()