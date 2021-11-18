import numpy as np
import matplotlib.pyplot as plt
from scipy import io
 
# 초음파 측정 데이터

input_mat = io.loadmat('./Filtering/data/SonarAlt.mat')

def low_pass_filter(x_meas, x_esti):
    """저주파 통과 필터 함수를 통해 초음파 계산"""
    x_esti = alpha * x_esti + (1 - alpha) * x_meas
    return x_esti
    
 
def get_sonar(i):
    """초음파 측정값 반환"""
    z = input_mat['sonarAlt'][0][i]  # input_mat['sonaralt']: (1, 1501)

    return z

# Input parameters.
alpha = 0.9
n_samples = 500
time_end = 10


dt = time_end / n_samples
time = np.arange(0, time_end, dt)
x_meas_save = np.zeros(n_samples)

x_esti_save = np.zeros(n_samples)

x_esti = None
for i in range(n_samples):
    x_meas = get_sonar(i)
    if i == 0:
        x_esti = x_meas
    else:
        x_esti = low_pass_filter(x_meas, x_esti)
 
    x_meas_save[i] = x_meas

    x_esti_save[i] = x_esti

plt.plot(time, x_meas_save, 'r*', label='Measured')
plt.plot(time, x_esti_save, 'b-', label='Low-pass Filter')
plt.legend(loc='upper left')
plt.title('Measured Altitudes v.s. LPF Values')
plt.xlabel('Time [sec]')
plt.ylabel('Altitude [m]')
plt.show()