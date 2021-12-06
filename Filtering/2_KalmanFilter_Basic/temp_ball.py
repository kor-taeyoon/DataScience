import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.linalg import inv
from skimage.metrics import structural_similarity
np.random.seed(0)



def kalman_filter(z_meas, x_esti, P):
    # 측정값, 추정값, 오차공분산
    
    # (1) 예측
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q
 
    # (2) 칼만 이득 계산
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)
 
    # (3) 추정
    x_esti = x_pred + K @ (z_meas - H @ x_pred)
 
    # (4) 오차공분산 계산
    P = P_pred - K @ H @ P_pred
 
    return x_esti, P



def get_ball_pos():
    v1 = np.random.normal(-20,20);
    v2 = np.random.normal(-20,20);
    
    xpos_meas = 100+v1
    ypos_meas = 100+v2
        
    return np.array([xpos_meas, ypos_meas])



# Input parameters.
n_samples = 200
dt = 1



# Initialization for system model.
# Matrix: A, H, Q, R, P_0
# Vector: x_0
A = np.array([[ 1, dt,  0,  0],
              [ 0,  1,  0,  0],
              [ 0,  0,  1, dt],
              [ 0,  0,  0,  1]])
H = np.array([[ 1,  0,  0,  0],
              [ 0,  0,  1,  0]])
Q = 1.0 * np.eye(4)
R = np.array([[50,  0],
              [ 0, 50]])



# Initialization for estimation.
x_0 = np.array([0, 0, 0, 0])  # (x-pos, x-vel, y-pos, y-vel) by definition in book. / 추정값 초기위치
P_0 = 100 * np.eye(4)



xpos_meas_save = np.zeros(n_samples)
ypos_meas_save = np.zeros(n_samples)
xpos_esti_save = np.zeros(n_samples)
ypos_esti_save = np.zeros(n_samples)



# 칼만필터 실행
x_esti, P = None, None
for i in range(n_samples):
    z_meas = get_ball_pos()
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = kalman_filter(z_meas, x_esti, P)
 
    xpos_meas_save[i] = z_meas[0]
    ypos_meas_save[i] = z_meas[1]
    xpos_esti_save[i] = x_esti[0]
    ypos_esti_save[i] = x_esti[2]



fig = plt.figure(figsize=(8, 8))
plt.gca().invert_yaxis()
plt.scatter(xpos_meas_save, ypos_meas_save, s=30, c="r", marker='*', label='Position: Measurements')
plt.scatter(xpos_esti_save[-1], ypos_esti_save[-1], s=50, c="b", marker='o', label='Position: Estimation (KF)')
plt.legend(loc='lower right')
plt.title('Position: Meas. v.s. Esti. (KF)')
plt.xlabel('X-pos. [m]')
plt.ylabel('Y-pos. [m]')
plt.xlim((-10, 210))
plt.ylim((-10, 210))
plt.show()
