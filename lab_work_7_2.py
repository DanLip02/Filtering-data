import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import lsim
import control as ctrl
from scipy.signal import StateSpace, lsim, impulse, step, freqresp
from scipy.io import loadmat
from scipy.stats import alpha
import pandas as pd
from scipy.linalg import null_space
from numpy.linalg import det, eigvals
from scipy.linalg import expm
from sympy import Matrix, symbols, exp, pprint
import sympy as sp
# from lab_work_7 import signal_phi1

# Загрузка данных
# data = np.load('/Users/danilalipatov/Downloads/Lab7/signal.mat', allow_pickle=True)  # Замените на ваш формат, например .npy
# data = pd.read_excel('/Users/danilalipatov/Filtering-data/data/data_aam_.xlsx')
# signal_1 = data['X']
# signal_2 = data['Y']
# YEAR = data['YEAR']
# N_signal = len(signal_1)
# dt = YEAR[1] - YEAR[0]

N_signal = 1024
a = 18  # day
b = 10   # month
c = 2002  # year

# Periods (in years)
T1 = 0.5
T2 = 1
T3 = 4.6

# Harmonic amplitudes, normalized
A1 = (a / 31) * 20
A2 = (b / 12) * 20
A3 = ((c - 2000) / 50) * 20
N = 1024
t = np.linspace(0, N * 0.05, N)
print(t)

phi1 = np.pi / 2
phi2 = np.pi / 2
phi3 = np.pi / 2

X_model = (A1 * np.cos(2 * np.pi * t / T1 + phi1) +
           A2 * np.cos(2 * np.pi * t / T2 + phi2) +
           A3 * np.cos(2 * np.pi * t / T3 + phi3))

dt = 0.05
eps = 2.2 * np.random.randn(N)
signal = X_model + eps
# Временные шкалы
time_old = np.arange(0, N_signal / 12, 1 / 12)
time_new = np.arange(time_old[0], time_old[-1], dt)

# # Интерполяция сигнала
# interp_func = interp1d(time_old, signal_1, kind='zero')
# interp_signal = interp_func(time_new)

# График интерполированного сигнала
YEAR = t
signal_1 = signal
signal_2 = signal

plt.figure()
plt.plot(YEAR, signal_1, label='Signal 1', alpha=0.7)
plt.plot(YEAR, signal_2, label='Signal 2', alpha=0.7)
plt.legend()
plt.title("Original and Interpolated Signal")
plt.xlabel("Time (years)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Генерация шума AR(2)
ar = np.zeros(len(YEAR))
ar[0] = 0.5 * np.random.randn()
ar[1] = -0.8 * ar[0] + 0.5 * np.random.randn()
for i in range(2, len(ar)):
    ar[i] = -0.8 * ar[i-1] + 0.1 * ar[i-2] + 0.5 * np.random.randn()
ar = 20 * ar

# График шума
plt.figure()
plt.plot(ar)
plt.title("Generated Noise")
plt.grid()
plt.show()

# Суммирование сигнала с шумом
noisy_signal_1 = signal_1 + ar
noisy_signal_2 = signal_2 + ar

# График зашумленного сигнала
plt.figure()
plt.plot(YEAR, noisy_signal_1)
plt.plot(YEAR, noisy_signal_2, linestyle='--', alpha=0.75)
plt.title("Signal with Noise")
plt.xlabel("Time (years)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Определение параметров для модели системы
Q = 100
f_c = 365 / 433
# f_c =
alpha = 2 * np.pi * f_c
beta = np.pi * f_c / Q

# Матрицы системы
G = np.array([[-beta, -alpha], [alpha, -beta]])
F = -G
C = np.array([[1, 0], [0, 1]])
# C = np.eye(2)
D = np.zeros((2, 2))

# Создание модели системы
sys = ctrl.StateSpace(G, F, C, 0)

# Построение Bode-графика
# ctrl.bode(sys)
ctrl.bode(sys, dB=True, Hz=False, deg=False)
plt.show()

# Начальные условия и входной сигнал
# x0 = np.array([149, -116])
x0 = np.array([0, 0])
input_signal = np.vstack([noisy_signal_1, noisy_signal_2])

# Моделирование отклика системы
time_out, y_out = ctrl.forced_response(sys, T=YEAR, U=input_signal, X0=x0)

# input_signal_2 = np.vstack([noisy_signal_2, noisy_signal_2])
#
# time_out_2, y_out_2 = ctrl.forced_response(sys, T=YEAR, U=input_signal_2, X0=x0)

# График отклика системы
plt.figure()
plt.plot(time_out, y_out[0], label='Output 1')
plt.plot(time_out, y_out[1], label='Output 2')
plt.title("System Response")
plt.xlabel("Time (years)")
plt.ylabel("Response")
plt.legend()
plt.grid()
plt.show()

T_impulse, y_impulse = ctrl.impulse_response(sys, T=YEAR)
plt.figure()
plt.plot(T_impulse, y_impulse[0][0], label='Impulse Response φ1')
plt.plot(T_impulse, y_impulse[1][0], label='Impulse Response φ2')
plt.title("Impulse Response")
plt.xlabel("Time (years)")
plt.ylabel("Amplitude")
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.plot(T_impulse, y_impulse[0][1], label='Impulse Response φ1')
plt.plot(T_impulse, y_impulse[1][1], label='Impulse Response φ2')
plt.title("Impulse Response")
plt.xlabel("Time (years)")
plt.ylabel("Amplitude")
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()


T_step, y_step = ctrl.step_response(sys, T=YEAR)
plt.figure()
plt.plot(T_step, y_step[0][1], label='Step Response φ1')
plt.plot(T_step, y_step[1][1], label='Step Response φ2')
plt.title("Step Response")
plt.xlabel("Time (years)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Создание модели системы
sys = ctrl.StateSpace(G, F, C, 0)

# Построение Bode-графика
# ctrl.bode(sys)
ctrl.bode(sys, dB=True, Hz=False, deg=False)
plt.show()
#
# --- 1. Ядро матрицы G ---
kernel = null_space(G)  # Функция для нахождения нулевого пространства
if kernel.size == 0:
    print("Ядро матрицы G пусто (нулевое пространство отсутствует).")
else:
    print("Ядро матрицы G (нулевое пространство):")
    print(kernel)

# --- 2. Детерминант матрицы G ---
det_G = det(G)
print("\nОпределитель матрицы G:", det_G)
if np.isclose(det_G, 0):
    print("Матрица G вырождена (det = 0), она необратима.")
else:
    print("Матрица G невырождена (det ≠ 0), она обратима.")

# --- 3. Характеристическое уравнение ---
# Коэффициенты характеристического полинома: det(G - λI) = 0
coefficients = np.poly(G)  # poly возвращает коэффициенты характеристического полинома
print("\nКоэффициенты характеристического полинома матрицы G:")
print(coefficients)

# --- 4. Корни характеристического уравнения (собственные значения) ---
roots = np.roots(coefficients)
print("\nКорни характеристического уравнения (собственные значения матрицы G):")
print(roots)

# --- 5. Дополнительная проверка собственных значений ---
eigenvalues = eigvals(G)
print("\nСобственные значения матрицы G через eigvals:")
print(eigenvalues)

# --- Численное вычисление expm(tG) ---
t = 1.0  # Пример времени t
Phi = expm(t * G)  # Матричная экспонента
print("Численное значение переходной матрицы expm(tG) при t = 1:")
print(Phi)

# --- Символьное вычисление expm(tG) ---
t_sym = symbols('t')  # Символьная переменная времени
G_sym = Matrix([[-beta, -alpha],
                [alpha, -beta]])  # Символьная матрица G

# Вычисление матричной экспоненты в sympy
Phi_sym = G_sym.exp()  # Матричная экспонента
print("\nСимвольное выражение переходной матрицы expm(tG):")
pprint(Phi_sym)

# Частотный диапазон для анализа
omega = np.logspace(-1, 1, 100)  # Частоты от 0.1 до 10 (можно настроить)

# Матричное представление (pI - G)^(-1)
p = 1j * omega  # Комплексная переменная для преобразования Лапласа
I = np.eye(len(G))  # Единичная матрица
W_x_p = np.array([(p_val * I - G)**(-1) for p_val in p])

# Вычисление и визуализация
W_x_p_first = np.array([W_x[0, :] for W_x in W_x_p])  # Первый столбец матрицы

# Построение диаграммы Боде для матричной функции
# Диаграмма Боде для предаточной функции
plt.figure()
plt.semilogx(omega, np.abs(W_x_p_first))  # Амплитуда
plt.title('Магнитудная диаграмма предаточной функции')
plt.xlabel('Частота (рад/с)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.show()

from scipy.linalg import inv

# Определение p (например, p = 1j, что соответствует частоте 1 рад/с)
p_value = eigenvalues[0]

# Вычисление матрицы передачи
W_x_p = inv(p_value * np.eye(2) - G)

print("Матрица передачи W_x(p):\n", W_x_p)

# omega_year = np.logspace(-1, 1, 100)  # Частоты от 0.1 до 10 рад/год
omega_year = YEAR.values
# Преобразуем частоты из рад/год в рад/сек
seconds_in_year = 365.25 * 24 * 3600  # Количество секунд в году
omega_sec = omega_year * 2 * np.pi / seconds_in_year  # Частоты в рад/сек
sys = ctrl.StateSpace(G, F, C, D)
# Построение диаграммы Боде для системы с частотами в рад/сек
ctrl.bode(sys, omega=omega_sec, dB=True, Hz=False, deg=False)
plt.show()
G_siso = G[0, 0]  # Выбираем первый вход и первый выход
F_siso = F[0, 0]
C_siso = C[0, :]
D_siso = D[0, 0]
sys_siso = ctrl.StateSpace(G_siso, F_siso, C_siso, 0)
ctrl.bode(sys_siso, dB=True, Hz=False, deg=False)
plt.show()
G_siso = G[0, 1]  # Выбираем первый вход и второй выход
F_siso = F[0, 1]
C_siso = C[1, :]
D_siso = D[1, 1]
sys_siso = ctrl.StateSpace(G_siso, F_siso, C_siso, 0)
ctrl.bode(sys_siso, dB=True, Hz=False, deg=False)
plt.show()


# # Преобразуем систему в передаточную функцию
#
# # Преобразование в передаточную функцию
# num, den = ctrl.ss2tf(sys)  # num - числитель, den - знаменатель
#
# # Отображение передаточной функции
# print("Числитель:", num)
# print("Знаменатель:", den)
#
# # Построение диаграммы Боде для передаточной функции
# ctrl.bode([num, den], dB=True, Hz=True, deg=True)
#
# plt.show()

# Объявляем переменную Laplace (p)
p = sp.symbols('p')

# Матрица G
G = sp.Matrix([[-beta, -alpha], [alpha, -beta]])

# Вычисляем характеристическое уравнение: det(pI - G) = 0
I = sp.eye(2)  # Единичная матрица 2x2
char_eq = sp.det(p * I - G)

# Находим корни характеристического уравнения
eigenvalues = sp.solve(char_eq, p)

# Печатаем собственные частоты (корни)
print("Собственные частоты (корни уравнения):", eigenvalues)

# Матрица G
G = sp.Matrix([[-beta, -alpha], [alpha, -beta]])

# Матрицы C и D
C = sp.Matrix([[1, 0], [0, 1]])  # Пример для C
D = sp.Matrix([[0, 0], [0, 0]])  # Пример для D

# Матрица B (предположим B = [1, 0] для одно-входной системы)
B = -G

# Вычисляем матрицу (pI - G)
I = sp.eye(2)
pI_G = p * C - G

# Вычисляем обратную матрицу (pI - G)^(-1)
pI_G_inv = pI_G.inv()

# Вычисляем матричную передаточную функцию H(p)
H_p = C * pI_G_inv * B + D

# Печатаем матричную передаточную функцию
sp.pprint(H_p)

print(pI_G_inv)

stop = 0