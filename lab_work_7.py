import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import control as ctrl
from scipy.linalg import null_space
from sympy import Matrix, symbols, exp
import pandas as pd

def PMInversion(year, s, dt, FC, Q):
    """
    Фильтрация в полосе частот Чандлера с использованием фильтра Пантелеева.
    Входные параметры:
    - year: массив дат, ожидается равномерный временной шаг
    - s: комплексный массив сигнала для сглаживания
    - dt: временной шаг
    - FC, Q: параметры фильтра (частота и добротность)

    Выход:
    - out_signal: отфильтрованный сигнал
    """
    N = len(s)

    # Параметры фильтра
    sigmaC = 2 * np.pi * FC * (1 + 1j / (2 * Q))  # Комплексная частота
    tau = 1j / sigmaC

    # Преобразование Фурье сигнала
    sF = np.fft.fft(s)

    # Инициализация частотных параметров
    omega = np.zeros(N, dtype=np.float64)
    Sym = np.zeros(N, dtype=np.complex128)  # Передаточная функция

    for k in range(N):
        if k == 0:
            omega[k] = 0
            Sym[k] = 0  # Симметрия
        elif k <= N // 2:
            omega[k] = 2 * np.pi * k / (N * dt)
            Sym[k] = 1 + tau * 1j * omega[k]
        else:
            omega[k] = -2 * np.pi * (N - k) / (N * dt)
            Sym[k] = 1 + tau * 1j * omega[k]

    # Передаточная функция
    # omega = 2 * np.pi * np.fft.fftfreq(N, dt)
    # Sym = 1 + tau * 1j * omega
    # Sym[0] = 0  # Удаление нулевой частоты
    # Sym[N // 2:] = 0  # Удаление высоких частот

    # Применение фильтра
    out_signal_fft = sF * Sym

    # Обратное преобразование Фурье
    out_signal = np.fft.ifft(out_signal_fft)

    return out_signal

# def ChandPantFreqFilter(year, s, f_o, fc, dt, FC, Q, inv, outfilename=None):
#     """
#     Фильтрация в полосе частот Чандлера с использованием фильтра Пантелеева.
#
#     Параметры:
#     - year: массив дат
#     - s: комплексный массив сигнала для фильтрации
#     - f_o: параметр фильтра
#     - fc: центральная частота фильтра
#     - dt: временной шаг
#     - FC, Q: параметры (частота и добротность резонанса)
#     - inv: параметр инверсии
#     - outfilename: имя выходного файла (если нужно сохранить результаты)
#
#     Выход:
#     - out_signal: отфильтрованный сигнал
#     """
#
#     N = len(s)  # Количество точек сигнала
#
#     om = 2 * np.pi * f_o
#     omc = 2 * np.pi * fc
#     sigmaC = 2 * np.pi * FC * (1 + 1j / (2 * Q))
#     tau = 1j / sigmaC
#
#     # Преобразование Фурье
#     sF = np.fft.fft(s)
#
#     # Инициализация массивов
#     omega = np.zeros(N)
#     Sym = np.zeros(N, dtype=complex)
#     TRF = np.zeros(N)
#
#     omega = 2 * np.pi * np.fft.fftfreq(N, dt)
#
#     for k in range(N):
#         if k == 0:
#             omega[k] = 1
#             TRF[k] = 1 if omc == 0 else 0
#             Sym[k] = 1 if inv == -1 else 0
#         elif k <= N // 2:
#             # omega[k] = 2 * np.pi * k / (N * dt)
#             Sym[k] = 1 + tau * 1j * omega[k]
#             TRF[k] = om**4 / ((omega[k] - omc)**4 + om**4)
#         else:
#             # omega[k] = -2 * np.pi * (N - k) / (N * dt)
#             Sym[k] = 1 + tau * 1j * omega[k]
#             TRF[k] = om**4 / ((omega[k] - omc)**4 + om**4)
#
#
#     # # Сохранение в файл (если указано имя файла)
#     # if outfilename:
#     #     with open(outfilename, 'w') as fout:
#     #         for j in range(N):
#     #             fout.write(f"{omega[j] / (2 * np.pi):10.8f} {abs(TRF[j]):10.8e} {abs(Sym[j]):10.8e} {abs(sF[j]) / N:10.8e}\n")
#
#     # Применение фильтрации
#     resSym = sF * TRF
#     if inv == 1:
#         out_signal_fft = resSym * Sym
#     elif inv == -1:
#         out_signal_fft = resSym / Sym
#     else:
#         out_signal_fft = resSym
#
#     # Обратное преобразование Фурье
#     out_signal = np.fft.ifft(out_signal_fft)
#
#     return out_signal

import numpy as np


def ChandPantFreqFilter(year, s, f_o, fc, dt, FC, Q, inv, outfilename=None):
    """
    Фильтрация в диапазоне частоты Чандлера с использованием фильтра Пантелеева.

    :param year: массив дат (ожидается равномерный шаг времени)
    :param s: комплексный массив сигнала для обработки
    :param f_o: параметр фильтра
    :param fc: центральная частота фильтра Пантелеева
    :param dt: шаг времени
    :param FC: центральная частота резонанса (Гц)
    :param Q: добротность резонанса
    :param inv: параметр инверсии (1: с инверсией, -1: обратная, 0: без)
    :param outfilename: имя файла для записи данных (если `None`, данные не записываются)
    :return: обработанный сигнал
    """
    N = len(s)  # Длина сигнала

    om = 2 * np.pi * f_o
    omc = 2 * np.pi * fc
    sigmaC = 2 * np.pi * FC * (1 + 1j / (2 * Q))
    tau = 1j / sigmaC

    # Быстрое преобразование Фурье сигнала
    sF = np.fft.fft(s)

    # Инициализация массивов
    t = np.zeros(N)
    omega = np.zeros(N)
    TRF = np.zeros(N, dtype=complex)
    Sym = np.zeros(N, dtype=complex)

    # Построение передаточной функции и фильтра
    for k in range(N):
        if k == 0:
            omega[k] = 0
            t[k] = 0
            Sym[k] = 1 if inv == -1 else 0
            TRF[k] = 1 if omc == 0 else 0
        elif k <= N // 2:
            t[k] = N * dt / (k - 1)
            omega[k] = 2 * np.pi / t[k]
            Sym[k] = (1 + tau * 1j * omega[k])
            TRF[k] = om ** 4 / ((omega[k] - omc) ** 4 + om ** 4)
        else:
            t[k] = N * dt / (N - k + 1)
            omega[k] = -2 * np.pi / t[k]
            Sym[k] = (1 + tau * 1j * omega[k])
            TRF[k] = om ** 4 / ((omega[k] - omc) ** 4 + om ** 4)

    # Если указан выходной файл, записываем данные
    if outfilename is not None:
        with open(outfilename, 'w') as fout:
            for j in range(N):
                fout.write(f"{omega[j] / (2 * np.pi):.8f} ")
                fout.write(f"{abs(TRF[j]):.8e} ")
                fout.write(f"{abs(Sym[j]):.8e} ")
                fout.write(f"{abs(sF[j]) / N:.8e}\n")

    # Применение фильтра
    resSym = sF * TRF

    if inv == 1:
        out_signal_fft = resSym * Sym
    elif inv == -1:
        out_signal_fft = resSym / Sym
    else:
        out_signal_fft = resSym

    # Обратное преобразование Фурье
    out_signal = np.fft.ifft(out_signal_fft)

    return out_signal


def ampl_fft(f, T):
    """
    f - входной сигнал
    T - временной шаг
    Выход:
    - ampl_spectr: амплитудный спектр
    - omega: частоты
    """
    # f = np.array(f)
    # N = len(f)  # Размерность сигнала

    # # Прямое преобразование Фурье
    # fourier_transform = np.fft.fft(f)
    # ampl_spectr = np.abs(fourier_transform) / N

    # t = np.zeros(N)
    # omega = np.zeros(N)

    # # Вычисление частот omega
    # for j in range(N):
    #     if j == 0:
    #         t[j] = 0
    #         omega[j] = 0
    #     elif j <= N // 2:
    #         t[j] = N * T / (j)
    #         omega[j] = 2 * np.pi / t[j]
    #     else:
    #         t[j] = N * T / (N - j)
    #         omega[j] = -2 * np.pi / t[j]

    f = np.array(f)
    N = len(f)
    fourier_transform = np.fft.fft(f)
    ampl_spectr = np.abs(fourier_transform) / N
    omega = 2 * np.pi * np.fft.fftfreq(N, T)  # Используем np.fft.fftfreq вместо цикла
    return ampl_spectr, omega

# filename = '/Users/danilalipatov/Downloads/Lab7/AAMWPgfz0.050-year.dat'
# data = np.loadtxt(filename)

# Извлечение данных
data = pd.read_excel('/Users/danilalipatov/Filtering-data/data/data_aam_.xlsx')
signal_1 = data['X']
signal_2 = data['Y']
YEAR = data['YEAR']
N_signal = len(signal_1)
dt = YEAR[1] - YEAR[0]

# N = data.shape[1]
x0 = np.array([0, 0])
input_signal = np.vstack([signal_1, signal_2])

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
# Моделирование отклика системы
time_out, y_out = ctrl.forced_response(sys, T=YEAR, U=input_signal, X0=x0)
N_max = 955
year = YEAR.values
phi1_in = signal_1
phi2_in = signal_2
m_1 = y_out[0]
m_2 = y_out[1]

# year = years[1] - years[0]
# year =
# Определение временного шага
# dt = year[1] - year[0]
inp = phi1_in + 1j * phi2_in
m = m_1 + 1j * m_2
# Вейвлет преобразование
scales = np.arange(1, 40)  # Задаём диапазон масштабов
coefficients, frequencies = pywt.cwt(inp, scales, 'morl', dt)  # CWT с вейвлетом Морле
# Преобразование масштаба в период
frequencies = pywt.scale2frequency('morl', scales) / dt  # Частота в циклах на год
period = 1 / frequencies  # Период = 1 / частота

plt.imshow(np.abs(coefficients), extent=[year[0], year[-1], period[0], period[-1]], cmap='jet', aspect='auto')
plt.colorbar(label='Amplitude')
plt.title('Wavelet Scalogram')
plt.xlabel('Time (years)')
plt.ylabel('Period (years)')
plt.yscale('log')
plt.show()

# Вейвлет преобразование
scales = np.arange(1, 40)  # Задаём диапазон масштабов
coefficients, frequencies = pywt.cwt(m, scales, 'morl', dt)  # CWT с вейвлетом Морле
# Преобразование масштаба в период
frequencies = pywt.scale2frequency('morl', scales) / dt  # Частота в циклах на год
period = 1 / frequencies  # Период = 1 / частота

plt.imshow(np.abs(coefficients), extent=[year[0], year[-1], period[0], period[-1]], cmap='jet', aspect='auto')
plt.colorbar(label='Amplitude')
plt.title('Wavelet Scalogram')
plt.xlabel('Time (years)')
plt.ylabel('Period (years)')
plt.yscale('log')
plt.show()

# Вейвлет преобразование
scales = np.arange(1, 40)  # Задаём диапазон масштабов
coefficients, frequencies = pywt.cwt(m_1, scales, 'morl', dt)  # CWT с вейвлетом Морле
# Преобразование масштаба в период
frequencies = pywt.scale2frequency('morl', scales) / dt  # Частота в циклах на год
period = 1 / frequencies  # Период = 1 / частота

plt.imshow(np.abs(coefficients), extent=[year[0], year[-1], period[0], period[-1]], cmap='jet', aspect='auto')
plt.colorbar(label='Amplitude')
plt.title('Wavelet Scalogram')
plt.xlabel('Time (years)')
plt.ylabel('Period (years)')
plt.yscale('log')
plt.show()

# Вейвлет преобразование
scales = np.arange(1, 40)  # Задаём диапазон масштабов
coefficients, frequencies = pywt.cwt(m_2, scales, 'morl', dt)  # CWT с вейвлетом Морле
# Преобразование масштаба в период
frequencies = pywt.scale2frequency('morl', scales) / dt  # Частота в циклах на год
period = 1 / frequencies  # Период = 1 / частота

plt.imshow(np.abs(coefficients), extent=[year[0], year[-1], period[0], period[-1]], cmap='jet', aspect='auto')
plt.colorbar(label='Amplitude')
plt.title('Wavelet Scalogram')
plt.xlabel('Time (years)')
plt.ylabel('Period (years)')
plt.yscale('log')
plt.show()

#
# plt.rcParams["figure.figsize"] = (10,10)
# plt.rcParams['figure.dpi'] = 400

# Комплексный сигнал
inp = phi1_in + 1j * phi2_in
m = m_1 + 1j * m_2

# Построение графиков
# plt.figure(figsize=(10, 10))
plt.plot(year, m, label=r"$m$ (Комплексный отклик системы)")
plt.plot(year, inp, label=r"$inp$ (Комплексные отклик данные)")
plt.xlabel("Время (годы)")
plt.ylabel("Амплитуда")
plt.title("Динамическая система вращения Земли в пространстве состояний")
# plt.ylim(bottom=1e-50, top=1e2)
# plt.xlim(left=20, right=30)
ax = plt.gca()
# ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
plt.legend()
plt.grid()

# Расчет спектра
spectr, omega = ampl_fft(inp, dt)
spectr_m, omega = ampl_fft(m, dt)

plt.figure()
# plt.plot(omega[:N_max-1] / (2 * np.pi), np.abs(spectr[:N_max-1]), label="Сигнал inp")
plt.plot(np.fft.fftshift(omega) / (2 * np.pi), np.fft.fftshift(spectr), label="Сигнал inp (вход)")
plt.plot(np.fft.fftshift(omega) / (2 * np.pi), np.fft.fftshift(spectr_m), label="Сигнал m (отклик)")
plt.title("Амплитудный спектр")
plt.xlabel("Частота (1/год)")
plt.ylabel("Амплитуда")
plt.xlim(left=-4, right=4)
# plt.yscale('log')
plt.legend()
plt.grid()

plt.show()

Q = 100
FC = 365 / 433

# Фильтрация
out_signal = PMInversion(year, m, dt, FC, Q)

# Добавление шума
noise = 10 * (np.random.randn(2, N_max))
m_noisy = m + (noise[0, :] + 1j * noise[1, :])

# Фильтрация зашумленного сигнала
out_noisy_signal = PMInversion(year, m_noisy, dt, FC, Q)

# Построение графиков
plt.figure()
plt.plot(year, np.real(m), label="Исходный сигнал m")
plt.plot(year, np.real(out_signal), label="Фильтрованный сигнал m_filt")
plt.plot(year, phi1_in, label="Сигнал phi1_in")
plt.title("Фильтрация сигнала PMInversion()")
plt.xlabel("Время (годы)")
plt.ylabel("Амплитуда")
plt.ylim(bottom=-200, top=200)
# plt.xlim(left=5, right=20)
plt.legend()
plt.grid()

plt.figure()
plt.plot(year, np.real(m_noisy), label="Зашумленный m_noisy")
plt.plot(year, np.real(m), label="Оригинальный m")
plt.title("Зашумленный сигнал сравнение")
plt.xlabel("Время (годы)")
plt.ylabel("Амплитуда")
# plt.xlim(left=5, right=20)
plt.legend()
plt.grid()

plt.figure()
plt.plot(year, np.real(out_noisy_signal), label="Фильтрованный зашумленный m_noisy_filt")
plt.plot(year, np.real(m_noisy), label="Оригинальный зашумленный m_noisy")
plt.plot(year, phi1_in, label="Сигнал phi1_in") # phi1_in
plt.title("Фильтрация зашумленного сигнала PMInversion()")
plt.xlabel("Время (годы)")
plt.ylabel("Амплитуда")
# plt.ylim(bottom=-2000, top=2000)
# plt.xlim(left=5, right=20)
plt.legend()
plt.grid()

plt.show()


# Параметры фильтра
# f_c = 0.843
f_0 = 3
f_c = 0
# f_0 = 0.045
inv = 1
FC = 365 / 433
Q = 100
# outfilename = f"CW_TRF_{round(1 / f_c)}_{round(1 / f_0)}_1.dat"

# Фильтрация сигнала
# filtered_signal = ChandPantFreqFilter(year, m_noisy, f_0, f_c, dt, FC, Q, inv)
filtered_signal = ChandPantFreqFilter(year, m, f_0, f_c, dt, FC, Q, inv)

# Построение графиков
plt.plot(year, np.real(filtered_signal), 'g', label="Фильтрованный сигнал filtered_signal")
plt.plot(year, np.real(m), 'r',label="Зашумленный сигнал m_noisy")
plt.plot(year, np.real(inp), 'k', label="Оригинальный сигнал inp")
plt.xlabel("Время (годы)")
plt.ylabel("Амплитуда")
plt.title("Фильтрация сигнала ChandPantFreqFilter()")
plt.xlim(left=10, right=35)
plt.legend()
plt.grid()
plt.show()

# Спектры сигналов
spectr_in, freq_in = ampl_fft(m_noisy, dt)
spectr, freq = ampl_fft(m, dt)

plt.plot(np.fft.fftshift(freq_in) / (2 * np.pi), np.abs(np.fft.fftshift(spectr_in)), label="Спектр зашумленного сигнала")
plt.plot(np.fft.fftshift(freq) / (2 * np.pi), np.abs(np.fft.fftshift(spectr)), label="Спектр оригинального сигнала (уменьшен в 5 раз)")
plt.xlabel("Частота (1/год)")
plt.ylabel("Амплитуда")
plt.xlim(left=-4, right=4)
plt.legend()
plt.grid()
plt.show()

spectr_fltr, freq_fltr = ampl_fft(filtered_signal, dt)
plt.figure()
plt.plot(np.fft.fftshift(freq) / (2 * np.pi), np.abs(np.fft.fftshift(spectr)), label="Спектр оригинального сигнала")
plt.plot(np.fft.fftshift(freq_fltr) / (2 * np.pi), np.abs(np.fft.fftshift(spectr_fltr)), label="Спектр фильтрованного сигнала")
plt.xlabel("Частота (1/год)")
plt.ylabel("Амплитуда")
plt.xlim(left=-4, right=4)
plt.legend()
plt.grid()
plt.show()

# Добавление шума
noise_ = 10 * (np.random.randn(2, N_max))
m_noisy_ = m + (noise_[0, :] + 1j * noise_[1, :])

mn = m + (np.random.randn(N_max) + 1j * np.random.randn(N_max))  # Зашумленный сигнал

inverted_signal = ChandPantFreqFilter(year, mn, f_0, f_c, dt, FC, Q, inv)

# График с инверсной фильтрацией
plt.plot(year, np.real(mn), label="Зашумленный сигнал mn")
plt.plot(year, np.real(inp), label="Оригинальный сигнал inp")
plt.plot(year, np.real(inverted_signal), label="Инверсно-фильтрованный сигнал inverted_signal")
plt.xlabel("Время (годы)")
plt.ylabel("Амплитуда")
plt.legend()
plt.title("Инверсная фильтрация сигнала ChandPantFreqFilter()")
plt.grid()

plt.show()