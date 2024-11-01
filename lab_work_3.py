import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import alpha


def plot_pywt(coefficients, signal, scales):
    plt.figure(figsize=(12, 6))
    plt.plot(np.abs(coefficients))
    # plt.colorbar(label='Амплитуда')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('Wavelet spectrogram of LOD signal')
    plt.show()

def plot_AR_noise(coeffs, noisy_signal, n, X):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(noisy_signal, color='blue')
    plt.plot(X, linestyle="--", color='orange')
    plt.title("Signal with added AR noise")

    plt.subplot(2, 1, 2)
    # plt.imshow(np.abs(coeffs), extent=[0, n, 1, 128], cmap='viridis', aspect='auto')
    plt.imshow(np.abs(coeffs), cmap='viridis', aspect='auto')
    plt.title("Wavelet transform with noise")
    plt.xlabel("Time")
    plt.ylabel("Scales")
    plt.show()


def plot_3D(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.title("3D Поверхность")
    plt.show()

def example_urge(t, signal):
    # t = np.linspace(0, 1, 1000)  # Временная шкала
    # signal = np.sin(2 * np.pi * 5 * t)  # Основной сигнал

    # Добавление импульса
    impulse_position = 500  # Позиция импульса
    impulse_amplitude = 100  # Амплитуда импульса
    signal[impulse_position] += impulse_amplitude

    # Вейвлет-преобразование
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(signal, scales, 'morl')

    # Построение скейлограммы
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(coefficients), aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time (Year)')
    plt.ylabel('Scale')
    plt.title('Wavelet Scalegramm')

    # Треугольник влияния
    triangle_x = [t[impulse_position] - 0.1, t[impulse_position], t[impulse_position] + 0.1]  # Время
    triangle_y = [scales[-1], 1, scales[-1]]  # Скейлы
    plt.plot(triangle_x, triangle_y, color='red', linestyle='--', linewidth=2)
    plt.text(t[impulse_position], scales[-1] + 10, 'triangle', color='red', ha='center')

    plt.show()

if __name__ == '__main__':
    df = pd.read_excel('/Users/danilalipatov/Downloads/Lab3/data_lab_3.xlsx')
    print(df)
    # df['YEARS'] = pd.to_datetime(df['YEARS'])
    # YEAR = df['YEARS']
    dT = df['YEARS'][1] - df['YEARS'][0]
    # df.set_index('YEARS', inplace=True)
    # sig = df['X'] + 1j*df['Y']
    sig = df['Y']
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['YEARS'], sig, label="X(t)")
    plt.xlabel('Years')
    plt.ylabel('LOD')
    plt.title('Change LOD over Time')
    plt.grid(True)
    plt.show()
    signal = sig.values
    # wavelet = 'cmor'  # Комплексный вейвлет Морле
    # scales = np.arange(1, len(df['YEARS']))  # Масштабы (подбираются в зависимости от сигнала)
    # coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1)
    # plot_pywt(coefficients, signal, scales)
    wavelet = 'morl'  # вейвлет Морле
    max_l = 400
    scales = np.arange(1, max_l)  # Масштабы

    # Вычисление CWT для компоненты Z
    coefficients_z, freq = pywt.cwt(signal, scales, wavelet)

    # Выбор нескольких масштабов для графика
    selected_scales = [i for i in range(1, max_l, max_l // 10)]  # Масштабы для построения
    plt.figure(figsize=(12, 8))

    for i, scale in enumerate(selected_scales, 1):
        plt.subplot(len(selected_scales), 1, i)
        plt.plot(df['YEARS'], np.abs(coefficients_z[scale, :]), label=f'Scale {scale}')
        plt.ylabel(f'Амплитуда (Scale {scale})')
        plt.legend(loc='upper right')

    plt.xlabel('Time')
    plt.suptitle('CWT analysis of X component at different scales')
    plt.show()
    years = np.linspace(df['YEARS'].min(), df['YEARS'].max(), 708)  # Интерполяция для 708 точек
    plt.imshow(np.abs(coefficients_z), cmap='viridis', aspect='auto', extent=[years[0], years[-1], scales[0], scales[-1]])
    plt.xlabel('Years')
    plt.ylabel('Scale')
    plt.show()

    upp_1900 = df[df['YEARS'] <= 1900]['Y']
    chi_fft = np.fft.fft(upp_1900) / len(upp_1900)
    freqs = 2 * math.pi * np.fft.fftfreq(len(upp_1900), dT)
    plt.plot(freqs, chi_fft)
    plt.show()
    data = pd.read_excel('/Users/danilalipatov/PycharmProjects/HSE_lab_1/data_lab_1.xlsx')
    # Разделение данных
    # YEARS = data[0, :]
    # X = data[1, :]
    # Y = data[2, :]
    YEARS = data['YEAR']
    X = data['X']
    Y = data['Y']
    # Генерация авторегрессионного шума
    ar_params = [0.125, -0.55, -0.15, 0.35]  # коэффициенты AR(4)

    noise = np.random.normal(0, 1.1, len(X))

    ar_noise = np.zeros(len(X))
    for t in range(4, len(X)):
        ar_noise[t] = 0 + ar_params[0] * ar_noise[t - 1] + ar_params[1] * ar_noise[t - 2] + ar_params[2] * ar_noise[t - 3] + ar_params[3] * ar_noise[t - 4] + noise[t]

    # Сигнал с добавленным шумом
    noisy_signal = X + ar_noise

    # Вейвлет-анализ (например, используя вейвлет Морле)

    coeffs, freqs = pywt.cwt(noisy_signal, scales=np.arange(1, 40), wavelet='morl')
    print(coeffs)
    print(freqs)
    plot_AR_noise(coeffs, noisy_signal, len(X), X)

    time = YEARS
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(noisy_signal, scales, 'morl')
    # np.arange(1, 128, 128 / len(noisy_signal))
    T, F = np.meshgrid(time, 1 / freqs)
    Z = np.abs(coeffs)

    plot_3D(T, F, Z)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(T, F, Z, cmap='viridis')

    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency (Scales)")
    ax.set_zlabel("Amplitude")
    plt.title("3D Вейвлет-анализ сигнала")
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

    impulse_position = [100, 200, 300, 400, 542, 232, 111]
    impulse_magnitude = 100
    impulsive_signal = X.copy()
    for ind in impulse_position:
        impulsive_signal[ind] += impulse_magnitude

    # Построение графика
    plt.plot(impulsive_signal)
    # plt.plot(noisy_signal)
    plt.title("Сигнал с импульсом")
    plt.show()

    # Вейвлет-анализ для импульсного сигнала
    scales = np.arange(1, 128)
    coeffs_imp, freqs_imp = pywt.cwt(impulsive_signal, scales=scales, wavelet='morl')

    # Визуализация вейвлет-преобразования
    plt.imshow(np.abs(coeffs_imp), extent=[0, len(X), 1, 128], cmap='viridis', aspect='auto')
    plt.title("Wavelet transform of a impulse signal")
    plt.xlabel("Time")
    plt.ylabel("Scales")
    # plt.show()
    for pos in impulse_position:
            plt.plot([pos, pos - 64 / 2, pos + 64 / 2, pos],
                     [128, 1, 1, 128], 'w-', alpha=0.5)
    plt.show()
    example_urge(YEARS, noisy_signal)