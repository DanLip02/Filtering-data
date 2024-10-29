import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from mpl_toolkits.mplot3d import Axes3D

def plot_pywt(coefficients, signal, scales):
    plt.figure(figsize=(12, 6))
    plt.plot(np.abs(coefficients))
    # plt.colorbar(label='Амплитуда')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('Wavelet spectrogram of LOD signal')
    plt.show()

def plot_AR_noise(coeffs, noisy_signal, n):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(noisy_signal)
    plt.title("Signal with added AR noise")

    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(coeffs), extent=[0, n, 1, 128], cmap='viridis', aspect='auto')
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

if __name__ == '__main__':
    df = pd.read_excel('/Users/danilalipatov/Downloads/Lab3/data_lab_3.xlsx')
    print(df)
    df['YEARS'] = pd.to_datetime(df['YEARS'])
    YEAR = df['YEARS']
    dT = df['YEARS'][1] - df['YEARS'][0]
    # df.set_index('YEARS', inplace=True)
    # sig = df['X'] + 1j*df['Y']
    sig = df['Y']
    signal = sig.values
    # wavelet = 'cmor'  # Комплексный вейвлет Морле
    # scales = np.arange(1, len(df['YEARS']))  # Масштабы (подбираются в зависимости от сигнала)
    # coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1)
    # plot_pywt(coefficients, signal, scales)
    wavelet = 'morl'  # вейвлет Морле
    scales = np.arange(1, 128)  # Масштабы

    # Вычисление CWT для компоненты Z
    coefficients_z, _ = pywt.cwt(signal, scales, wavelet)

    # Выбор нескольких масштабов для графика
    selected_scales = [i for i in range(1, 100, 10)]  # Масштабы для построения
    plt.figure(figsize=(12, 8))

    for i, scale in enumerate(selected_scales, 1):
        plt.subplot(len(selected_scales), 1, i)
        plt.plot(np.abs(coefficients_z[scale, :]), label=f'Scale {scale}')
        plt.ylabel(f'Амплитуда (Scale {scale})')
        plt.legend(loc='upper right')

    plt.xlabel('Time')
    plt.suptitle('CWT analysis of X component at different scales')
    plt.show()

    plt.imshow(np.abs(coefficients_z), cmap='viridis', aspect='auto')
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
    ar_params = np.array([1, -0.75])  # коэффициенты AR модели
    noise = np.random.normal(size=len(X))
    ar_noise = lfilter([1], ar_params, noise)

    # Сигнал с добавленным шумом
    noisy_signal = X + ar_noise

    # Вейвлет-анализ (например, используя вейвлет Морле)
    coeffs, freqs = pywt.cwt(noisy_signal, scales=np.arange(1, 128), wavelet='morl')
    plot_AR_noise(coeffs, noisy_signal, len(X))

    time = YEARS
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(noisy_signal, scales, 'morl')

    # Создаем 3D сетку: время, частоты (или масштабы), амплитуды вейвлет-преобразования
    # np.arange(1, 128, 128 / len(noisy_signal))
    T, F = np.meshgrid(time, 1 / freqs)
    Z = np.abs(coeffs)  # Модуль коэффициентов как амплитуда

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
    coeffs_imp, freqs_imp = pywt.cwt(impulsive_signal, scales=np.arange(1, 128), wavelet='morl')

    # Визуализация вейвлет-преобразования
    plt.imshow(np.abs(coeffs_imp), extent=[0, len(X), 1, 128], cmap='viridis', aspect='auto')
    plt.title("Вейвлет-преобразование импульсного сигнала")
    plt.xlabel("Время")
    plt.ylabel("Масштабы")
    plt.show()