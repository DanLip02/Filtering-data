from cProfile import label

import numpy as np
import matplotlib.pyplot as plt

# Генерация двухсинусоидального сигнала
N_signal = 1024
k = np.arange(N_signal)
signal = np.sin(2 * np.pi / 10 * (k - 1)) + np.sin(2 * np.pi / 100 * (k - 1))

# Добавление шума
eps = 0.2 * np.random.randn(N_signal)

# Гистограмма шума
plt.hist(eps, bins=20)
plt.show()

# Сигнал с шумом
plt.plot(k, signal + eps, label='signal with noise')
plt.plot(k, signal, linestyle="--", label='based signal')
plt.legend()
plt.title("Signal with noise")
plt.show()

# Генерация ARMA процесса
ar = np.zeros(N_signal)
ar[0] = eps[0]
ar[1] = -0.7 * ar[0] + eps[1]

for i in range(2, N_signal):
    ar[i] = -0.7 * ar[i - 1] + 0.2 * ar[i - 2] + eps[i]

# Построение ARMA сигнала
plt.plot(k, ar)
plt.title("ARMA Process")
plt.show()

# Построение сигнала с добавленным ARMA процессом
plt.plot(k, signal + ar)
plt.title("Signal with ARMA Process")
plt.show()

# Быстрое преобразование Фурье (FFT)
spectr = np.fft.fft(signal + ar)
plt.plot(np.abs(spectr))
plt.title("FFT of signal with ARMA process")
plt.show()

final_signal = signal + ar
# Вычисление автокорреляционной функции (ACF)
signal_centered = final_signal - np.mean(final_signal)
acf_biased = np.zeros(N_signal)
acf_unbiased = np.zeros(N_signal)

for tau in range(1, N_signal + 1):
    acf_biased[tau - 1] = np.mean(signal_centered[:N_signal - tau + 1] * signal_centered[tau - 1:N_signal])
    acf_unbiased[tau - 1] = np.sum(signal_centered[:N_signal - tau + 1] * signal_centered[tau - 1:N_signal]) / (
                N_signal - tau + 1)

# Построение АКФ
plt.figure()
plt.plot(acf_biased, label="Biased ACF")
plt.plot(acf_unbiased, label="Unbiased ACF", linestyle="--")
plt.title("Autocorrelation Function (ACF)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.legend()
plt.grid()
plt.show()

acf = np.zeros(N_signal)
acf_un = np.zeros(N_signal)
# Вычисление смещенной оценки АКФ
for tau in range(1, N_signal + 1):
    for j in range(N_signal - tau):
        acf[tau - 1] += signal_centered[j] * signal_centered[j + tau - 1]
    acf[tau - 1] /= N_signal  # Скалирование на размер сигнала

for tau in range(1, N_signal + 1):
    for j in range(N_signal - tau):
        acf_un[tau - 1] += signal_centered[j] * signal_centered[j + tau - 1]
    acf_un[tau - 1] /= (N_signal - tau + 1)  # Нормализация на оставшееся число пар

# Построение графика АКФ
plt.plot(acf, label="Biased ACF", color='blue')
plt.plot(acf_un, label="Unbiased ACF", linestyle="--", color='orange')
plt.title("Autocorrelation Function (ACF)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid()
plt.legend()
plt.show()

# Спектральная плотность (FFT от ACF)
n = len(acf_un)
n_bias = len(acf)
freqs_bias = np.fft.fftshift(np.fft.fftfreq(n_bias))  #
freqs = np.fft.fftshift(np.fft.fftfreq(n))  #
spectr_dens = np.fft.fft(acf)
plt.plot(freqs_bias, np.abs(np.fft.fftshift(spectr_dens)))
spectr_dens_un = np.fft.fft(acf_un)
plt.plot(freqs, np.abs(np.fft.fftshift(spectr_dens_un)), linestyle="--", color='orange')
plt.title("Spectral Density of ACF")
plt.show()


# Спектральная плотность (FFT от ACF)
n_white = N_signal
n_color = N_signal
freqs_white = np.fft.fftshift(np.fft.fftfreq(n_white))  #
freqs_color = np.fft.fftshift(np.fft.fftfreq(n_color))  #
spectr_dens_white = np.fft.fft(signal + eps)
plt.plot(freqs_white, np.abs(np.fft.fftshift(spectr_dens_white)), label='white noise')
spectr_dens_color = np.fft.fft(signal + ar)
plt.plot(freqs_color, np.abs(np.fft.fftshift(spectr_dens_color)), linestyle="--", color='orange', label='color noise')
plt.title("Spectral Density of ACF with white and color")
plt.yscale('log')
plt.legend()
plt.show()


# Параметры для гармоник
a = 18  # День рождения
b = 10   # Месяц рождения
c = 2002  # Год рождения

# Периоды (в годах)
T1 = 0.5
T2 = 1
T3 = 4.6

# Амплитуды гармоник, нормализованные
A1 = (a / 31) * 20
A2 = (b / 12) * 20
A3 = ((c - 2000) / 50) * 20

# Временная шкала
N = 1024  # Количество точек данных как в реальном сигнале
t = np.linspace(0, N * 0.05, N)  # Время в годах от 2000 года
print(t)
# Фазы, чтобы косинус обнулялся в 2000 году
phi1 = np.pi / 2
phi2 = np.pi / 2
phi3 = np.pi / 2
# Модельный сигнал как сумма трёх гармоник
X_model = (A1 * np.cos(2 * np.pi * t / T1 + phi1) +
           A2 * np.cos(2 * np.pi * t / T2 + phi2) +
           A3 * np.cos(2 * np.pi * t / T3 + phi3))

eps = 2.2 * np.random.randn(N_signal)
# Сигнал с шумом
plt.plot(k, X_model + eps, linestyle="--", color='orange', label='signal with noise')
plt.plot(k, X_model, label='current signal')
plt.legend()
plt.title("Signal with noise")
plt.show()

ar = np.zeros(N_signal)
ar[0] = eps[0]
ar[1] = -0.7 * ar[0] + eps[1]
ar[2] = X_model[2]
for i in range(3, N_signal):
    ar[i] = -0.7 * ar[i - 1] + 0.2 * ar[i - 2] + eps[i]

# Построение сигнала с добавленным ARMA процессом
plt.plot(k, X_model + ar)
plt.plot(k, X_model, linestyle="--", color='orange')
plt.title("Signal with ARMA Process")
plt.show()

# Быстрое преобразование Фурье (FFT)
spectr = np.fft.fft(X_model + ar)
plt.plot(np.abs(spectr))
plt.title("FFT of signal with ARMA process")
plt.show()

final_signal = X_model + ar
# Вычисление автокорреляционной функции (ACF)
signal_centered = final_signal - np.mean(final_signal)

acf = np.zeros(N_signal)
acf_un = np.zeros(N_signal)
# Вычисление смещенной оценки АКФ
for tau in range(1, N_signal + 1):
    for j in range(N_signal - tau):
        acf[tau - 1] += signal_centered[j] * signal_centered[j + tau - 1]
    acf[tau - 1] /= N_signal  # Скалирование на размер сигнала

for tau in range(1, N_signal + 1):
    for j in range(N_signal - tau):
        acf_un[tau - 1] += signal_centered[j] * signal_centered[j + tau - 1]
    acf_un[tau - 1] /= (N_signal - tau + 1)  # Нормализация на оставшееся число пар

# Построение графика АКФ
plt.plot(acf, label="Biased ACF", color='blue')
plt.plot(acf_un, label="Unbiased ACF", linestyle="--", color='orange')
plt.title("Autocorrelation Function (ACF)")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid()
plt.legend()
plt.show()

# Спектральная плотность (FFT от ACF)
n = len(acf_un)
n_bias = len(acf)
freqs_bias = np.fft.fftshift(np.fft.fftfreq(n_bias))  #
freqs = np.fft.fftshift(np.fft.fftfreq(n))  #
spectr_dens = np.fft.fft(acf)
plt.plot(freqs_bias, np.abs(np.fft.fftshift(spectr_dens)))
spectr_dens_un = np.fft.fft(acf_un)
plt.plot(freqs, np.abs(np.fft.fftshift(spectr_dens_un)), linestyle="--", color='orange')
plt.title("Spectral Density of ACF")
plt.show()

# Спектральная плотность (FFT от ACF)
n_white = N_signal
n_color = N_signal
freqs_white = np.fft.fftshift(np.fft.fftfreq(n_white))  #
freqs_color = np.fft.fftshift(np.fft.fftfreq(n_color))  #
spectr_dens_white = np.fft.fft(X_model + eps)
plt.plot(freqs_white, np.abs(np.fft.fftshift(spectr_dens_white)), label='white noise')
spectr_dens_color = np.fft.fft(X_model + ar)
plt.plot(freqs_color, np.abs(np.fft.fftshift(spectr_dens_color)), linestyle="--", color='orange', label='color noise')
plt.title("Spectral Density of ACF with white and color")
plt.yscale('log')
plt.legend()
plt.show()