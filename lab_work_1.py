import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Чтение данных из файла
# filename = r'/Users/danilalipatov/Downloads/Lab1/AAMWPgfz0.050-year.dat'
#
# # Загружаем данные из файла, пропуская первую строку (заголовок)
# data = np.loadtxt(filename, skiprows=1)
data = pd.read_excel('/Users/danilalipatov/PycharmProjects/HSE_lab_1/data_lab_1.xlsx')
print(data)
# Разделение данных
# YEARS = data[0, :]
# X = data[1, :]
# Y = data[2, :]
YEARS = data['YEAR']
X = data['X']
Y = data['Y']
# Вычисление временного шага
dT = YEARS[1] - YEARS[0]

# Построение графика X от времени
# Построение графиков X и Y по годам
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(YEARS, X, label="X(t)")
plt.xlabel('Years')
plt.ylabel('X')
plt.title('X over Time')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(YEARS, Y, label="Y(t)", color="red")
plt.xlabel('Years')
plt.ylabel('Y')
plt.title('Y over Time')
plt.grid(True)
plt.show()

# БПФ для X и Y
dT = YEARS[1] - YEARS[0]  # Шаг времени
N = len(X)

# Быстрое преобразование Фурье (БПФ)
X_fft = np.fft.fft(X) / N
Y_fft = np.fft.fft(Y) / N
freqs = np.fft.fftfreq(N, dT)

# Ограничение до положительных частот
positive_frequencies = freqs[:N // 2]
X_fft_pos = X_fft[:N // 2]
Y_fft_pos = Y_fft[:N // 2]

# Спектры для X и Y
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(positive_frequencies, np.abs(X_fft_pos), label="X Spectrum")
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum for X')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(positive_frequencies, np.abs(Y_fft_pos), label="Y Spectrum", color="red")
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum for Y')
plt.grid(True)
plt.show()

# Периоды
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(1 / positive_frequencies, np.abs(X_fft_pos), label="X Periodogram")
plt.xlabel('Period')
plt.ylabel('Amplitude')
plt.title('Periodogram for X')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(1 / positive_frequencies, np.abs(Y_fft_pos), label="Y Periodogram", color="red")
plt.xlabel('Period')
plt.ylabel('Amplitude')
plt.title('Periodogram for Y')
plt.grid(True)
plt.show()

# Комплексный ряд χ = X + iY и его БПФ
complex_signal = X + 1j * Y
complex_fft = np.fft.fft(complex_signal) / N

# Полные частоты для комплексного сигнала
plt.figure(figsize=(12, 6))
plt.plot(freqs, np.abs(complex_fft), label="Complex Spectrum", color="purple")
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum for Complex Signal')
plt.grid(True)
plt.show()

# Находим индекс с наибольшей амплитудой в спектре
dominant_freq_index = np.argmax(np.abs(X_fft_pos))
dominant_freq = positive_frequencies[dominant_freq_index]

# Значения для положительной и отрицательной частот
C_k = X_fft[dominant_freq_index]
C_minus_k = X_fft[-dominant_freq_index]

# Вычисление амплитуды и фазы для прямой и ретроградной гармоник
amplitude_k = np.abs(C_k)
amplitude_minus_k = np.abs(C_minus_k)
phase_k = np.angle(C_k)
phase_minus_k = np.angle(C_minus_k)

print(f"Dominant frequency: {dominant_freq}")
print(f"Amplitude for positive frequency (C_k): {amplitude_k}")
print(f"Phase for positive frequency (C_k): {phase_k}")
print(f"Amplitude for negative frequency (C_-k): {amplitude_minus_k}")
print(f"Phase for negative frequency (C_-k): {phase_minus_k}")


chi = complex_signal
N = len(chi)  # Количество точек данных

# БПФ для комплексного сигнала χ
chi_fft = np.fft.fft(chi) / N
freqs = np.fft.fftfreq(N, dT)

# Определяем положительные и отрицательные частоты
positive_frequencies = freqs[:N // 2]
negative_frequencies = freqs[N // 2:]

# Находим индекс с наибольшей амплитудой в спектре для положительных частот
dominant_freq_index = np.argmax(np.abs(chi_fft[:N // 2]))
dominant_freq_index_neg = np.argmax(np.abs(chi_fft[N // 2:]))
dominant_freq = positive_frequencies[dominant_freq_index]
dominant_freq_neg = negative_frequencies[dominant_freq_index_neg]
# Значения для положительной и отрицательной частот
C_k = chi_fft[dominant_freq_index]  # Положительная частота
C_minus_k = chi_fft[-dominant_freq_index]  # Отрицательная частота

# Вычисление амплитуды и фазы для положительной и отрицательной частот
amplitude_k = np.abs(C_k)  # Амплитуда для положительной частоты
amplitude_minus_k = np.abs(C_minus_k)  # Амплитуда для отрицательной частоты
phase_k = np.angle(C_k)  # Фаза для положительной частоты
phase_minus_k = np.angle(C_minus_k)  # Фаза для отрицательной частоты

# Вывод результатов
print('C_k', C_k)
print('C_-k', C_minus_k)
print(f"Dominant frequency: {dominant_freq} Hz")
print(f"Dominant frequency: {dominant_freq_neg} Hz")
print(f"Amplitude for positive frequency (C_k): {amplitude_k}")
print(f"Phase for positive frequency (C_k): {phase_k} rad")
print(f"Amplitude for negative frequency (C_-k): {amplitude_minus_k}")
print(f"Phase for negative frequency (C_-k): {phase_minus_k} rad")

# Визуализация спектра
plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, np.abs(chi_fft[:N // 2]), label="Amplitude Spectrum")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum of Complex Signal')
plt.grid(True)
plt.legend()
plt.show()

# Определяем функцию для вычисления спектра и частот с помощью FFT
# def ampl_fft(signal, dT):
#     N = len(signal)  # Количество точек данных
#     spectr = np.fft.fft(signal)  # Вычисляем преобразование Фурье
#     omega = np.fft.fftfreq(N, dT) * 2 * np.pi  # Частоты в радианах
#     return spectr, omega
import numpy as np
import matplotlib.pyplot as plt

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
N = len(X)  # Количество точек данных как в реальном сигнале
t = np.linspace(0, N * (YEARS[1] - YEARS[0]), N)  # Время в годах от 2000 года
print(t)
# Фазы, чтобы косинус обнулялся в 2000 году
phi1 = np.pi / 2
phi2 = np.pi / 2
phi3 = np.pi / 2
# phi1 = 0
# phi2 = 0
# phi3 = 0
# Модельный сигнал как сумма трёх гармоник
X_model = (A1 * np.cos(2 * np.pi * t / T1 + phi1) +
           A2 * np.cos(2 * np.pi * t / T2 + phi2) +
           A3 * np.cos(2 * np.pi * t / T3 + phi3))

# Построение реального и модельного сигнала вместе
plt.figure(figsize=(10, 6))
plt.plot(YEARS, Y, label="Real Signal", color="blue")
plt.plot(YEARS, X_model, label="Model Signal", color="orange", linestyle="--")
plt.xlabel('Years')
plt.ylabel('Amplitude')
plt.title('Real vs Model Signal')
plt.grid(True)
plt.legend()
plt.show()

# Спектральный анализ
X_model_fft = np.fft.fft(X_model) / len(X_model)
freqs_model = np.fft.fftfreq(len(X_model), YEARS[1] - YEARS[0])

# Ограничение до положительных частот
positive_frequencies_model = freqs_model[:len(X_model) // 2]
X_model_fft_pos = X_model_fft[:len(X_model) // 2]

# Построение амплитудного спектра для модельного сигнала
plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies_model, np.abs(X_model_fft_pos), label="Model Signal Spectrum", color="green")
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Amplitude Spectrum of Model Signal')
plt.grid(True)
plt.legend()
plt.show()

# Сравнение спектра модельного и реального сигнала
X_fft_pos = np.fft.fft(Y)[:len(Y) // 2] / len(Y)
positive_frequencies = np.fft.fftfreq(len(Y), YEARS[1] - YEARS[0])[:len(Y) // 2]

plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, np.abs(X_fft_pos), label="Real Signal Spectrum", color="blue")
plt.plot(positive_frequencies_model, np.abs(X_model_fft_pos), label="Model Signal Spectrum", color="orange", linestyle="--")
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Real vs Model Signal Spectrum')
plt.grid(True)
plt.legend()
plt.show()

def ampl_fft(f, dT):
    # Размер массива и количество точек
    N = len(f)

    # Преобразование Фурье и нормировка спектра
    fourier_transform = np.fft.fft(f)
    spectr = fourier_transform / N

    # Вычисляем угловые частоты omega
    omega = np.zeros(N)  # Инициализация массива для частот
    t = np.zeros(N)  # Инициализация массива для периодов

    for j in range(N):
        if j == 0:
            t[j] = 0
            omega[j] = 0
        elif j <= N // 2:
            t[j] = N * dT / (j)
            omega[j] = 2 * np.pi / t[j]
        else:
            t[j] = N * dT / (N - j)
            omega[j] = -2 * np.pi / t[j]

    return spectr, omega
# Преобразование данных X и Y в комплексный сигнал
complex_signal = X + 1j * Y

# Вычисляем спектр и частоты
spectr, omega = ampl_fft(complex_signal, dT)

# Построение графика спектра
plt.figure()
plt.plot(1.0 / (omega / (2 * np.pi)), np.abs(spectr))  # 1/omega для периода
plt.xlabel('Period (years)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.grid(True)
plt.show()