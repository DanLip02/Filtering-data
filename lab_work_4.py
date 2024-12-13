import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd

from sklearn.decomposition import TruncatedSVD


def mssa_analysis(signal, dt, L, group_seq):
    """
    Выполняет MSSA анализ сигнала и возвращает сгруппированные компоненты.
    signal: ndarray, входной сигнал
    dt: float, временной шаг
    L: int, длина окна
    group_seq: list, группировка компонент

    Возвращает:
    grouped_components: dict, компоненты по группам
    """
    N = len(signal)
    K = N - L + 1

    # Шаг 1: Построение траекторной матрицы
    trajectory_matrix = np.zeros((L, K))
    for i in range(L):
        trajectory_matrix[i] = signal[i:i + K]

    # Шаг 2: SVD разложение
    svd = TruncatedSVD(n_components=L)
    U = svd.fit_transform(trajectory_matrix)  # Левые сингулярные векторы
    S = svd.singular_values_  # Сингулярные значения
    V = svd.components_  # Правые сингулярные векторы

    # Шаг 3: Реконструкция компонент
    components = np.zeros((L, N))
    for i in range(L):
        component_matrix = np.outer(U[:, i], V[i, :]) * S[i]
        for j in range(L):
            components[i, j:j + K] += component_matrix[j, :]
        components[i, :] /= np.arange(1, K + 1).tolist() + [K] * (N - K)

    # Шаг 4: Группировка компонент
    grouped_components = {i: np.zeros(N) for i in range(len(group_seq))}
    for group_idx, group in enumerate(group_seq):
        for idx in group:
            if idx > 0:  # Пропускаем нулевые индексы
                grouped_components[group_idx] += components[idx - 1]

    return grouped_components

def ampl_fft(signal, dT):
    N = len(signal)
    fourier_transform = np.fft.fft(signal)
    spectr = fourier_transform / N

    omega = np.zeros(N)
    t = np.zeros(N)

    for j in range(N):
        if j == 0:
            omega[j] = 0
        elif j <= N // 2:
            t[j] = N * dT / (j)
            omega[j] = 2 * np.pi / t[j]
        else:
            t[j] = N * dT / (N - j)
            omega[j] = -2 * np.pi / t[j]

    return spectr, omega


def hankelization(G1, L, N, L_min, K_max):
    if L != L_min:
        return np.zeros(N)

    X1 = np.zeros(N)

    # Краевые элементы
    for ii in range(1, L_min + 1):
        for j in range(ii):
            X1[ii - 1] += G1[ii - j - 1, j]
            X1[N - ii] += G1[L_min - (ii - j), K_max - j - 1]
        X1[ii - 1] /= ii
        X1[N - ii] /= ii

    # Центральные элементы
    for ii in range(L_min, K_max):
        for j in range(L_min):
            X1[ii] += G1[L_min - j - 1, ii - L_min + j]
        X1[ii] /= L_min

    return X1

def mssa(Date, SIGNAL, N_loc, N, L, N_ev, coef, pathout, group_seq):
    K = N - L + 1
    A = np.zeros((L * N_loc, K))

    for ii in range(L):
        for j in range(K):
            for k in range(N_loc):
                A[k * L + ii, j] = SIGNAL[k, ii + j]

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T

    RX = np.zeros((N_ev, N_loc, N))
    for ii in range(N_ev):
        G = np.outer(U[:, ii], S[ii] * V[:, ii])
        for k in range(N_loc):
            Block = G[k * L:(k + 1) * L, :]
            RX[ii, k, :] = hankelization(Block, L, N, min(L, K), max(L, K))

    components = np.zeros((N_loc, len(group_seq), N))
    # fig, axes = plt.subplots(N_loc, len(group_seq), figsize=(15, 5 * N_loc), sharex=True)
    fig, axes = plt.subplots(N_loc * len(group_seq), 1, figsize=(10, 5 * N_loc * len(group_seq)), sharex=True)
    for l in range(N_loc):
        for m, group in enumerate(group_seq):
            label = ""
            for ind in group:
                if ind > 0:
                    components[l, m, :] += RX[ind - 1, l, :]
                    label += str(ind)
            # plt.plot(Date, components[l, m, :], label=f"Group {label}")
            # ax = axes[l, m] if N_loc > 1 else axes[m]  # Если только 1 компонент, индексация упрощается
            # ax.plot(Date, components[l, m, :], label=f"Group {label}")
            ax_idx = l * len(group_seq) + m  # Индекс текущего графика
            ax = axes[ax_idx] if len(axes) > 1 else axes  # Если всего один график, убрать массивность
            ax.plot(Date, components[l, m, :], label=f"Group {label}")
            ax.legend()
            ax.set_title(f"Signal {l + 1}, Group {m + 1}")
            ax.set_ylabel("Amplitude")
            if ax_idx == len(axes) - 1:
                ax.set_xlabel("Date")
        # plt.legend()
        plt.tight_layout()
        plt.title(f"Component Groups (Signal {l + 1})")
        plt.show()

    return components


def my_signal():
    data = pd.read_excel('/Users/danilalipatov/PycharmProjects/HSE_lab_1/data_lab_1.xlsx')
    YEARS = data['YEAR']
    X = data['X']
    # Вычисление временного шага
    # dT = YEARS[1] - YEARS[0]
    # Параметры для гармоник
    a = 18  # День рождения
    b = 10  # Месяц рождения
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
    t = np.linspace(0, N * (0.05), N)  # Время в годах от 2000 года
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
    X_model_1 = (A1 * np.cos(2 * np.pi * t / T1 + phi1))
    X_model_2 = A2 * np.cos(2 * np.pi * t / T2 + phi2)
    X_model_3 = A3 * np.cos(2 * np.pi * t / T3 + phi3)
    return t, X_model, dt, X_model_1, X_model_2, X_model_3


# Генерация временного ряда


N_signal = 1024
dt = 1 / 12
P1 = 10 / dt
P2 = 1 / dt

garm1 = np.zeros(N_signal)
garm2 = np.zeros(N_signal)
trend = np.zeros(N_signal)
dates = np.zeros(N_signal)

# for k in range(1, N_signal):
#     garm1[k] = 0.1 * k * np.sin(2 * np.pi / P1 * (k - 1))
#     garm2[k] = 10 * np.cos(2 * np.pi / P2 * (k - 1))
#     trend[k] = 0.1 * k
#     dates[k] = 2000 + dt * (k - 1)
for k in range(N_signal):
    garm1[k] = 0.1 * (k + 1) * np.sin(2 * np.pi / P1 * k)
    garm2[k] = 10 * np.cos(2 * np.pi / P2 * k)
    trend[k] = 0.1 * (k + 1)
    dates[k] = 2000 + dt * k

noise = 2 * np.random.randn(N_signal)
# dates = np.arange(2000, 2000 + dt * N_signal, dt)
# garm1 = 0.1 * np.arange(N_signal) * np.sin(2 * np.pi / P1 * np.arange(N_signal))
# garm2 = 10 * np.cos(2 * np.pi / P2 * np.arange(N_signal))
# trend = 0.1 * np.arange(N_signal)
# noise = 2 * np.random.randn(N_signal)

signal = garm1 + garm2 + trend + noise

# График компонентов
plt.plot(dates, garm1, label="harmonic 1")
plt.plot(dates, garm2, label="harmonic 2")
plt.plot(dates, trend, label="trend")
plt.plot(dates, noise, label="noise")
plt.plot(dates, signal, 'black', label="signal")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Amplitude")
plt.title("Generated Signal Components")
plt.show()


spectr, omega = ampl_fft(signal, dt)
plt.plot(omega / (2 * np.pi), np.abs(spectr))
plt.xlabel("Frequency (cycles per year)")
plt.ylabel("Amplitude")
plt.title("Amplitude Spectrum (FFT)")
plt.show()

scales = np.arange(1, 256)
# extent=[dates[0], dates[-1], scales[-1], scales[0]],
# Непрерывное вейвлет-преобразование
coefficients, frequencies = pywt.cwt(garm1, scales, 'cmor', sampling_period=dt)
plt.imshow(np.abs(coefficients), cmap='jet')
plt.colorbar(label='Amplitude')
plt.title('Continuous Wavelet Transform (CWT) - Morlet Wavelet')
plt.xlabel('Time (Year)')
plt.ylabel('Scale')
plt.show()

# Преобразование масштаба в период
frequencies = pywt.scale2frequency('cmor', scales) / dt  # Частота в циклах на год
periods = 1 / frequencies  # Период = 1 / частота

# Построение скейлограммы
plt.figure(figsize=(10, 6))
dates = dates - 2000
plt.imshow(np.abs(coefficients), extent=[dates[0], dates[-1], periods[0], periods[-1]],
           aspect='auto', cmap='jet')
plt.colorbar(label='Amplitude')
plt.title('Scalogram (CWT) with Period on Y-axis')
plt.xlabel('Time (Year)')
plt.ylabel('Period (Years)')
plt.yscale('log')  # Логарифмическая шкала по периоду
plt.show()

# Пример вызова MSSA
N_loc = 1  # Количество компонент
L = 300     # Параметр задержки
N = len(signal)  # Длина сигнала
N_ev = 7  # Число собственных значений
coef = 1  # Коэффициенты
pathout = './output'  # Путь для вывода
# group_seq = [[1]]  # Группировка компонент
group_seq = [
    [1, 0],          # Тренд
    [2, 3],       # Гармоника 1
    [4, 5]        # Гармоника 2
]
components = mssa(dates, signal.reshape(1, -1), N_loc, N, L, N_ev, coef, pathout, group_seq)

t, signal, dt, garm_1, garm_2, garm_3 = my_signal()
# N_loc = 1  # Количество компонент
L = 300     # Параметр задержки
# L=int(4.6/dt)
# N = len(signal)  # Длина сигнала
# N_ev = 7  # Число собственных значений
# coef = 1  # Коэффициенты
# pathout = './output'  # Путь для вывода
# # group_seq = [[1]]  # Группировка компонент
# group_seq = [
#     [1],         # Тренд
#     [2, 3],      # Первая гармоника
#     [4, 5],      # Вторая гармоника
#     [6, 7]       # Остаточный шум
# ]
# L = int(1.5 * 4.6 / dt)  # L > T_max / dt
group_seq = [
    [1],         # Тренд
    [2, 3],      # Первая гармоника
    [4, 5]       # Вторая гармоника
]
grouped = mssa_analysis(signal, dt, L, group_seq)

# Визуализация результатов
plt.figure(figsize=(10, 6))
for i, (key, component) in enumerate(grouped.items()):
    plt.plot(t, component, label=f'Группа {key+1}')
plt.legend()
plt.title("Результаты MSSA анализа")
plt.xlabel("Время")
plt.ylabel("Амплитуда")
plt.grid()
plt.show()

N_loc = 1  # Количество компонент
# L =      # Параметр задержки
# L=int(4.6/0.05)
# L = 450
N = len(signal)  # Длина сигнала
N_ev = 9  # Число собственных значений
coef = 1  # Коэффициенты
pathout = './output'  # Путь для вывода
# group_seq = [[1]]  # Группировка компонент
group_seq = [
    [1, 2],
    [3, 4],         # Тренд
    [5, 6],      # Первая гармоника
    [7, 8],
]
components = mssa(dates, signal.reshape(1, -1), N_loc, N, L, N_ev, coef, pathout, group_seq)

spectr, omega = ampl_fft(signal, 0.05)
plt.plot(omega / (2 * np.pi), np.abs(spectr))
plt.xlabel("Frequency (cycles per year)")
plt.ylabel("Amplitude")
plt.title("Amplitude Spectrum (FFT)")
plt.show()

plt.plot(dates, signal, 'black', label="signal")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Amplitude")
plt.title("Generated Signal Components")
plt.show()

plt.plot(dates, garm_1, 'green', label="signal")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Amplitude")
plt.title("Generated Signal Components")
plt.show()

plt.plot(dates, garm_2, 'red', label="signal")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Amplitude")
plt.title("Generated Signal Components")
plt.show()

plt.plot(dates, garm_3, 'blue', label="signal")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Amplitude")
plt.title("Generated Signal Components")
plt.show()


scales = np.arange(1, 64)
# extent=[dates[0], dates[-1], scales[-1], scales[0]],
# Непрерывное вейвлет-преобразование
plt.figure(figsize=(10, 6))
coefficients, frequencies = pywt.cwt(signal, scales, 'morl')
plt.imshow(np.abs(coefficients), cmap='jet', aspect='auto')
plt.colorbar(label='Amplitude')
plt.title('Continuous Wavelet Transform (CWT) - Morlet Wavelet')
plt.xlabel('Time (Year)')
plt.ylabel('Scale')
plt.show()

# Преобразование масштаба в период
frequencies = pywt.scale2frequency('morl', scales) / 0.05  # Частота в циклах на год
periods = 1 / frequencies  # Период = 1 / частота

# Построение скейлограммы
plt.figure(figsize=(10, 6))
# dates = dates - 2000
plt.imshow(np.abs(coefficients), extent=[dates[0], dates[-1], periods[0], periods[-1]],
           aspect='auto', cmap='jet')
plt.colorbar(label='Amplitude')
plt.title('Scalogram (CWT) with Period on Y-axis')
plt.xlabel('Time (Year)')
plt.ylabel('Period (Years)')
plt.yscale('log')  # Логарифмическая шкала по периоду
plt.show()
