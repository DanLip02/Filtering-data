import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import inv

def predict_poly_last(dates, signal, pred_dates, degree):
    """
    Polynomial Trend Prediction
    dates: array of dates - массив временных меток
    signal: input signal - временной ряд (сигнал)
    pred_dates: dates for prediction - даты, для которых нужно построить прогноз
    degree: degree of polynomial - степень полинома
    """
    # 1. Аппроксимация временного ряда полиномом
    p_coef = np.polyfit(dates, signal, degree)

    # 2. Построение модели полинома на известных данных
    poly_model = np.polyval(p_coef, dates)

    # 3. Построение прогноза для будущих дат
    poly_pred = np.polyval(p_coef, pred_dates)

    # 4. Удаление полиномиального тренда из сигнала
    detrended_signal = signal - poly_model

    return detrended_signal, poly_pred

def harm_fit_homo(dates, signal, periods, date_start):
    """
    Least Squares Harmonic Fit
    Осуществляет гармонический анализ временного ряда методом наименьших квадратов.

    Параметры:
    - dates: массив временных меток.
    - signal: массив значений сигнала.
    - periods: массив предполагаемых периодов гармонических компонент.
    - date_start: начальная временная точка для расчета фаз.

    Возвращает:
    - coeffs: массив коэффициентов гармонической модели (амплитуды и фазы).
    - model: массив значений восстановленного сигнала (гармоническая модель).
    - covariance_matrix: ковариационная матрица коэффициентов (оценка погрешности).
    """
    N = len(signal)  # Количество точек сигнала
    M = len(periods)  # Количество периодов гармоник
    signal_centered = signal - np.mean(signal)  # Центрирование сигнала

    # Построение дизайн-матрицы
    A = np.ones((N, 2 * M))
    for i in range(N):  # Для каждой временной точки
        for j in range(M):  # Для каждого периода
            omega = 2 * np.pi / periods[j]  # Угловая частота гармоники
            A[i, 2 * j] = np.cos((dates[i] - date_start) * omega)
            A[i, 2 * j + 1] = np.sin((dates[i] - date_start) * omega)

    # Решение нормальных уравнений методом наименьших квадратов
    AtA = A.T @ A  # Матрица А транспонированная на А
    Atb = A.T @ signal_centered  # Матрица А транспонированная на центрированный сигнал
    coeffs = inv(AtA) @ Atb  # Коэффициенты гармонической модели

    # Построение гармонической модели
    model = A @ coeffs

    # Оценка дисперсии остатков
    residuals = signal_centered - model  # Остатки (разность сигнала и модели)
    variance = np.sum(residuals ** 2) / (N - 2 * M)  # Дисперсия остатков
    covariance_matrix = variance * inv(AtA)  # Ковариационная матрица коэффициентов

    return coeffs, model, covariance_matrix

def predict_harm_last(dates, signal, periods, pred_dates):
    """
    Harmonic Prediction
    dates: array of dates - массив временных меток
    signal: input signal - временной ряд (сигнал)
    periods: array of periods - массив периодов гармоник
    pred_dates: dates for prediction - даты, для которых нужно построить прогноз
    """
    # 1. Гармонический анализ на основе метода наименьших квадратов
    coeffs, harmonic_model, _ = harm_fit_homo(dates, signal, periods, dates[0])

    # 2. Удаление гармонического сигнала (остаточный сигнал)
    detrended_signal = signal - harmonic_model

    # 3. Прогнозирование гармонических компонент на новые даты
    pred_harmonics = np.zeros(len(pred_dates))
    for k, period in enumerate(periods):
        omega = 2 * np.pi / period
        pred_harmonics += coeffs[2 * k] * np.cos((pred_dates - dates[0]) * omega)
        pred_harmonics += coeffs[2 * k + 1] * np.sin((pred_dates - dates[0]) * omega)

    return detrended_signal, pred_harmonics


# def predict_ar(dates, signal, pred_dates, order):
#     """
#     Autoregression Prediction (AR)
#
#     dates: array-like
#         Временные метки для исходного сигнала.
#     signal: array-like
#         Исходный временной ряд для анализа.
#     pred_dates: array-like
#         Временные метки, на которых будет строиться предсказание.
#     order: int
#         Порядок модели авторегрессии (количество лагов).
#     """
#     from statsmodels.tsa.ar_model import AutoReg
#
#     # Построение модели авторегрессии
#     model = AutoReg(signal, lags=order, old_names=False).fit()
#
#     # Прогнозирование на новых временных точках
#     ar_pred = model.predict(start=len(signal), end=len(signal) + len(pred_dates) - 1)
#
#     return ar_pred, model.params

def predict_ar(mjd_sc, xin, mjd_pred, ar_order):

    """
    Model autoregression and predict next values.

    Parameters:
    mjd_sc (array): Time steps of the input signal.
    xin (array): Input data.
    mjd_pred (array): Dates to predict.
    ar_order (int): Order of the autoregression model.

    Returns:
    xar_pred (array): Predicted values.
    model_params (array): Coefficients of the autoregression model.
    """

    from statsmodels.tsa.ar_model import AutoReg
    # Centering the input signal
    mean_xin = np.mean(xin)
    xin_centered = xin - mean_xin

    # Fit AR model
    ar_model = AutoReg(xin_centered, lags=ar_order, old_names=False).fit()
    model_params = ar_model.params

    # Generate white noise for prediction
    noise_variance = ar_model.sigma2
    white_noise = np.sqrt(noise_variance) * np.random.randn(len(mjd_pred))

    # Initialize input values for prediction
    inp = np.flip(xin_centered[-ar_order:])

    # Predict N_p points
    xar_pred = []
    for j in range(len(mjd_pred)):
        pred_value = -np.dot(model_params[1:], inp) + white_noise[j]
        xar_pred.append(pred_value)
        inp = np.roll(inp, 1)
        inp[0] = pred_value

    # Add the mean back to the predictions
    xar_pred = np.array(xar_pred) + mean_xin

    # Plot results
    plt.plot(mjd_sc, xin, label="Original Signal")
    plt.plot(mjd_pred, xar_pred, color='red', label="AR Prediction")
    plt.legend()
    plt.title(f"Autoregression Prediction (Order {ar_order})")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.show()

    return xar_pred, model_params

def predict_poly(dates, signal, pred_dates, degree):
    """
    Predicts a polynomial model of a given degree and removes it from the signal.

    Parameters:
    dates (array): Initial dates of the signal.
    signal (array): Signal data.
    pred_dates (array): Dates for prediction.
    degree (int): Degree of the polynomial (e.g., 2 for quadratic, 1 for linear).

    Returns:
    tuple:
        - detrended_signal (array): Signal after removing the polynomial model.
        - poly_pred (array): Predictions based on the polynomial model for pred_dates.
    """
    # Fit a polynomial model to the data
    p_coef = np.polyfit(dates, signal, degree)

    # Evaluate the polynomial model on the original dates
    poly_model = np.polyval(p_coef, dates)

    # Detrend the signal by removing the polynomial model
    detrended_signal = signal - poly_model

    # Predict the polynomial values for the new dates
    poly_pred = np.polyval(p_coef, pred_dates)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(dates, signal, '--', label="Original Signal")
    plt.plot(dates, poly_model, '-', label=f"Polynomial Model (degree={degree})")
    plt.plot(pred_dates, poly_pred, label="Polynomial Prediction", color='red')
    plt.legend()
    plt.xlabel("Dates")
    plt.ylabel("Signal")
    plt.title("Polynomial Model and Prediction")
    plt.grid()
    plt.show()

    return detrended_signal, poly_pred

def predict_harm(dates, signal, periods, pred_dates):
    """
    Least Squares fit and prediction of harmonics.

    Parameters:
    dates (array): Initial dates.
    signal (array): Signal to be modeled.
    periods (array): Array of periods for harmonic fitting.
    pred_dates (array): Dates for prediction.

    Returns:
    tuple:
        - detrended_signal (array): Signal after removing harmonic model.
        - harm_pred (array): Predictions based on harmonic model for pred_dates.
    """
    # Center the signal
    signal_centered = signal - np.mean(signal)

    # Harmonic fit
    N = len(signal)
    M = len(periods)
    A = np.zeros((N, 2 * M))

    for i in range(N):
        for j in range(M):
            omega = 2 * np.pi / periods[j]
            A[i, 2 * j] = np.cos((dates[i] - dates[0]) * omega)
            A[i, 2 * j + 1] = np.sin((dates[i] - dates[0]) * omega)

    AtA = A.T @ A
    Atb = A.T @ signal_centered
    coeffs = np.linalg.inv(AtA) @ Atb

    # Compute harmonic model
    harmonic_model = A @ coeffs
    detrended_signal = signal - harmonic_model

    # Predict harmonics for new dates
    N_pred = len(pred_dates)
    harm_pred = np.zeros(N_pred)

    for k, period in enumerate(periods):
        omega = 2 * np.pi / period
        harm_pred += coeffs[2 * k] * np.cos((pred_dates - dates[0]) * omega)
        harm_pred += coeffs[2 * k + 1] * np.sin((pred_dates - dates[0]) * omega)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(dates, signal, label="Original Signal")
    plt.plot(dates, harmonic_model, label="Harmonic Model", linestyle='--')
    # plt.plot(dates, detrended_signal, label="Detrended Signal")
    plt.plot(pred_dates, harm_pred, label="Harmonic Prediction", color='red')
    plt.legend()
    plt.xlabel("Dates")
    plt.ylabel("Signal")
    plt.title("Harmonic Model and Prediction")
    plt.grid()
    plt.show()

    return detrended_signal, harm_pred

data = pd.read_excel('/Users/danilalipatov/Filtering-data/data/data_lab_5.xlsx')

time = data['Column1']
x = data['Column2']
signal = x
dt = time[1] - time[0]
N = len(signal)
signal_centered = signal - np.mean(signal)
acf = np.correlate(signal_centered, signal_centered, mode='full')[N - 1:] / N

# Plot ACF
# dt = 0.05
plt.plot(np.arange(N) * dt, acf)
plt.title("Autocovariance Function")
plt.show()

# Смещенная оценка АКФ
acf_bias = np.correlate(signal_centered, signal_centered, mode='full')[N - 1:] / N

# Несмещенная оценка АКФ
acf_unbias = np.correlate(signal_centered, signal_centered, mode='full')[N - 1:] / (N - np.arange(N))

# Построение графика
# dt = 0.05
plt.figure(figsize=(10, 6))

# График смещенной оценки АКФ
plt.plot(np.arange(N) * dt, acf_bias, label="Biased ACF")

# График несмещенной оценки АКФ
plt.plot(np.arange(N) * dt, acf_unbias, label="Unbaised ACF", linestyle='--')

plt.title("ACF")
plt.legend()
plt.show()

spectrum= np.fft.fft(acf)
freqs = np.fft.fftshift(np.fft.fftfreq(len(acf)))  #
plt.plot(freqs, np.abs(np.fft.fftshift(spectrum)))
plt.yscale('log')
plt.show()

spectrum= np.fft.fft(acf)
freqs = np.fft.fftshift(np.fft.fftfreq(len(acf)))  #
plt.plot(freqs, np.abs(np.fft.fftshift(spectrum)))
plt.show()
pred_dates = np.arange(2025, 2040, dt)
detrended_signal, poly_pred = predict_poly(time, signal, pred_dates, degree=3)

plt.plot(time, signal, label="Original Signal")
plt.plot(time, signal - detrended_signal, label="Polynomial Trend (Degree 4)")
plt.plot(pred_dates, poly_pred, label="Polynomial Prediction", linestyle="--")
plt.legend()
plt.show()

periods = [1, 4.6, 2]
pred_dates = np.arange(2025,  2040, dt)
detrended_signal_harm, harm_pred = predict_harm(time,  detrended_signal, periods, pred_dates)

# График детрендированного сигнала
plt.figure(figsize=(12, 6))
plt.plot(time, detrended_signal, label="Detranded signal", color='blue')
plt.plot(time, signal, label="Based signal", color='red', linestyle=':')
# График гармонического предсказания
plt.plot(pred_dates, harm_pred, label="Prediction of harm", color='orange', linestyle='--')

# Настройка легенды и подписей
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Harmonic Analysis and Prediction")
plt.legend()
plt.grid(True)
plt.show()

ar_order = 3
ar_pred, _ = predict_ar(time, signal, pred_dates, ar_order)
plt.plot(time, signal, label="Original Signal")
plt.plot(pred_dates, ar_pred, label="AR Prediction")
plt.show()

plt.plot(time, signal, label="Original Signal")
plt.plot(time, detrended_signal, label="Detrended Signal")
plt.plot(pred_dates, poly_pred, label="Polynomial Prediction")
plt.plot(pred_dates, ar_pred, label="AR Prediction")
plt.legend()
plt.show()

comb = ar_pred + poly_pred + harm_pred
plt.plot(time, signal, label="Original Signal")
plt.plot(pred_dates, comb, label="Combined Prediction")
plt.legend()
plt.show()
