import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.fftpack import fft, fftfreq

def polynom_coef(time, signal, deg=2):
    return np.polyfit(time, signal, deg)

def polynom(coef, time):
    return np.polyval(coef, time)

def autocovariance(signal_):
    N = len(signal_)
    acf = np.correlate(signal_, signal_, mode='full')/ N
    lags = signal.correlation_lags(len(signal_), len(signal_))
    return acf, lags

def ccf_(signal1, signal2):
    p = signal1
    q = signal2
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))
    c = np.correlate(p, q, 'full')
    return c

def plot_polynom(time, signal, polyn):
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, 'b.', label='Original Signal X(t)')
    plt.plot(time, polyn, 'r-', label='Fitted Polynomial (degree 2)')
    plt.title('Polynomial Fit (Degree 2) to X(t)')
    plt.xlabel('Years')
    plt.ylabel('X(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_autocov(acf_mean, acf_detrended, lags_mean, lags_detrended):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(lags_mean, acf_mean, label='ACF (mean-subtracted)')
    plt.plot(lags_detrended, acf_detrended, label='ACF (polynomial-subtracted)')
    plt.title('Autocovariance Functions')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_spectrum(fft_mean, fft_detrended, fft_original, freqs, freqs_mean, freqs_detrended):
    plt.subplot(2, 1, 2)
    plt.plot(freqs_mean, fft_mean, label='FFT of ACF (mean-subtracted)')
    plt.plot(freqs_detrended, fft_detrended, label='FFT of ACF (polynomial-subtracted)')
    plt.plot(freqs, fft_original, label='FFT of original X')
    plt.title('Spectral Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def ccf_plot(lags, ccf):
    fig, ax =plt.subplots(figsize=(9, 6))
    ax.plot(lags, ccf)
    ax.set_ylabel('Correlation', weight='bold', fontsize=12)
    ax.set_xlabel('Time Lags', weight='bold', fontsize = 12)
    plt.legend()
    plt.show()

def csf_plot(frequencies, cross_specturm):
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies[:len(frequencies) // 2], np.abs(cross_spectrum)[:len(frequencies) // 2],
             label='Cross-Spectrum')
    plt.title('Cross-Spectrum between X and Y (without lags)')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.show()

def autocorr(x, biased=True):
    n = len(x)
    result = np.correlate(x, x, mode='full')  # Полная корреляция
    # result = result[n-1:]  # Оставляем только правую часть
    if biased:
        result /= n  # Смещённая АКФ
    else:
        result = result[n - 1:]
        result /= (n - np.arange(n))  # Несмещённая АКФ
    return result

def plot_ARMA(acf_biased, acf_unbiased):
    plt.figure(figsize=(10, 5))
    plt.plot(acf_biased, label='Biased ACF (Смещённая АКФ)')
    plt.plot(acf_unbiased, label='Unbiased ACF (Несмещённая АКФ)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Autocorrelation Function (Biased vs Unbiased)')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_spectr_ARMA(frequencies, spectral_density):
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies[:n_samples // 2], spectral_density[:n_samples // 2], label='Spectral Density')
    plt.title('Spectral Density of ARMA Process')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = pd.read_excel('/Users/danilalipatov/PycharmProjects/HSE_lab_1/data_lab_1.xlsx')
    print(data)
    # Разделение данных
    YEARS = data['YEAR']
    X = data['X']
    Y = data['Y']

    #Task N. 1
    coef = polynom_coef(YEARS, X)
    polyn = polynom(coef, YEARS)
    plot_polynom(YEARS, X, polyn)

    #Task N. 2
    X_mean = X - np.mean(X)
    X_detrended = X - polyn
    acf_mean, lags_mean = autocovariance(X_mean)
    acf_detrended, lags_detrended = autocovariance(X_detrended)
    fft_mean = np.abs(np.fft.fft(acf_mean)) / len(acf_mean)
    fft_detrended = np.abs(np.fft.fft(acf_detrended)) / len(acf_detrended)
    fft_original = np.abs(np.fft.fft(X)) / len(X)
    dT = YEARS[1] - YEARS[0]  # Шаг времени
    freqs = np.fft.fftfreq(len(X), dT)
    freqs_mean = np.fft.fftfreq(len(acf_mean), dT)
    freqs_detrended = np.fft.fftfreq(len(acf_detrended), dT)
    positive_frequencies = freqs
    plot_autocov(acf_mean, acf_detrended, lags_mean, lags_detrended)
    plot_spectrum(fft_mean, fft_detrended, fft_original, positive_frequencies, freqs_mean, freqs_detrended)

    #Task N. 3
    stop = 0
    cross_cor = ccf_(X, Y)
    print(cross_cor)
    lags = signal.correlation_lags(len(X), len(Y))
    ccf_plot(lags, cross_cor)
    Y_mean = Y - np.mean(Y)
    X_fft = np.fft.fft(X_mean)
    Y_fft = np.fft.fft(Y_mean)

    # Кросс-спектр: произведение FFT X на комплексно-сопряжённое FFT Y
    cross_spectrum = X_fft * np.conj(Y_fft)
    frequencies = np.fft.fftfreq(len(X_mean), d=1)  # Частоты для оси
    csf_plot(frequencies, cross_spectrum)

    f, Pxy = signal.csd(X_mean, Y_mean)
    plt.semilogy(f, np.abs(Pxy))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('CSD [V**2/Hz]')
    plt.show()


    #Task N. 4
    ar = np.array([0.5, -0.25])  # Коэффициенты AR (авторегрессия)
    ma = np.array([0.5, -0.3])  # Коэффициенты MA (скользящее среднее)

    roots = np.roots(ar)
    print(f"Корни полинома AR: {roots}")
    if np.all(np.abs(roots) > 1):
        print("Модель ARMA устойчива (все корни вне единичного круга).")
    else:
        print("Модель ARMA неустойчива!")
    arma_process = ArmaProcess(ar, ma)
    n_samples = 1000
    ARMA = arma_process.generate_sample(nsample=n_samples)
    plt.plot(range(n_samples), ARMA)
    plt.show()

    acf_biased = autocorr(ARMA, biased=True)
    acf_unbiased = autocorr(ARMA, biased=False)
    plot_ARMA(acf_biased, acf_unbiased)

    fft_result = fft(ARMA)
    frequencies = fftfreq(n_samples)
    spectral_density = np.abs(fft_result)
    plot_spectr_ARMA(frequencies, spectral_density)








