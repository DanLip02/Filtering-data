import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn
from scipy.linalg import inv
from numpy.polynomial.polynomial import Polynomial


def harm_fit_homo(dates, signal, periods, date_start):
    """
    Least Squares Harmonic Fit
    dates: array of dates
    signal: signal data
    periods: array of periods
    date_start: starting date
    """
    N = len(signal)
    M = len(periods)
    signal_centered = signal - np.mean(signal)

    # Build design matrix
    A = np.ones((N, 2 * M))
    for i in range(N):
        for j in range(M):
            omega = 2 * np.pi / periods[j]
            A[i, 2 * j] = np.cos((dates[i] - date_start) * omega)
            A[i, 2 * j + 1] = np.sin((dates[i] - date_start) * omega)

    # Solve normal equations
    AtA = A.T @ A
    Atb = A.T @ signal_centered
    coeffs = inv(AtA) @ Atb

    # Predicted signal
    model = A @ coeffs

    # Variance estimation
    residuals = signal_centered - model
    variance = np.sum(residuals ** 2) / (N - 2 * M)
    covariance_matrix = variance * inv(AtA)

    return coeffs, model, covariance_matrix


def predict_poly(dates, signal, pred_dates, degree):
    """
    Polynomial Trend Prediction
    dates: array of dates
    signal: input signal
    pred_dates: dates for prediction
    degree: degree of polynomial
    """
    p_coef = np.polyfit(dates, signal, degree)
    poly_model = np.polyval(p_coef, dates)
    poly_pred = np.polyval(p_coef, pred_dates)

    detrended_signal = signal - poly_model

    return detrended_signal, poly_pred


def predict_harm(dates, signal, periods, pred_dates):
    """
    Harmonic Prediction
    dates: array of dates
    signal: input signal
    periods: array of periods
    pred_dates: dates for prediction
    """
    coeffs, harmonic_model, _ = harm_fit_homo(dates, signal, periods, dates[0])
    detrended_signal = signal - harmonic_model

    # Predict harmonics for new dates
    pred_harmonics = np.zeros(len(pred_dates))
    for k, period in enumerate(periods):
        omega = 2 * np.pi / period
        pred_harmonics += coeffs[2 * k] * np.cos((pred_dates - dates[0]) * omega)
        pred_harmonics += coeffs[2 * k + 1] * np.sin((pred_dates - dates[0]) * omega)

    return detrended_signal, pred_harmonics


def predict_ar(dates, signal, pred_dates, order):
    """
    Autoregression Prediction
    dates: array of dates
    signal: input signal
    pred_dates: dates for prediction
    order: order of autoregression
    """
    from statsmodels.tsa.ar_model import AutoReg

    model = AutoReg(signal, lags=order, old_names=False).fit()
    ar_pred = model.predict(start=len(signal), end=len(signal) + len(pred_dates) - 1)

    return ar_pred, model.params


def spect_fftn(dates, signal):
    """
    Spectrum Analysis using FFT
    dates: array of equally spaced dates
    signal: input signal
    """
    N = len(signal)
    dt = dates[1] - dates[0]

    spectrum = fftn(signal)
    freqs = np.fft.fftfreq(N, d=dt)

    return spectrum, freqs


# Main script for Lab5
def lab5_main():
    filename = f"/Users/danilalipatov/Downloads/Lab5 2/v3/amon.us.long.dat"

    # Load data
    data = np.loadtxt(filename).T
    dates = data[0]
    signal = data[1]

    # Autocovariance Function (ACF)
    N = len(signal)
    signal_centered = signal - np.mean(signal)
    acf = np.correlate(signal_centered, signal_centered, mode='full')[N - 1:] / N

    # Plot ACF
    dt = 0.05
    plt.plot(np.arange(N) * dt, acf)
    plt.title("Autocovariance Function")
    plt.show()

    # Power Spectral Density
    spectrum, freqs = spect_fftn(dates, acf)
    plt.plot(1 / freqs[1:N // 2], np.abs(spectrum[1:N // 2]))
    plt.title("Power Spectral Density")
    plt.show()

    # Polynomial Trend Removal
    pred_dates = np.arange(dates[-1], 2050, dt)
    detrended_signal, poly_pred = predict_poly(dates, signal, pred_dates, degree=3)
    plt.plot(dates, signal, label="Original Signal")
    plt.plot(dates, detrended_signal, label="Detrended Signal")
    plt.plot(pred_dates, poly_pred, label="Polynomial Prediction")
    plt.legend()
    plt.show()

    # Harmonic Analysis
    periods = [1, 4.6, 0.5]
    detrended_signal_harm, harm_pred = predict_harm(dates, detrended_signal, periods, pred_dates)
    plt.plot(dates, detrended_signal, label="Detrended Signal")
    plt.plot(pred_dates, harm_pred, label="Harmonic Prediction")
    plt.legend()
    plt.show()

    # Autoregression Prediction
    ar_order = 5
    ar_pred, _ = predict_ar(dates, detrended_signal_harm, pred_dates, ar_order)
    plt.plot(dates, detrended_signal_harm, label="Signal After Harmonics")
    plt.plot(pred_dates, ar_pred, label="AR Prediction")
    plt.legend()
    plt.show()

    # Combine Predictions
    combined_pred = harm_pred + poly_pred + ar_pred
    plt.plot(dates, signal, label="Original Signal")
    plt.plot(pred_dates, combined_pred, label="Combined Prediction")
    plt.legend()
    plt.show()

    # Save predictions
    output_file = f"AAM_prediction.dat"
    np.savetxt(output_file, np.column_stack((pred_dates, combined_pred)), fmt="%10.8e", header="Date Prediction")


if __name__ == "__main__":
    lab5_main()
