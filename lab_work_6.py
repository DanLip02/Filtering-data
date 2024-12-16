import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

def pantelleev_filter(t, omega_0):
    """
    Pulse response of Panteleev filter.

    Parameters:
    t (array): Initial dates.
    omega_0 (int/float): half-width parameter ω0

    Returns:
    array:
        - Impulse response of the filter
    """
    coef = omega_0 / (2 * np.sqrt(2))
    exp_part = np.exp(-omega_0 * np.abs(t) / np.sqrt(2))
    cos_part = np.cos(omega_0 * t / np.sqrt(2))
    sin_part = np.sin(omega_0 * np.abs(t) / np.sqrt(2))
    return coef * exp_part * (cos_part + sin_part)

def pantelleev_ach(omega, omega_0):
    """
        Amplitude-frequency response (AFR) of the Panteleev filter.
        How does the amplitude 𝑊(𝜔) change with increasing frequency 𝜔

        Parameters:
        omega(t) (array): Initial dates.
        omega_0 (int/float): half-width parameter ω0

        Returns:
        array:
            - AFR
        """
    return (omega_0**4) / (omega**4 + omega_0**4)

def plot_filter(H, h, dt):
    freqs = fftfreq(len(h), dt)
    # H = np.abs(fft(h)) / len(h)
    plt.plot(freqs[:len(h) // 2], H[:len(h) // 2])
    plt.title("Panteleev filter frequency response")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid()

def plot_signal(signal, filtered_signal, t):
    plt.plot(t, signal, label="Based signal")
    plt.plot(t, filtered_signal, label="Filtered signal")
    plt.title("Comparing of signals")
    plt.legend()
    plt.grid()

def plot_spectrum(signal, dt, title="Spectrum of signal"):
    n = len(signal)
    freqs = fftfreq(n, dt)
    fft_values = np.abs(fft(signal)) / n
    plt.plot(freqs[:n // 2], fft_values[:n // 2])
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid()

if __name__ == '__main__':
    omega_0 = 6.5  # Filter half width

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

    # Impulse response of the filter
    h = pantelleev_filter(t - t[N // 2], omega_0)

    # Convolution of a signal with a filter
    filtered_signal = np.convolve(signal, h, mode='same') * dt

    # Visualization of signals
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plot_signal(signal, filtered_signal, t)

    # Spectrums
    plt.subplot(2, 2, 2)
    plot_spectrum(signal, dt, title="Spectrum of the original signal")

    plt.subplot(2, 2, 3)
    plot_spectrum(filtered_signal, dt, title="Filtered signal spectrum")

    # Filter frequency response
    plt.subplot(2, 2, 4)
    # H = np.abs(fft(h)) / len(h)
    H = pantelleev_ach(t, omega_0)
    plot_filter(H, h, dt)
    plt.tight_layout()
    plt.show()