import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


def trim_keypoint_frames(keypoint_frames, start_time, end_time, fps=25):
    """
    Trims a list of keypoint frames to include only the frames within the specified start and end times of the movement.

    Parameters:
        keypoint_frames (list): A list of dictionaries, where each dictionary contains keypoint data for a frame.
        start_time (float): The start time of the movement in seconds.
        end_time (float): The end time of the movement in seconds.
        fps (int): The frame rate at which keypoints were recorded, default is 30 FPS.

    Returns:
        list: A trimmed list of keypoint frames that fall within the start and end times.
    """
    # Calculate the frame indices to start and end the trimming
    start_index = int(start_time * fps)
    end_index = int(end_time * fps)

    # Trim the list of keypoint frames
    trimmed_frames = keypoint_frames[start_index:end_index + 1]

    return trimmed_frames


def plot_trimmed_angles(time_series, start_times, end_times, fps=30):
    """
        Plots the movement time series and marks the start and end of repetitions.

        Parameters:
            time_series (list or array): The time or frame numbers for the data points.
            start_times (list): A list of start times or frame numbers for the repetitions.
            end_times (list): A list of end times or frame numbers for the repetitions.

        """
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, label='Angle over Time')
    plt.title('Movement Analysis with Repetition Markers')
    plt.xlabel('Time or Frame')
    plt.ylabel('Angle (degrees)')

    start_index = int(start_times * fps)
    end_index = int(end_times * fps)

    # Mark the start and end times of the repetitions
    plt.axvline(x=start_index, color='green', linestyle='--', linewidth=2, label='Start of Repetition')
    plt.axvline(x=end_index, color='red', linestyle='--', linewidth=2, label='End of Repetition')
    plt.grid(True)
    plt.legend()
    plt.show()


def define_skeleton_connections():
    """
    Define the connections between keypoints in the skeleton.

    Returns:
        list: A list of tuples, where each tuple contains the indices of the keypoints to connect.
    """
    skeleton_connections = [
        ("nose", "left_eye"),
        ("left_eye", "left_ear"),
        ("nose", "right_eye"),
        ("right_eye", "right_ear"),
        ("left_ear", "left_shoulder"),
        ("right_ear", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    return skeleton_connections


def time_series_angles(angles, title):
    """
    This function plots a time series of angles.

    Parameters:
    angles (list): A list of angles. Each angle is a float representing the angle in degrees.
    title (str): The title of the plot. This will be appended to 'Serie Temporal de los Ángulos del ' to form the full title.
    """
    plt.figure(figsize=(30, 6))
    plt.plot(angles, marker='o', linestyle='-', color='blue')
    plt.title('Serie Temporal de los Ángulos del ' + title)
    plt.xlabel('Frame')
    plt.ylabel('Ángulo (grados)')
    plt.grid(True)
    plt.show()


def clean_time_series(ts):
    """
    This function cleans a time series of angles.

    Parameters:
    ts (list): A list of angles. Each angle is a float representing the angle in degrees or None.
    Returns:
    list: A cleaned list of angles with None or NaN values removed.
    """
    clean_ts = [angle for angle in ts if angle is not None]
    return clean_ts


def time_series_smooth(angles, window_length=15, poly_order=3):
    """
    This function smooths a time series of angles using the Savitzky-Golay filter.

    Parameters:
    angles (list): A list of angles. Each angle is a float representing the angle in degrees.
    window_length (int): The length of the filter window (i.e., the number of coefficients). This value should be a positive odd integer. Default is 15.
    polyorder (int): The order of the polynomial used to fit the samples. This value should be less than window_length. Default is 3.

    The function performs the following steps:
    - Applies the Savitzky-Golay filter to the list of angles.
    - Returns the smoothed time series.

    Returns:
    list: A smoothed list of angles.
    """
    angles_smooth = savgol_filter(angles, window_length=window_length,
                                  polyorder=poly_order)  # Ajustar los parámetros según sea necesario
    return angles_smooth


def normalize_time_series(angles):
    """
    This function normalizes a time series of angles using the Min-Max normalization method.

    Parameters:
    angles (list): A list of angles. Each angle is a float representing the angle in degrees.

    The function performs the following steps:
    - Applies the Min-Max normalization to the list of angles. This scales the angles to a range of 0 to 1.
    - Returns the normalized time series.

    Returns:
    list: A normalized list of angles with values ranging from 0 to 1.
    """
    norm_angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles))
    return norm_angles


def find_peaks_valleys(angulos, hp, hv, dp, dv, pp, pv):
    """
    This function identifies the peaks (maximum points) and valleys (minimum points) in a time series of angles.

    Parameters:
    angulos (list): A list of angles. Each angle is a float representing the angle in degrees.
    hp (float): The minimum height to consider a peak.
    hv (float): The maximum height (in negative) to consider a valley.
    dp (float): The minimum number of samples between peaks.
    dv (float): The minimum number of samples between valleys.
    pp (float): The required prominence of peaks. Prominence is a measure of how much a peak stands out due to its intrinsic height and its location relative to other peaks.
    pv (float): The required prominence of valleys.

    The function performs the following steps:
    - Detects peaks in the list of angles. A peak is considered if its height is greater than hp, its distance from the nearest peak is greater than dp, and its prominence is greater than pp.
    - Detects valleys in the list of angles by inverting the signal. A valley is considered if its height is less than hv, its distance from the nearest valley is greater than dv, and its prominence is greater than pv.
    - Returns the indices of the detected peaks and valleys.

    Returns:
    tuple: A tuple of two lists. The first list contains the indices of the detected peaks. The second list contains the indices of the detected valleys.
    """
    # Detect peaks (maximum points)
    peaks, peak_heights = find_peaks(angulos, height=hp, distance=dp, prominence=pp)

    # Detect valleys (minimum points) by inverting the signal
    valleys, valleys_heights = find_peaks(-angulos, height=-hv, distance=dv, prominence=pv)

    return peaks, valleys


def visualize_peaks(angles, peaks, valleys):
    x_axis = range(len(angles))
    plt.figure(figsize=(14, 7))
    plt.plot(x_axis, angles, label='Ángulos Suavizados')  # Usar x_axis como eje X
    plt.plot(peaks, angles[peaks], "x", color='r',
             label='Picos Detectados')  # Graficar los picos usando los mismos índices
    plt.plot(valleys, angles[valleys], "o", color='r',
             label='Valles Detectados')  # Graficar los picos usando los mismos índices

    plt.title('Serie Temporal de Ángulos y Picos Detectados')
    plt.xlabel('Índice del Frame')
    plt.ylabel('Ángulo')
    plt.legend()
    plt.show()
