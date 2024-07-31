import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis


def get_keypoint_coordinates(data, name):
    """Encuentra las coordenadas de un keypoint por su nombre."""
    for item in data:
        if item['name'] == name:
            return np.array([item['x'], item['y']])
    return None


def calculate_angle(p1, p2, p3):
    """Calcula el ángulo formado por tres puntos.
    p1, p2, p3: Son tuples o listas que representan las coordenadas (x, y) de cada punto.
    """
    # Vectores
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    # Producto punto y magnitudes de los vectores
    dot_prod = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    # Cálculo del ángulo usando el producto punto
    angle = np.arccos(dot_prod / (mag1 * mag2)) * (180 / np.pi)
    return angle


def calculate_movement_angles(all_frames_keypoints, first_keypoint, second_keypoint, thrid_keypoint):
    """
    This function calculates the angles of movement for a sequence of frames.

    Parameters:
    all_frames_keypoints (list): A list of dictionaries. Each dictionary represents a frame and contains the keypoints.
    first_keypoint (str): The name of the first keypoint.
    second_keypoint (str): The name of the second keypoint.
    thrid_keypoint (str): The name of the third keypoint.

    Returns:
    list: A list of angles calculated for each frame. If the angle cannot be calculated for a frame, None is appended.

    """
    angles = []
    for frame in all_frames_keypoints:
        # Get the coordinates of the keypoints for the current frame
        thrid = get_keypoint_coordinates(frame, thrid_keypoint)
        second = get_keypoint_coordinates(frame, second_keypoint)
        first = get_keypoint_coordinates(frame, first_keypoint)

        # If all keypoints are found, calculate the angle and append it to the list
        if thrid is not None and second is not None and first is not None:
            angle = calculate_angle(first, second, thrid)
            angles.append(angle)
        else:
            # If any keypoint is not found, append None to the list
            angles.append(None)  # Para frames donde no se pueda calcular el ángulo

    return angles


def calculate_distance(keypoint1, keypoint2):
    """
    Calcula la distancia euclidiana entre dos keypoints en un plano (x, y).

    Args:
    - keypoint1: Tupla o lista con las coordenadas (x, y) del primer keypoint.
    - keypoint2: Tupla o lista con las coordenadas (x, y) del segundo keypoint.

    Returns:
    - La distancia euclidiana entre los dos keypoints.
    """
    distance = np.linalg.norm(keypoint1 - keypoint2)
    return distance


def calculate_movement_positions(all_frames_keypoints, keypoint1, keypoint2):
    distances = []
    for frame in all_frames_keypoints:
        first = get_keypoint_coordinates(frame, keypoint1)
        second = get_keypoint_coordinates(frame, keypoint2)

        if second is not None and first is not None:
            distance = calculate_distance(first, second)
            distances.append(distance)
        else:
            distances.append(None)  # Para frames donde no se pueda calcular el ángulo

    return distances


def calculate_statistics_variables(segments):
    """
    This function calculates various statistical variables for each segment in the provided list.

    Parameters:
    segments (list): A list of segments. Each segment is a list of numerical values.

    Returns:
    list: A list of dictionaries. Each dictionary contains the calculated statistical variables for a segment.

    """
    stats = []
    for index, segment in enumerate(segments):
        segment = np.array(segment)
        # Calculate the interquartile range
        iqr = np.percentile(segment, 75) - np.percentile(segment, 25)
        # Calculate the total displacement
        total_displacement = np.sum(np.abs(np.diff(segment)))

        # Calculate the entropy of the signal
        from scipy.stats import entropy
        hist, bin_edges = np.histogram(segment, bins=10, density=True)
        signal_entropy = entropy(hist)

        # Calculate the smoothness of the signal
        smoothness = np.mean(np.diff(segment, n=2) ** 2)

        # Calculate the symmetry of the signal
        symmetry = np.mean(np.abs(segment - np.mean(segment)))

        # Append the calculated statistical variables to the list
        stats.append({
            'Repetition': index + 1,
            'min': np.min(segment),
            'max': np.max(segment),
            'median': np.median(segment),
            'duration': len(segment),
            'standardDeviation': segment.std(),
            'mean': segment.mean(),
            'range': segment.max() - segment.min(),
            'variance': segment.var(),
            'CoV': segment.std() / segment.mean(),  # coefficient of variation
            'skewness': skew(segment),  # Asimetría
            'kurtosis': kurtosis(segment),  # Curtosis,
            'IQR': iqr,
            'TotalDisplacement': total_displacement,
            'Entropy': signal_entropy,
            'Smoothness': smoothness,
            'Symmetry': symmetry
        })
    return stats
