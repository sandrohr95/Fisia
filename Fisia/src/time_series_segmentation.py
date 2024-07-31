import matplotlib.pyplot as plt
import numpy as np


def segment_series(data, peaks, valleys, expected_reps):
    """
    Segment the series into repetitions based on valleys and ensure each segment has proper start and end points.

    Parameters:
        data (np.array): The series data, e.g., angles or any other measurement.
        peaks (list): Indices of peaks in the data.
        valleys (list or np.array): Indices of valleys in the data which are considered start and end of each repetition.
        expected_reps (int): The expected number of repetitions.

    Returns:
        list: List of segmented data based on valleys.
    """
    # Convert valleys to list if it's an np.array
    valleys = list(valleys)

    # Ensure the first element in valleys is the start of a repetition
    if valleys[0] > peaks[0]:
        print(peaks[0])
        # If the first valley comes after the first peak, prepend the start of the series as a valley
        valleys.insert(0, 0)

    # Ensure there is a valley at the end of the series
    if valleys[-1] < peaks[-1]:
        # If the last valley is before the last peak, append the end of the series as a valley
        valleys.append(len(data) - 1)

    # Generate segments based on valleys
    segments = [data[valleys[i]:valleys[i + 1]] for i in range(len(valleys) - 1)]

    # Check if the number of segments matches the expected number of repetitions
    if len(segments) < expected_reps:
        # Adjust the valleys list if there are fewer segments than expected
        segments = [data[valleys[i]:valleys[i + 1]] for i in range(len(valleys) - 1)]

    return segments


def plot_generated_segments(segments, title='Segmentación de Series Temporales', xlabel='Frames', ylabel='Ángulo'):
    """
    Plot the time series data with each generated segment highlighted using fill_between.

    Parameters:
        segments (list of lists): A list where each sub-list contains the data points for a segment.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
    """
    plt.figure(figsize=(14, 7))

    # Plot each segment with a different color and use fill_between for visualization
    start_index = 0
    colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))  # Generate color map for distinction

    for i, segment in enumerate(segments):
        end_index = start_index + len(segment)
        x_range = range(start_index, end_index)

        # Plot each segment
        plt.plot(x_range, segment, color=colors[i])
        # Fill area under the plot for each segment
        plt.fill_between(x_range, segment, color=colors[i], alpha=0.5)

        start_index = end_index

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([f'Segmento {i + 1}' for i in range(len(segments))], loc='upper right')
    plt.show()
