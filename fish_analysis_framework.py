import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import leastsq
import scipy.signal as signal
import openpyxl


def timeframe_nums(df_readings):
    """
    Extracts the timeframe numbers from the dataset based on the information in header columns.

    :param df_readings: Readings from the dataset into pandas DataFrame
    :return: timeframe
    """
    timeframe = []
    # Loop through the columns in the dataset to get timeframes
    for col in df_readings.columns:
        # Check if the column header contains 'x'
        if 'x' in col:
            # If it does, extract the timeframe number by removing the 'x' character
            timeframe.append(int(col[:-1]))
    return timeframe


def get_info_from_dataset(df_path):
    """
    Extracts information regarding the subject fish from the dataset file path.

    :param df_path: The path of the dataset file
    :return: full_filename, title, fish_length, tailbeat_speed
    """

    # Extract the filename from the file path
    full_filename = os.path.basename(df_path)

    # Split the filename into parts using the '.' delimiter
    parts = full_filename.split('.')

    # Assign the required parts to Fish variables
    title = parts[0]
    fish_length = parts[2].replace("cm", "")
    tail_beat_speed = parts[3]

    return full_filename, title, fish_length, tail_beat_speed


def get_row_indices(fish_midline, proportions):
    """
    Calculates the corresponding robot joint locations on the fish midline using locations of the joints based
    on body length ratio. As rows in the datasets represent datapoints, row indices gives us the datapoint index.

    :param fish_midline: Number of datapoints on the fish midline
    :param proportions: Location of robot joints on robot body in terms of body proportion
    :return: row_indices
    """
    row_indices = []
    for prop in proportions:
        # Calculate robot segment datapoint
        robot_seg_datapoint = fish_midline * prop
        row_indices.append(round(robot_seg_datapoint, 0))
    return row_indices


def get_robot_coordinates(df_readings, row_indices):
    """
    Extracts the x and y coordinates of the robot for the specified row indices.

    :param df_readings: Readings from the dataset into pandas DataFrame
    :param row_indices: Rows in the dataset that represent robot datapoints
    :return: x_robot, y_robot
    """
    x_robot = []
    y_robot = []
    for i, timeframes in enumerate(timeframe_nums(df_readings)):
        # Assign x and y coordinates to variables
        x_col = f"{timeframes}x"
        y_col = f"{timeframes}y"

        # Find x and y coordinates at row indices representing robot datapoints
        x_vals = df_readings.loc[row_indices, x_col].tolist()
        y_vals = df_readings.loc[row_indices, y_col].tolist()
        x_robot.append(x_vals)
        y_robot.append(y_vals)
    return x_robot, y_robot


def get_fish_coordinates(df_readings):
    """
    Extracts the x and y coordinates of the fish from the dataset.

    :param df_readings: Readings from the dataset into pandas DataFrame
    :return: x_fish, y_fish
    """
    x_fish = []
    y_fish = []
    for i, timeframes in enumerate(timeframe_nums(df_readings)):
        # Assign x and y coordinates to variables
        x_col = f"{timeframes}x"
        y_col = f"{timeframes}y"
        x_vals = list(df_readings[x_col])
        y_vals = list(df_readings[y_col])
        x_fish.append(x_vals)
        y_fish.append(y_vals)
    return x_fish, y_fish


def calculate_angles(df_readings, row_indices):
    """
    Calculates angle changes of each robot segment over timeframes using linear regression.
    Creates an array of arrays with angle values in degrees for each segment.

    :param df_readings: Readings from the dataset into pandas DataFrame
    :param row_indices: Rows in the dataset that represent robot datapoints
    :return: grouped_angle_array
    """
    # Get x and y coordinates of robot plot
    robot_coordinates = get_robot_coordinates(df_readings, row_indices)
    x_robot, y_robot = robot_coordinates

    angle_array = []
    grouped_angle_array = []

    for i in range(len(x_robot)):
        for j in range(len(x_robot[i]) - 1):
            # Part of the robot graph that we want to calculate linear regression
            x = np.array([x_robot[i][j], x_robot[i][j + 1]])
            y = np.array([y_robot[i][j], y_robot[i][j + 1]])

            # Fit a linear regression line to the data
            m, b = np.polyfit(x, y, 1)

            # Calculate the angle of the slope
            angle = np.arctan(m) * 180 / np.pi
            angle_array.append(angle)

    # Group the arrays into number of robot segments
    for i in range(0, len(angle_array), (len(row_indices) - 1)):
        grouped_angle_array.append(angle_array[i:i + (len(row_indices) - 1)])
    return grouped_angle_array


def calculate_errors(df_readings, row_indices):
    """
    Calculates error between fish midline and robot segments. Finds the y-coordinates along the robot plot for distance
    calculation between fish midline plot and robot plot. Creates an array of arrays with error values for each segment.

    :param df_readings: Readings from the dataset into pandas DataFrame
    :param row_indices: Rows in the dataset that represent robot datapoints
    :return: grouped_error_array
    """
    # Get x and y coordinates of robot and fish midline
    robot_coordinates = get_robot_coordinates(df_readings, row_indices)
    x_robot, y_robot = robot_coordinates

    fish_coordinates = get_fish_coordinates(df_readings)
    x_fish, y_fish = fish_coordinates

    # Finding y values on the corresponding x coordinates with fish
    x_robot_array = np.array(x_fish)

    # Creating array of fish y coordinates and slicing it at where x_robot final segment is
    y_fish_array = np.array(y_fish)

    error_array = []
    grouped_error_array = []
    # Iterate through each segment of the robot and calculate the maximum distance between fish midline and robot plots
    for i in range(len(x_robot)):
        for j in range(len(x_robot[i]) - 1):
            # Part of the robot graph that we want to calculate linear regression
            x = np.array([x_robot[i][j], x_robot[i][j + 1]])
            y = np.array([y_robot[i][j], y_robot[i][j + 1]])

            # Fit a linear regression line to the data
            m, b = np.polyfit(x, y, 1)

            # Slicing the x_find array so it only includes the x coordinates within the range of the part of the
            # robot graph
            mask_x = (x_robot_array[i] <= x[1])
            mask_x &= (x_robot_array[i] >= x[0])

            # Use the boolean mask to slice the array
            sliced_robot = x_robot_array[i][mask_x]

            # Finding y values along x coordinates on the line graphx
            y_find = (m * sliced_robot) + b

            # Creating array of x and y coordinates of the robot on the specific part of the graph
            robot_array = np.array(list(zip(sliced_robot, y_find)))

            # Slicing the x_find array, so it only includes the x coordinates within the range of the part of the
            # robot graph
            mask_y = (y_fish_array[i] <= y[1])
            mask_y &= (y_fish_array[i] >= y[0])
            sliced_fish = y_fish_array[i][mask_x]

            # Array of fish coordinates within the parth of the graph range
            fish_array = np.array(list(zip(sliced_robot, sliced_fish)))

            # Subtracting y coordinates of both plots
            distance = abs(sliced_fish - y_find)
            # Find the maximum difference in y_diff
            max_diff = np.max(distance)
            # Find the index of the maximum difference in y_diff
            max_diff_index = np.argmax(distance)

            error_array.append(max_diff)

    # Group the arrays into number of robot segments and add 0s to arrays if the length of error array is less
    # than number of robot segments
    if len(error_array) % (len(row_indices) - 1) != 0:
        num_to_add = (len(row_indices) - 1) - (len(error_array) % (len(row_indices) - 1))
        error_array += [0] * num_to_add

    for i in range(0, len(error_array), (len(row_indices) - 1)):
        grouped_error_array.append(error_array[i:i + (len(row_indices) - 1)])
    return grouped_error_array


def segment_growing_opt(df_readings, error_threshold):
    """
    Segment growing method for calculating optimal robot joint configuration.This method was implemented based on the
    work of Fetherstonhaugh et al. (2021).

    :param df_readings: Readings from the dataset into pandas DataFrame
    :param error_threshold: User specified error threshold value
    :return: seg_positions
    """
    # Initialize the segment positions
    seg_positions = [0]
    current_error = 0
    new_position = 0

    # Get the number of datapoints in the dataset by calculating length of first fish x coordinate row
    x, y = get_fish_coordinates(df_readings)
    num_datapoints = len(x[0])

    while new_position < num_datapoints:
        # Add a new segment next to the last one and move it away until the error threshold is reached
        last_position = seg_positions[-1]
        new_position = last_position + 1
        while current_error < float(error_threshold) and new_position < num_datapoints:
            row = [round(last_position), round(new_position)]
            errors = calculate_errors(df_readings, row)
            current_error = np.mean(errors)
            new_position += 1

        # If the error threshold was reached, add the new segment to the list of positions and continue
        seg_positions.append(new_position - 1)
        # Update the current error back to 0
        current_error = 0
    # Return the final list of segment positions
    return seg_positions


def error_table(df_readings, row_indices, df_path, target_folder_path):
    """
    Excel table creation of calculated error between fish midline plot and robot plot at each robot segment.

    :param df_readings: Readings from the dataset into pandas DataFrame
    :param row_indices: Rows in the dataset that represent robot datapoints
    :param df_path: df_path: The path of the dataset file
    :param target_folder_path: User specified folder path to save the table
    :return: None
    """
    # Call the error_calc function to get the grouped_diff_array
    error_array = calculate_errors(df_readings, row_indices)

    # Get information about the subject fish from fish dataset file path
    info = get_info_from_dataset(df_path)
    full_filename, title, fish_length, tail_beat_speed = info

    # Create an empty DataFrame to store the error table
    col_names = ['timeframe'] + [f'seg.{j}-{j + 1}' for j in range(len(row_indices) - 1)]
    error_table = pd.DataFrame(columns=col_names)

    # Iterate through each timeframe and add a row to the DataFrame
    for i, timeframes in enumerate(timeframe_nums(df_readings)):
        # Set the values for the error_table_row_dict dictionary
        error_table_row_dict = {'timeframe': timeframes}

        # Iterate through each segment in the row_indices list and add the corresponding error value to the dictionary
        for j in range(0, len(row_indices) - 1):
            error_table_row_dict[f'seg.{j}-{j + 1}'] = error_array[i][j]

        # Append the dictionary as a row to the DataFrame
        error_table = pd.concat([error_table, pd.DataFrame(error_table_row_dict, index=[0])])

    # Reset the index of the error_table DataFrame
    error_table = error_table.reset_index(drop=True)
    # Saving the df_table to Excel file
    modified_error_filename = str(full_filename) + '_error_val_table.xlsx'
    error_table.to_excel(os.path.join(target_folder_path, modified_error_filename), index=False)


def angle_table(df_readings, row_indices, df_path, target_folder_path):
    """
    Excel table creation of calculated angle changes of each robot segment over timeframes.

    :param df_readings: Readings from the dataset into pandas DataFrame
    :param row_indices: Rows in the dataset that represent robot datapoints
    :param df_path: The path of the dataset file
    :param target_folder_path: User specified folder path to save the table
    :return: None
    """
    # Call the error_calc function to get the grouped_diff_array
    angle_array = calculate_angles(df_readings, row_indices)

    # Get information about the subject fish from fish dataset file path
    info = get_info_from_dataset(df_path)
    full_filename, title, fish_length, tail_beat_speed = info

    # Create an empty DataFrame to store the error table
    col_names = ['timeframe'] + [f'seg.{j}' for j in range(len(row_indices) - 1)]
    angle_table = pd.DataFrame(columns=col_names)

    # Iterate through each timeframe and add a row to the fish DataFrame
    for i, timeframes in enumerate(timeframe_nums(df_readings)):

        # Set the values for the error_table_row_dict dictionary
        angle_table_row_dict = {'timeframe': timeframes}

        # Iterate through each segment in the row_indices list and add the corresponding error value to the dictionary
        for j in range(0, len(row_indices) - 1):
            angle_table_row_dict[f'seg.{j}'] = angle_array[i][j]

        # Append the dictionary as a row to the DataFrame
        angle_table = pd.concat([angle_table, pd.DataFrame(angle_table_row_dict, index=[0])])

    # Reset the index of the error_table DataFrame
    angle_table = angle_table.reset_index(drop=True)

    # Saving the df_table to Excel file
    modified_angle_filename = str(full_filename) + '_angle_table.xlsx'
    angle_table.to_excel(os.path.join(target_folder_path, modified_angle_filename), index=False)


def plot_robot_segmentation(fish_midline, proportions, df_readings, df_path, target_folder):
    """
    Plotting the fish midline and robot segments at the initial timeframe.

    :param fish_midline: Number of datapoints on the fish midline
    :param proportions: Location of robot joints on robot body in terms of body proportion
    :param df_readings: Readings from the dataset into pandas DataFrame
    :param df_path: The path of the dataset file
    :param target_folder: User specified folder path to save the plot
    :return: None
    """

    # Get information about the subject fish from fish dataset file path
    info = get_info_from_dataset(df_path)
    full_filename, title, fish_length, tail_beat_speed = info

    # Get row indices representing robot joints as datapoints
    row_indices = get_row_indices(fish_midline, proportions)

    # Get x and y coordinates of robot and fish midline
    robot_coordinates = get_robot_coordinates(df_readings, row_indices)
    x_robot, y_robot = robot_coordinates
    fish_coordinates = get_fish_coordinates(df_readings)
    x_fish, y_fish = fish_coordinates

    plt.figure(figsize=(15, 7))

    plt.xlabel("x coordinates")
    plt.ylabel("y coordinates")

    # Plot both fish and robot midlines
    plt.title(info)
    plt.scatter(x_fish[0], y_fish[0], marker=".", s=100, edgecolors="black", c="blue")
    plt.scatter(x_robot[0], y_robot[0], marker=".", s=100, edgecolors="black", c="red")
    plt.plot(x_robot[0], y_robot[0], color="red")
    plt.plot(x_fish[0], y_fish[0], color="blue")

    # Add a text label on top of each point indicating the segment datapoint index
    for i, row_index in enumerate(row_indices):
        plt.text(x_robot[0][i], y_robot[0][i] + 0.5, str(row_index), fontsize=12,
                 ha="center", va="bottom", bbox=dict(facecolor='green'))

    modified_filename = str(full_filename) + '_robot_segmentation.png'
    plt.savefig(os.path.join(target_folder, modified_filename))


def sin_params(df_readings, row_indices):
    """
    Calculating the parameters of fitted sin wave into angle change plots calculated in calculate_angles. The parameters
    are amplitude, frequency, phi (time shift) and b (vertical shift).

    :param df_readings: Readings from the dataset into pandas DataFrame
    :param row_indices: Rows in the dataset that represent robot datapoints
    :return: segment_a, segment_f, segment_phi, segment_b
    """
    # Get angle calculation data
    angle_array = calculate_angles(df_readings, row_indices)

    # Get the numbers of timeframes in the dataset
    timeframe = timeframe_nums(df_readings)

    num_segments = (len(row_indices) - 1)

    # Initialize an array of arrays with each array being the size of num_segments
    segment_angles = [[] for _ in range(num_segments)]

    segment_a = []
    segment_f = []
    segment_phi = []
    segment_b = []

    # Fill in segment angles array with angles calculated for each segment
    for i, timeframes in enumerate(timeframe):
        for j in range(num_segments):
            segment_angles[j].append(angle_array[i][j])

    # For each robot segment, fit in a sin wave into angle waves and optimise the parameters
    for i, segment_angle in enumerate(segment_angles):
        arg_max_peak = np.argmax(segment_angle)
        arg_min_peak = np.argmin(segment_angle)
        val_max_peak = np.max(segment_angle)
        val_min_peak = np.min(segment_angle)

        # Calculate amplitude
        amplitude = (val_max_peak - val_min_peak) / 2
        segment_a.append(amplitude)

        # Calculate period and frequency
        period = timeframe[-1] / 500
        frequency = 1 / period
        segment_f.append(frequency)

        # Define time space of the sine wave plot
        t_sin = np.linspace(timeframe[0], timeframe[-1], 1000, endpoint=False)
        # Define time space of the angle plot
        t_angle = np.linspace(timeframe[0], timeframe[-1], len(segment_angle), endpoint=False)

        # Generate shifted sine wave
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t_sin)

        # Find peaks
        peaks, _ = signal.find_peaks(sine_wave)

        # Calculate phase shift
        time_diff_btw_waves = t_angle[arg_max_peak] - t_sin[arg_max_peak]
        phi = time_diff_btw_waves / period
        # Ensure phi stays within [0, 2pi] range
        phi = phi % (2 * np.pi)

        # Calculate average value of head_pitch
        mean = np.mean(segment_angle)

        shifted_sine = amplitude * np.sin(2 * np.pi * frequency * t_sin + phi) + mean

        # Define the function to optimize, in this case, we want to minimize the difference
        # between the actual data and our "guessed" parameters
        optimize_func = lambda x: x[0] * np.sin(x[1] * t_angle + x[2]) + x[3] - segment_angle
        amplitude, frequency, phi, mean = leastsq(optimize_func, [amplitude, frequency, phi, mean])[0]

        # Ensure phi is still within [0, 2pi] range
        phi = phi % (2 * np.pi)


        segment_phi.append(phi)
        segment_b.append(mean)

        # recreate the fitted curve using the optimized parameters
        data_fit = amplitude * np.sin(frequency * t_sin + phi) + mean

    return segment_a, segment_f, segment_phi, segment_b


class Application(tk.Frame):
    """
    Creates a GUI for analyzing data from fish datasets for robotic fish. The class inherits from tk.Frame and has an
    __init__ method that sets up the window and creates the widgets.

    :return: None
    """
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Robotic Fish Analysis")
        width = self.master.winfo_screenwidth()
        height = self.master.winfo_screenheight()
        self.master.geometry("%dx%d" % (width, height))
        self.pack()

        # Create the widgets inside the canvas
        self.create_widgets()

    def create_widgets(self):
        """
        Creates widgets on the GUI.

        :return: None
        """
        # Top section
        self.top_frame = tk.Frame(self, bg="white")
        self.top_frame.pack(side="top", padx=10, pady=10, fill="both", expand=True)
        self.top_label = tk.Label(self.top_frame, text="Fish Data Analysis and Controller Parameters",
                                  font=("Helvetica", 20))
        self.top_label.pack(side="top", padx=10, pady=10)

        # Create line frame to divide top and bottom section
        line_frame = tk.Frame(self, height=2, bg='black')
        line_frame.pack(fill='x', padx=5, pady=5)

        # Create a user message
        self.user_message_label = tk.Label(self.top_frame,
                                           text="Note: make sure that you modified the dataset to appropriate format using "
                                                "manipulate_datasets script",
                                           font=("Helvetica", 18))
        self.user_message_label.pack(side="top", padx=5, pady=10)

        # Create widgets for the user to select fish dataset from the system file explorer
        self.filename_frame = tk.Frame(self.top_frame)
        self.filename_frame.pack(side="top", pady=10)
        self.filename_label = tk.Label(self.filename_frame, text="Select dataset file to analyze:",
                                       font=("Helvetica", 18))
        self.filename_label.pack(side="left")
        self.filename_entry = tk.Entry(self.filename_frame, font=("Helvetica", 18))
        self.filename_entry.pack(side="left")
        self.filename_button = tk.Button(self.filename_frame, text="Browse", font=("Helvetica", 18),
                                         command=self.browse_filename)
        self.filename_button.pack(side="left")

        # Create widgets for the user to select folder to save data analysis files from the system file explorer
        self.folder_frame = tk.Frame(self.top_frame)
        self.folder_frame.pack(side="top", pady=20)
        self.folder_label = tk.Label(self.folder_frame, text="Select folder to save data:",
                                     font=("Helvetica", 18))
        self.folder_label.pack(side="left")
        self.folder_entry = tk.Entry(self.folder_frame, font=("Helvetica", 18))
        self.folder_entry.pack(side="left")
        self.folder_button = tk.Button(self.folder_frame, text="Browse", font=("Helvetica", 18),
                                       command=self.browse_folder)
        self.folder_button.pack(side="left")

        # Create widgets for the user to input number of active joints on the robot
        self.current_locations_frame = tk.Frame(self.top_frame)
        self.current_locations_frame.pack(side="top", pady=10)

        self.num_segments_label = tk.Label(self.current_locations_frame, text="Number of active joints:",
                                           font=("Helvetica", 18))
        self.num_segments_label.pack(side="left")
        self.num_segments_entry = tk.Entry(self.current_locations_frame, font=("Helvetica", 18))
        self.num_segments_entry.pack(side="left")

        self.entry_widgets = []
        self.generate_locations_button = tk.Button(self.current_locations_frame,
                                                   text="Enter",
                                                   command=self.generate_locations)
        self.generate_locations_button.pack(side="left", padx=10)

        # This will disappear after user inputs number of active joints and hits enter
        self.current_locations_label = tk.Label(self.current_locations_frame,
                                                text="Widgets for entering robot joint proportions will appear here",
                                                font=("Helvetica", 18))
        self.current_locations_label.pack(side="left", padx=10)

        self.analyse_button = tk.Button(self.top_frame, text="Analyse", font=("Helvetica", 18), bg="#4CAF50",
                                        fg="black",
                                        command=self.analyse)
        self.analyse_button.pack(side="top", pady=20)

        # Label widget for saved data
        self.data_parameters = tk.Label(self.top_frame, text="", font=("Helvetica", 18))
        self.data_parameters.pack(side="left")

        # Label widget for controller parameters
        self.controller_parameters = tk.Label(self.top_frame, text="", font=("Helvetica", 18))
        self.controller_parameters.pack(side="right")

        # Bottom section
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side="bottom", pady=10)

        self.bottom_label = tk.Label(self.bottom_frame, text="Optimal Robot Segment Allocations",
                                     font=("Helvetica", 20))
        self.bottom_label.pack(side="top")

        # Widgets for optimal joint configuration based on error threshold
        self.error_frame = tk.Frame(self.bottom_frame)
        self.error_frame.pack(side="top", pady=20)
        self.error_label = tk.Label(self.error_frame, text="Enter error threshold", font=("Helvetica", 18))
        self.error_label.pack(side="left")
        self.error_entry = tk.Entry(self.error_frame, font=("Helvetica", 18))
        self.error_entry.pack(side="left")
        self.error_button = tk.Button(self.error_frame, text="Enter", font=("Helvetica", 18),
                                      command=self.error)
        self.error_button.pack(side="left")

    def browse_filename(self):
        """
        Opens a file dialog for the user to select a file from the system file explorer.

        :return: None
        """
        # Function to browse for file name
        filename = filedialog.askopenfilename(initialdir="/", title="Select a file",
                                              filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
        self.filename_entry.delete(0, tk.END)
        self.filename_entry.insert(0, filename)

    def browse_folder(self):
        """
        Opens a file dialog for the user to select a folder from the system file explorer.

        :return: None
        """
        # Function to browse for target folder
        foldername = filedialog.askdirectory(initialdir="/", title="Select a folder")
        self.folder_entry.delete(0, tk.END)
        self.folder_entry.insert(0, foldername)

    def generate_locations(self):
        """
        Generates current location entry widgets based on the user specified number of joint
        The number of joints is obtained from a tkinter Entry widget called num_segments_entry.

        :return: None
        """
        num_segments = int(self.num_segments_entry.get())

        # Destroy any existing current location entry widgets
        for widget in self.current_locations_frame.winfo_children():
            if widget != self.num_segments_label and widget != self.num_segments_entry and widget != self.generate_locations_button:
                widget.destroy()

        # Generate new current location entry widgets based on number of segments
        current_frame = None
        for i in range(num_segments):
            # Create a new frame every third iteration to ensure widgets are arranged in a grid
            if i % 3 == 0:
                current_frame = tk.Frame(self.current_locations_frame)
                current_frame.pack(side="left", padx=10)
            # Create a label and Entry widget for each segment
            label = tk.Label(current_frame, text="% Joint Ratio " + str(i + 1), font=("Helvetica", 18))
            label.pack(side="top")
            entry = tk.Entry(current_frame, font=("Helvetica", 18))
            entry.pack(side="top")
            self.entry_widgets.append(entry)  # add the Entry widget to the list

        # Get the values of all Entry widgets and store them in a list
        self.current_location_entries = [entry.get() for entry in self.current_locations_frame.winfo_children() if
                                         isinstance(entry, tk.Entry)]

    def error(self):
        """
        Callback function for the error button. Reads dataset Excel datafile, calculates the optimal position based on
        the segment growing optimization algorithm with a user-specified error threshold, and displays the result in
        the error label widget.

        :return: None
        """
        file_path = self.filename_entry.get()
        error_threshold = self.error_entry.get()
        datafile = pd.read_excel(file_path)

        optimal_position = segment_growing_opt(datafile, error_threshold)
        self.error_label.config(text=optimal_position)
        self.error_label.pack(side="right")

    def analyse(self):
        """
        This function starts the analysis of the data and calculates controller parameters.

        :return: None
        """

        # Function to start analysis
        file_path = self.filename_entry.get()
        target_folder_path = self.folder_entry.get()

        # Get the values entered in each current location Entry widget
        current_locations = []
        for entry in self.entry_widgets:
            current_locations.append(float(entry.get()))

        seg_prop_1 = current_locations[0] / 100
        seg_prop_2 = current_locations[1] / 100
        seg_prop_3 = current_locations[2] / 100

        proportions = [0, seg_prop_1, seg_prop_2, seg_prop_3, 0.99]

        # Read Excel file
        datafile = pd.read_excel(file_path)

        x, y = get_fish_coordinates(datafile)
        num_datapoints = len(x[0])

        row_indices = get_row_indices(num_datapoints, proportions)
        # Analyse the file and calculate controller parameters
        plot_robot_segmentation(num_datapoints, proportions, datafile, file_path, target_folder_path)
        error_table(datafile, row_indices, file_path, target_folder_path)
        angle_table(datafile, row_indices, file_path, target_folder_path)
        # Get the arrays from sin_parameters
        amplitude, frequency, phi, mean = sin_params(datafile, row_indices)

        self.data_parameters.config(
            text="Data saved to the target folder: \n Plot of Robot Segment Allocation \n Robot-Fish Error Table"
                 "\n Segment Angle Change Over Time Table ")

        parameter_list = []
        # Iterate through the arrays and print the values in the desired format
        for i in range(len(amplitude)):
            # Display the calculated controller parameters
            param_text = f"s{i + 1}: a = {round(amplitude[i], 2)}, f = {round(frequency[i], 2)}, " \
                         f"phi = {round(phi[i], 2)},b = {round(mean[i], 2)}\n"
            parameter_list.append(param_text)

        self.controller_parameters.config(text='Controller parameters:\n' + ' '.join(parameter_list))


root = tk.Tk()
app = Application(master=root)
app.mainloop()
