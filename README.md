# Robotic Fish Analysis and Optimisation Framework

The framework is a tool for comparing different multi-segment robot designs based on their segment joint locations and their capability to mimic a specific fish species. The framework analyses steady swimming fish data and output robot controller parameters resembling fish movement to aid in the study of different swimming patterns, analyse the performance of the robot when mimicking a specific type of fish and suggest optimal joint configuration for the robot based on data analysis results.

The submission also includes:
1. A folder (fish_analysis_methods) with Python files for each data analysis methods
2. A folder (manipulate_datasets) with Python script to modify fish dataset format
3. A folder (Modified Fish Datasets) with modified sturgeon datasets
4. A folder (Artificial Datasets) with artificial fish datasets used in testing
5. A folder (Robotic Fish Analysis Results) including data analysis results files from all fish datasets.


## Authors

- [dak29@aber.ac.uk]

    
## Libraries Used
This project was developed using the following Python libraries:

tkinter (version 0.0.9): for GUI development
numpy (version 1.24.2): for numerical calculations
pandas (version 2.0.0): for data manipulation and analysis
matplotlib (version 3.7.1): for data visualization
os (version 2.1.4): for interacting with the operating system
scipy.optimize (version 1.10.1): for optimization functions
scipy.signal (version 1.10.1): for signal processing functions
openpyxl (version 3.1.2): for working with Excel files

Make sure you have installed the required versions of these libraries before running the project. 

You can install them using pip by running the following commands:
```bash
  pip install tkinterx
  pip install numpy
  pip install pandas
  python -m pip install -U pip
  python -m pip install -U matplotlib
  pip install scipy
  pip install openpyxl


```

## Usage

To use the dataset modification script, follow these steps:

1. Run the code from your Python IDE
2. Input the path to folder which has unmodified fish dataset files on Run tool window.
3. Input the path to folder which you want to save modified fish dataset files on Run tool window.


To use the framework, follow these steps:

1. Install the necessary libraries by running the following 
2. Run the code from your Python IDE
3. Follow the prompts to input the required data and generate the desired output.

When finished, exit the program.
