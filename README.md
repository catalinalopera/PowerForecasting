# Solar Power Forecasting

This repository contains data gathered from two solar power plants in India over a 34-day period. The data consists of two pairs of files, with each pair containing one power generation dataset and one sensor readings dataset. The power generation datasets are collected at the inverter level, where each inverter is connected to multiple lines of solar panels. The sensor data is collected at the plant level, with a single array of sensors optimally placed at the plant.

## Overview

This project aims to analyze the data collected from the solar power plants and address the following concern:

- Can we predict the power generation for the next couple of days?

## File Structure

- `data/`: This Contains the power generation datasets.
- `Model/`: This Contains Keras Model.
- `Templates/`: This Contains all htmls files.
- `app.py/`: This Contains all Flask functions.
- `full-stack-project.ipynb/`: Notebook with model code. 
- `Solar_Regression_Script_Solution.py/`: This contains all the scripts needed for creating Flask file.
- `Solar_Regression_Script.py/`: Contains all scripts for running the model and generate file keras/h5.
- `Requirements/`:  This lists the packages and their specific versions that are necessary to run a the project.  

## Installation and Setup

To run the analysis and predictions locally, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/catalinalopera/PowerForecasting.git
    ```

2. Navigate to the project directory:

    ```bash
    cd solarproject
    ```

3. Install the required dependencies:

   ```bash
    Python Version: 3.8.19
    ```

    ```bash
    pip install -r requirements.txt
    ```

## Usage

- Explore the data in the provided Jupyter notebooks to understand the structure and contents.
- Run the analysis notebooks to generate insights and predictions regarding power generation.
- Experiment with different machine learning models and techniques to improve prediction accuracy.

## Contributing

Contributions to this project are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.

- `Juan Henao Barrios`
- `Zarina Dossayeva `
- `Diana Catalina Lopera Bedoya`

