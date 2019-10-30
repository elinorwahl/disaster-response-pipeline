# Disaster Response Pipeline Project

### Introduction

This project combines data engineering, natural language processing, and machine learning to create a pipeline that analyzes and classifies data from a Figure Eight database of messages related to disasters, then uses a web app to visualize the data from the classified messages. This web app can also be used with the trained machine learning model to classify new messages input by the user.

The project consists of three parts:

1. An ETL pipeline contained in `data/process_data.py` that loads the message and category datasets, merges them, cleans the resulting data, and stores it in a SQLite database.

2. A machine learning pipeline contained in `model/train_classifier.py` that loads the SQLite data; creates a language processing model; trains the model and fine-tunes the parameters using GridSearchCV; displays the model's performance on a test set; and saves the model as a Pickle file.

3. A Flask web app contained in the `app` folder, which provides visualizations of the relative numbers of the three message genres, a heatmap of correlations between types of disaster message categories, and the top ten most frequent words that occur across the message database.

### Installation

Clone this GIT repository:
`git clone https://github.com/elinorwahl/disaster-response-pipeline.git`

### Usage
1. Run the following commands in the project's root directory to set up the database and model.

    - To run the ETL pipeline that cleans data and stores it in a database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains a classifier and saves it:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app:
    `python run.py` Go to http://0.0.0.0:3001/ or http://localhost:3001
