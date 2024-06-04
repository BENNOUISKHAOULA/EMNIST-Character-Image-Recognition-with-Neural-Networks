#Character Image Recognition with EMNIST Dataset
This project focuses on recognizing characters from images using the EMNIST dataset. It consists of two main components: App.py for real-time character recognition and Train.py for training the recognition model.

#Features
Real-time Recognition: App.py allows users to draw characters on a whiteboard and instantly receive predictions for the drawn characters.
Interactive Interface: Users can choose actions like drawing, cutting, resetting the whiteboard, and exiting the application.
Prediction Display: The application displays the best predictions for the drawn characters along with their accuracy.
Model Training: Train.py provides functionality to load and preprocess the EMNIST dataset, train a Convolutional Neural Network (CNN) model, and save the trained model for later use.

#Dependencies
upload EMNIST-balanced-train.csv from kaggle  and add it to the project's folder 
App.py: Requires numpy, tensorflow, keras, and opencv.
Train.py: Requires numpy, pandas, matplotlib, keras, and sklearn.

#Usage
App.py
Run App.py to start the application.
Use the following keys for actions:
'D': Draw characters on the whiteboard.
'C': Cut characters (not fully implemented in this version).
'R': Reset the whiteboard.
'E': Exit the application.

Train.py
Run Train.py to train the recognition model.
The trained model will be saved as ModelPresentation.h5 for later use.

#Getting Started
Ensure that all dependencies are installed using pip install -r requirements.txt.
Adjust file paths and configurations in both App.py and Train.py as necessary for your environment.
Run App.py for real-time character recognition or Train.py to train the recognition model.

#Further Improvements
Implement the 'cut' functionality in App.py for removing incorrectly drawn characters.
Enhance the user interface with more intuitive controls and feedback.
Experiment with different CNN architectures and hyperparameters for improved recognition accuracy.
