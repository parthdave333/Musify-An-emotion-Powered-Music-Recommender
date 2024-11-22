# Musify-An-emotion-Powered-Music-Recommender

## Project Overview
This project uses facial expressions and hand gestures to classify emotions with a neural network model trained on MediaPipe landmarks. The classified emotions are used to recommend songs through a Streamlit-based web application, enhancing the user experience by aligning music with the user's mood.An innovative system that detects emotions using facial and hand gestures via MediaPipe and a trained neural network model. It recommends music tailored to the user's mood, combining real-time emotion recognition with a Streamlit-powered web app for an engaging and personalized experience.

## Key Features
- **Emotion Detection**: Detects emotions using facial and hand landmarks with MediaPipe.
- **Music Recommendation**: Suggests songs based on detected emotions and user preferences (language, singer).
- **Interactive Web App**: Built with Streamlit for real-time emotion detection and music recommendations.
- **Custom Training**: Allows users to collect data and train the model with their own gestures.

## Files
### Python Scripts:
- `data_collection.py`: Captures facial and hand landmarks and saves data as `.npy` files.
- `data_training.py`: Trains a neural network model using the `.npy` files and saves the model (`model.h5`) and labels (`labels.npy`).
- `inference.py`: Performs real-time emotion detection using a webcam and the trained model.
- `music.py`: A Streamlit app that recommends music based on detected emotions and user preferences.

### Data Files:
- `.npy` files: Pre-collected data representing various emotions (e.g., `happy.npy`, `sad.npy`).
- `model.h5`: Trained neural network model.
- `labels.npy`: Emotion labels for the model.

## Technologies Used
- **MediaPipe**: For landmark detection.
- **TensorFlow/Keras**: For building and training the neural network.
- **OpenCV**: For real-time video processing.
- **Streamlit**: For the web application.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/parthdave333/emotion-based-music-recommendation.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the data collection script (optional):
   ```bash
   python data_collection.py

4. Follow the instructions to collect new data.
  A. Train the model (optional):
     ```bash
     python data_training.py
  B. Ensure your .npy files are in the same directory.
  C. Run the Streamlit app:
  
     streamlit run music.py

## Dataset
The dataset consists of landmarks for facial and hand gestures captured using MediaPipe and saved as .npy files. Users can collect and add new gestures to enhance the dataset.

## Results
The trained model classifies emotions in real time, recommending songs based on the detected emotion and user preferences for language and singer.

## Future Improvements
Adding support for more gestures and emotions.
Incorporating voice input for more natural interaction.
Optimizing the model for faster inference on low-end devices.

## Contributing
Contributions are welcome! Fork the repository, make changes, and submit a pull request.
