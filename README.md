# Automatic Machine Fault Detection Using Deep Learning

## Introduction
This project implements a machine fault detection system using **Deep Learning** techniques on **Acoustic Data**. The system identifies four fault types: **Arcing**, **Corona**, **Looseness**, and **Tracking**. By leveraging Mel Spectrogram images generated from audio data, the system effectively classifies faults using a **Convolutional Neural Network (CNN)** model.

## Features
- Converts audio data into **black & white spectrogram images** for model training.  
- Splits data into **80% training** and **20% testing**.  
- Utilizes a CNN model for efficient fault classification.  
- Provides visual performance insights using a **Confusion Matrix**.  

## Workflow
1. **Data Preparation**  
   - Load `.wav` files containing machine sound data.  
   - Segment each audio file into 0.5-second clips with a 0.2-second overlap.  
   - Generate Mel Spectrogram images for each segment.  
2. **Data Splitting**  
   - Automatically split data into **80% training** and **20% testing** for each class.  
3. **Model Training**  
   - Train a CNN model with three convolutional layers and fully connected layers.  
4. **Evaluation**  
   - Evaluate performance using **Test Accuracy** and a **Confusion Matrix** for class-specific results.

## Requirements
- **Python 3.x**  
- **TensorFlow/Keras**  
- **Librosa** (for audio data processing)  
- **Matplotlib** (for visualizing Mel Spectrograms)  
- **Scikit-learn** (for confusion matrix generation)  

## How to Run
1. **Upload Dataset:** Place `.wav` files in Google Drive.  
2. **Run the Code:** The provided code will:
   - Generate spectrogram images.  
   - Split the data for training and testing.  
   - Train and evaluate the model.  
3. **Results:** View the **accuracy** and **confusion matrix** output.

## Results
- Achieved high accuracy in classifying faults.  
- The model effectively distinguishes between the four fault categories.

## Future Enhancements
- Improve accuracy with hyperparameter tuning.  
- Deploy the system on a **Raspberry Pi** for real-time machine fault detection.  

## Group Members
- **Umer Abid**  
- **Sallah Udin**  
- **Rafaqat Ali**  

## Supervisor
- **Engr Ahmad Khawaja**

## License
This project is open-source and available for use under the [MIT License](LICENSE).

