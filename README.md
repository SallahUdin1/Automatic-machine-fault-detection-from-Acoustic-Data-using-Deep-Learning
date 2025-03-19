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

## Results
- Achieved an accuracy of **72.10%** in classifying faults.  
- The model effectively distinguishes between the four fault categories.  

### Accuracy Graph
![image](https://github.com/user-attachments/assets/c2052a62-91b8-4b58-bcc6-1f904fb1a82c)


### Confusion Matrix
![image](https://github.com/user-attachments/assets/c86f99c4-014a-469a-a178-5d484ce44ddd)

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

