# OCR with CNNs Lesson

This repository provides a comprehensive guide to implementing Optical Character Recognition (OCR) using Convolutional Neural Networks (CNNs). The content includes theoretical lessons, practical coding notebooks, and a working Flask web application.

## Structure
- **data/**: Contains training and test datasets.
- **notebooks/**: Jupyter notebooks with lessons and experiments.
- **flask_app/**: A simple Flask app for deploying the OCR model.
- **scripts/**: Python scripts for data preprocessing, training, and evaluation.
- **requirements.txt**: Dependencies required to run the project.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python scripts/train_model.py
   ```

3. Evaluate the model:
   ```bash
   python scripts/evaluate_model.py
   ```

4. Run the Flask app:
   ```bash
   cd flask_app
   python app.py
   ```

## Requirements
- Python 3.8+
- TensorFlow or PyTorch
- Flask

## Acknowledgments
Inspired by modern deep learning practices for OCR.
