# Animal Classification using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of cats and dogs. The dataset consists of 10,000 images, split into training and test sets. Image preprocessing and augmentation are performed using `ImageDataGenerator` to improve model generalization.

## Dataset

The dataset can be gotten from (here)[https://www.dropbox.com/scl/fi/ppd8g3d6yoy5gbn960fso/dataset.zip?rlkey=lqbqx7z6i9hp61l6g731wgp4v&e=2&st=kz9n3c8w&dl=0]

- **Training Set:** Contains images of cats and dogs used to train the model.
- **Test Set:** Contains images used to evaluate model performance.
- Images are resized to **64x64 pixels** before training.
- Augmentation techniques (shear, zoom, horizontal flip) enhance generalization.

## Technologies Used

- **Python**
- **TensorFlow/Keras**
- **ImageDataGenerator** for image preprocessing and augmentation
- **Sequential CNN Model** for classification

## Project Structure

```
/animal_classification
│── dataset/
│   ├── training_set/
│   │   ├── cats/
│   │   ├── dogs/
│   ├── test_set/
│   │   ├── cats/
│   │   ├── dogs/
│── animal_classification.ipynb
│── README.md
```

## Model Architecture

The model consists of:

1. **Convolutional Layers:** Extract spatial features from images.
2. **Pooling Layers:** Reduce spatial dimensions to prevent overfitting.
3. **Flattening Layer:** Converts 2D feature maps into a 1D vector.
4. **Fully Connected Layers:** Classifies images into two categories (cat or dog).

## Installation & Usage

### Prerequisites

Ensure you have Python installed along with TensorFlow and necessary libraries:

```bash
pip install tensorflow numpy matplotlib
```

### Running the Notebook

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook animal_classification.ipynb
   ```
2. Run each cell sequentially to preprocess data, train the model, and evaluate performance.

## Results & Evaluation

- The model is trained using **binary cross-entropy loss** and **Adam optimizer**.
- Performance is evaluated using accuracy metrics on the test set.
- After training the model with 25 epochs, an accuracy of 81% was gotten



## Future Improvements

- Experiment with deeper architectures for improved accuracy.
- Utilize transfer learning with pre-trained models (e.g., VGG16, ResNet).
- Optimize hyperparameters for better performance.

## License

This project is open-source under the MIT License.

