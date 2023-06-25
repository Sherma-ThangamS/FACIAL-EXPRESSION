# Facial Expression Recognition

This project aims to recognize facial expressions from images using deep learning techniques. The model is trained to classify facial expressions into three categories: Happy, Sad, and Surprise.

## Project Structure

- `Facial_Expression.ipynb`: Jupyter Notebook containing the code for training and evaluating the facial expression recognition model.
- `model7.h5`: Pre-trained model file containing the trained weights and architecture for the facial expression recognition model.

## Dataset

The dataset used for training and evaluation consists of images labeled with facial expressions (Happy, Sad, Surprise). The dataset can be obtained from [source link or description of the dataset].

## Model Architecture

The facial expression recognition model is built using Keras, a high-level deep learning library. The architecture consists of convolutional and pooling layers followed by fully connected layers. Batch normalization and dropout are used for regularization.

## Getting Started

1. Clone the repository:

git clone  https://github.com/Sherma-ThangamS/FACIAL-EXPRESSION




2. Run the Jupyter Notebook `Facial_Expression.ipynb` to train the model and evaluate its performance.

3. Once the model is trained, you can use it for predicting facial expressions on new images.

## Usage

To use the trained model for predicting facial expressions, you can load the model using Keras and apply it to new images. Here's an example:

```python
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Load the trained model
model = keras.models.load_model('model7.h5')

# Load and preprocess the image
img = image.load_img('path/to/your/image.jpg', target_size=(84, 76))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize the pixel values

# Perform prediction
predictions = model.predict(np.array([img_array]))
predicted_class_index = np.argmax(predictions[0])

# Map the predicted class index to the corresponding label
class_labels = ['Happy', 'Sad', 'Surprise']
predicted_class_label = class_labels[predicted_class_index]

print("Predicted facial expression: ", predicted_class_label)
```
## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

