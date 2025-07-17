# -IMAGE-CLASSIFICATION-MODEL
COMPANY: CODTECH IT SOLUTIONS
NAME: MADANIKA PITTAM
INTERN ID: CT04DG1969
DOMAIN: MACHINE LEARNING
DURATION: 4 WEEKS
MENTOR: NEELA SANTOSH
Introduction
In the era of artificial intelligence, image classification plays a vital role in numerous real-world applications, from medical diagnostics to autonomous vehicles. The objective of this project is to build an efficient image classifier using Convolutional Neural Networks (CNNs) to identify handwritten digits from the MNIST dataset. CNNs are the most effective deep learning models for image recognition tasks due to their ability to automatically detect spatial hierarchies in images. This task was undertaken as part of an internship project to deepen practical understanding of computer vision and neural network concepts using TensorFlow.

2. Objective
The main goals of the project were:

To preprocess and normalize handwritten digit images from the MNIST dataset.

To design and train a CNN model that can accurately classify digits from 0 to 9.

To evaluate the performance of the model using training and testing accuracy.

To visualize the training process and make predictions on test samples.

3. Dataset Description
The MNIST dataset is a benchmark dataset in the machine learning community. It consists of:

60,000 training images

10,000 testing images

Each image is a grayscale image of size 28x28 pixels, representing digits from 0 to 9.

The dataset is well-suited for introducing neural networks due to its simplicity, cleanliness, and wide usage.

4. Methodology
The project workflow involved the following major steps:

a. Data Loading and Preprocessing
The dataset was loaded using tensorflow.keras.datasets.mnist.

All image pixel values were normalized to the range [0,1] by dividing by 255.0 to ensure faster and more stable training.

The data was reshaped to have a single channel dimension (28, 28, 1) to match CNN input requirements.

b. Model Architecture
A CNN was built using the Sequential API from Keras, consisting of:

Conv2D Layer (32 filters) – Detects features using a 3x3 filter.

MaxPooling2D Layer (2x2) – Reduces spatial dimensions to minimize computation.

Conv2D Layer (64 filters) – Captures more complex patterns.

MaxPooling2D Layer – Further dimensionality reduction.

Flatten Layer – Converts feature maps into a 1D vector.

Dense Layer (64 units, ReLU) – Fully connected layer for learning.

Dense Layer (10 units, Softmax) – Outputs class probabilities.

c. Compilation and Training
The model was compiled using the Adam optimizer and Sparse Categorical Crossentropy as the loss function (suitable for multi-class classification).

The model was trained for 5 epochs using a batch size of 128.

Training and validation accuracy and loss were monitored during each epoch.

d. Evaluation and Visualization
The model was evaluated on the test dataset using model.evaluate.

Accuracy and loss curves were plotted using matplotlib to understand the model's learning behavior over epochs.

5. Results
After training, the model achieved a high test accuracy, typically around 98%, indicating effective learning and generalization on unseen data.

The plots clearly showed:

Training and validation accuracy converged, showing no overfitting.

Loss values steadily decreased, confirming stable learning.

An example digit from the test set was also randomly selected and classified using the trained model. The prediction matched the actual digit label, demonstrating successful inference capability.

6. Tools and Technologies Used
Programming Language: Python

Libraries: TensorFlow, NumPy, Matplotlib

Platform: Jupyter Notebook (likely executed on Google Colab or local environment)

7. Conclusion
This project successfully demonstrated the use of Convolutional Neural Networks for digit classification using the MNIST dataset. By applying modern deep learning techniques, the model was able to learn spatial hierarchies from image data and generalize effectively to test samples. The high accuracy confirms the effectiveness of the model design and training strategy.

This hands-on experience not only strengthened the understanding of CNN architecture and data preprocessing but also emphasized the importance of model evaluation and visualization. The skills gained through this project can be applied to more complex image recognition tasks in real-world scenarios.

