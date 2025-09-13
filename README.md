This project is on image classifier using CIFAR 10 and tensorflow

The project uses the CIFAR-10 dataset from tensorflow.keras.datasets, which contains:
50,000 training images
10,000 test images
Image size: 32x32 pixels, RGB

Arichitecture:-
Conv2D + MaxPooling2D layers for feature extraction
Dropout layers for regularization
Dense layers for classification
Softmax output with 10 neurons

Training:-
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Epochs: 30–50
Batch Size: 64
Data Augmentation for better generalization

Results:-
Test accuracy: 70–78% after 30–50 epochs
Confusion matrix and classification report are included for evaluation
