🌸 Flower Recognition using CNN
This project uses Convolutional Neural Networks (CNN) with TensorFlow and Keras to classify images of flowers into five categories: Daisy, Dandelion, Rose, Sunflower, and Tulip. The model is trained on a labeled dataset of flower images using data augmentation and visualized using various Python libraries.

📂 Dataset
The dataset contains five classes of flowers located in respective directories:

daisy

dandelion

rose

sunflower

tulip

📁 Dataset Path: ../input/flowers/flowers/

🛠️ Technologies & Libraries Used
Python 3.x

TensorFlow / Keras

NumPy, Pandas

OpenCV

Matplotlib, Seaborn

Scikit-learn

📊 Project Workflow
1. Data Loading & Preprocessing
Images are loaded and resized to 150x150.

Labels are assigned based on folder names.

Dataset is normalized and split into training and testing sets.

Labels are one-hot encoded.

2. Data Visualization
Random flower images are visualized to inspect the dataset.

3. Model Building
A Sequential CNN model is used with:

4 Convolutional layers

MaxPooling layers

Dropout

Dense layers with softmax activation

4. Compilation & Training
Loss Function: categorical_crossentropy

Optimizer: Adam

Metrics: accuracy

Callback: ReduceLROnPlateau for learning rate tuning

Data Augmentation is used to avoid overfitting

5. Evaluation
Plotting training vs validation loss

Predicting flower types on validation data

Visualization of correctly and incorrectly classified images

📈 Results
Model performance is visualized using loss curves.

Correctly classified and misclassified images are displayed with predicted vs actual labels.

📌 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/astiktyagi27/Flower-Recognition.git
cd Flower-Recognition
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the training script:

bash
Copy
Edit
python flower_recognition.py
🔮 Future Improvements
Hyperparameter tuning using more advanced search methods

Adding transfer learning with pretrained models like VGG16, ResNet

Deployment as a web app

🙌 Acknowledgments
Inspired by standard flower classification datasets.

Special thanks to open-source contributors and the Keras/TensorFlow community.
