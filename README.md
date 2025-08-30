:

🚦 Traffic Sign Recognition using Deep Learning

A deep learning project that classifies traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset with high accuracy.
This project was developed as part of the Machine Learning Internship Program by Elevvo Pathways.

🔍 Our experiments found that Random Forest was a strong baseline, but the Convolutional Neural Network (CNN) outperformed with ~98% accuracy on unseen test data.

Deployed using Streamlit, the model is accessible from both desktop 💻 and mobile 📱.

✨ Features

✅ Multi-class classification of 43 traffic signs

✅ Preprocessing with OpenCV (color-based cropping & resizing)

✅ Trained CNN model with TensorFlow/Keras

✅ Easy-to-use Streamlit Web App for real-time testing

✅ Supports multiple image uploads

✅ Deployable locally (VSCode, PyCharm, Jupyter) or on Colab + ngrok

📊 Dataset

We used the German Traffic Sign Recognition Benchmark (GTSRB)
 dataset containing:

🖼️ Over 50,000 labeled images

🔖 43 traffic sign classes

⚡ Images of varying sizes, lighting conditions, and angles

🧠 Model Architecture

The final CNN architecture includes:

Convolutional layers with ReLU activation

MaxPooling layers

Dropout for regularization

Fully connected dense layers

Softmax output for 43-class classification

Training was done with:

Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: 30

Accuracy: 98%

📈 Results

Classification report on test set (12630 images):

Accuracy: 98%

Macro Avg F1-Score: 0.96

Weighted Avg F1-Score: 0.98

Example:


🖥️ Streamlit App

We built an interactive Streamlit interface to allow others to test the model easily.

▶️ Run Locally
# clone the repository
git clone https://github.com/your-username/traffic-sign-recognition.git
cd traffic-sign-recognition

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py

▶️ Run in Google Colab
!pip install streamlit pyngrok
from pyngrok import ngrok
ngrok.kill()
public_url = ngrok.connect(port=8501)
print("🌍 Public URL:", public_url)
!streamlit run app.py --server.port 8501 &> /dev/null &

📂 Repository Structure
├── app.py                  # Streamlit app
├── traffic_sign_model.h5   # Trained CNN model
├── label_names.csv         # Class ID to Sign Name mapping
├── requirements.txt        # Dependencies
├── notebooks/              # Jupyter/Colab training notebooks
└── README.md               # Project documentation

👨‍💻 Author

Developed by Muhammad Ahsan
Part of the Machine Learning Internship Program at Elevvo Pathways

🏷️ Tags

#DeepLearning #ComputerVision #TrafficSigns #TensorFlow #Streamlit #ElevvoPathways
