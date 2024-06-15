# Cloud Classification App ☁️

## Description

This application uses a deep learning model to classify images of clouds into different types. The app is built with Streamlit to provide a simple and interactive user interface.

## Project Structure



## Requirements

- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- Streamlit

You can install the dependencies using `pip`:

```sh
poetry add torch torchvision scikit-learn matplotlib seaborn streamlit

Usage

Model Training
Make sure your training data is organized into folders within data/raw_data/clouds_data/clouds_train and the test data is in data/raw_data/clouds_data/test_data/clouds_test.
Run main.py to train the model:

python main.py


Running the Web Application
Make sure cloud_classifier.pth and classes.pth are present in the root directory.
Run the Streamlit application:

streamlit run app.py

```

Code Structure

src/cloud_dataset.py
Defines the CloudDataset class to handle the loading and preprocessing of image data.

src/cloud_classifier.py
Defines the CloudClassifier class that implements the neural network architecture and methods for training, saving, and loading the model.

src/cloud_evaluator.py
Defines the CloudEvaluator class to evaluate the model's performance using metrics like the confusion matrix and classification report.

src/cloud_predictor.py
Defines the CloudPredictor class to make predictions on new images using the trained model.

main.py
Trains the model using the training data and saves the trained model and the class dictionary.

app.py
Implements the web application with Streamlit to upload images, make predictions, and display the results.

Contributions

Contributions are welcome. Please open an issue or a pull request to discuss any changes you would like to make.

License

This project is licensed under the MIT License. For more details, see the LICENSE file.


This `README.md` provides a complete description of the project, including the project structure, requirements, usage instructions, an explanation of the code, and details about contributions and licensing.
