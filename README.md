Pratilipi Recommendation System

This repository contains a recommendation system designed for Pratilipi, leveraging user interaction data and content metadata to provide personalized recommendations. The system utilizes advanced deep learning techniques, including Neural Collaborative Filtering (NCF), to model user preferences.
Features

    Neural Collaborative Filtering (NCF): A deep learning approach that learns user-item interactions through embedding layers and neural networks.
    Custom PyTorch Dataset and DataLoader for efficient batch processing.
    Dynamic User and Item Embeddings for improved recommendation accuracy.
    Data Preprocessing: Handles time-based sorting, mapping of users and items to indices, and normalization of ratings.

Algorithms Used
Neural Collaborative Filtering (NCF)

NCF is a neural network-based approach for collaborative filtering that replaces traditional matrix factorization methods with multi-layer perceptrons (MLPs). In this notebook:

    Embedding Layers are used to learn dense representations of users and items.
    Multi-Layer Perceptron (MLP) network is constructed with the following architecture:
        Input: Concatenated user and item embeddings.
        Hidden Layers:
            Layer 1: 128 neurons with ReLU activation and Dropout.
            Layer 2: 64 neurons with ReLU activation and Dropout.
            Layer 3: 32 neurons with ReLU activation.
        Output: Single neuron with Sigmoid activation to predict the likelihood of user interaction.

Dataset and Preprocessing

    User Interaction Data:
        Captures user behavior, including reading percentages of content.
        Sorted by interaction timestamps to maintain temporal sequence.

    Metadata:
        Contains additional content information, such as publication date and category.

    Data Mapping and Encoding:
        Users and items are mapped to numerical indices for embedding layers.
        Ratings (reading percentages) are normalized to a range of 0 to 1.

    Custom PyTorch Dataset:
        ReadingDataset class is implemented for efficient data handling and batching.

Model Architecture

User Input → Embedding Layer → | 
                               | → Concatenated → MLP → Output (Interaction Probability)
Item Input → Embedding Layer → |

    Embedding Dimension: 100
    Activation Function: ReLU
    Output Activation: Sigmoid
    Dropout Rate: 0.2 for regularization

Requirements

To run the notebook, ensure the following packages are installed:

pandas
numpy
torch

You can install them using:

pip install pandas numpy torch

Usage

    Clone the repository:

git clone <repository-url>
cd pratilipi-recommendation

Ensure the datasets are available in the data/ directory:

    data/user_interaction.csv
    data/metadata.csv

Run the Jupyter notebook:

    jupyter notebook pratilipi_recommendation.ipynb

Training and Evaluation

    Loss Function: Binary Cross-Entropy Loss for implicit feedback.
    Optimizer: Adam optimizer with a learning rate of 0.001.
    Metrics: Evaluation metrics like accuracy and AUC-ROC are used to assess model performance.

Results and Evaluation

The notebook includes evaluation metrics to assess the model's performance and suggest improvements for future iterations. It also provides visualizations to interpret the model's predictions and accuracy.
Future Enhancements

    Incorporate additional metadata features for better content representation.
    Experiment with different neural architectures like attention mechanisms.
    Hyperparameter tuning using techniques like Grid Search or Bayesian Optimization.

Contributing

Feel free to open issues or submit pull requests for enhancements and bug fixes.
License

This project is licensed under the MIT License. See the LICENSE file for more details.