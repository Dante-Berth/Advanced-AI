#AutoDL - Automated Deep Learning
AutoDL is a repository that automates the process of deep learning model selection, architecture search, and hyperparameter tuning using a Bayesian optimizer. It aims to find the best deep learning model for a given task by efficiently exploring the search space of different architectures and hyperparameters.

## Features
### Bayesian Optimization: 
AutoDL utilizes a Bayesian optimizer, such as Optuna or Hyperopt, to efficiently search the hyperparameter space and architecture space of deep learning models.

### Model Search Space: 
The repository provides a predefined search space for various deep learning models, including architectures like Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformer models.

### Hyperparameter Tuning: 
AutoDL performs hyperparameter tuning for each architecture using the Bayesian optimizer. It automatically searches for the optimal hyperparameters, such as learning rate, batch size, activation functions, optimizer, etc.

### Early Stopping: 
The models are trained with early stopping to prevent overfitting and to save time during the optimization process.
