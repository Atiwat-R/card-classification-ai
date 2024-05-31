# Card Classification AI

Artificial Intelligence/Machine Learning project. An AI model that can classify playing cards from a standard 52-card (+Joker) deck, including those with unconventional stylistic designs. Achieved roughly 97% accuracy.

Dataset: <u> https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification </u>

Based on & Extended from: <u> https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier </u> 

### Tools Used
- Coding
    - Python 3
    - Numpy
    - Python Notebook
    - Google Colab
- Artificial Intelligence & Machine Learning
    - PyTorch (AI training, testing, etc.)
    - timm (Pre-trained models)
- Visualization
    - Matplotlib
    - Seaborn


### Run in Colab
- Set up Dataset from the link above.
- Adjust paths in the file to the dataset
- In Colab, switch the Runtime type from CPU to using GPU
    - Runtime -> Change runtime type
- If needed, !pip install necessary dependencies (such as timm)


# Design Documentations

### 0: Setup

I am using PyTorch as the main library for training the AI model. I am also using timm for pre-trained models to get started. This is because using pre-trained models made for image classification helps with limited size of the dataset and other issues. And it speeds up the process rather than creating everything from scratch.

### 1: Prepare Dataset

In this phase I obtained the data from the dataset directories and oragnize them with into a class. Also list out all the possible classes for this classification task: 53 in total (standard 52 cards + Joker). Each of the index from 0-52 corresponds to a class, which is a card type.

Data cleaning would also be done in this step, but for this particular dataset no much cleaning is needed. Also defined a transform_config to transform each image into a consistent tensor shape.

Also made a data loader. This is primarily for batching; it is better to feed data into the model batch by batch for training, to maximize GPU and memory efficiency, as well as regularization to help with overfitting. I chose 32 as the batch size; 32 images in one batch.

### 2: Creating the Model

Use timm's pre-trained efficientnet_b0 for training. But modify it by shaving off the very last layer and replace it with a custom Classifier so the output would be in the shape that we want; one that list the probabilities of how likely the card would belong in one of the 53 classes/categories. Softmax is used as the final activation function.

### 3: Training the Model

For the loss function, I use Cross Entropy Loss as it is well-suited for models that outputs probabilities over multiple classes. Then, I train the model over 7 epochs. The reason I ended up using 7 epochs is that it give the best results for this task after many experiments. Also visualized the loss decreasing after each epoch.

### 4: Testing the Model

On the testing phase, I use Confusion Matrix and Classification Report (Precision, Recall, and F1-Score), as it gives a great overview of the model's performance. Also counted raw failures manually in addition.

### 5: Visualize Predictions

Lastly, I visualize the testing phase by printing out the card's image and the model's outputted probabilities over multiple classes, made into a barplot to illustrate the prediction for each model. From a glance, it seems the cards with unusual designs o fantastical images tended to confuse the AI model the most, leading to wrong predictions or higher predictions in multiple other classes. 

Based on my experience, the confusion was worse at 5 epoch, but improved greatly at 7 epoch. Training at higher epochs than 7 yielded limited improvements. In the end, the model settled at 97% accuracy, correctly predicting 256/265 test images.

