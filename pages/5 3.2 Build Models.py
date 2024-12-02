import streamlit as st
import scikit-learn as sklearn

st.markdown("""
Exploratory data analysis helps data scientists understand the data and identify potential quality issues in the data. While data scientists often glean important insights via data visualization, they often rely **machine learning (ML) models** to learn more comprehensive patterns in the data (which is often known as machine learning model); and then use the patterns learned to make predictions or forecasts. 

""")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Intro to ML Models", "Types of ML Models", "Regression", "Classification", "Clustering", "Anomaly Detection"])

with tab1:
    st.markdown("""
    In this section, we will explore the various types of machine learning algorithms that data scientists work with. Machine learning is generally a training system to learn from past experiences and improve performance over time. 
    
    So *what is **a machine learning model***? In general, a machine learning model is a function that helps maps the inputs to the output. You can think of the function as a pathway from the various inputs to the output; this pathway can be very simple; more often is it a winding path that has twists and turns. 
    
    *How do we build these machine learning models*? People learn by recognizing patterns. We can teach the computer to learn the same way. By showing it a lot of examples (aka data), the computer can learn the patterns in the data too.
    """)

    st.image("machine learning.png")

    st.markdown("""
    Using the example above: 
    
    If we give our model too little data, it might infer the wrong patterns. The top row shows a red car, and the machine learning model may think all cars are red. So if you show the model a picture of a red apple, it may predict the red apple is a car. But in reality we know it is an apple!

    To help the machine learning model learn the appropriate patterns, we can show it a variety of pictures of different cars, and more cars so that it can recognize the key features of our cars. In the second row of the cars picture, here we have a green car, a red car, and a blue car. Not all cars are red, but it seems that all cars have wheels; specifically, in this picture they might have four wheels. If we now show this machine learning model a picture of a red apple, it can accurately predict that the apple is not a car because it's missing a key feature of wheels. It says, okay, this apple doesn't have any wheels, so it must be something else. Maybe I don't know it is an apple, but at least I know it is not a car. That can be a smarter prediction and more helpful for us.

    Data or examples can be represented in different ways, just like what are introduced in the section: "*Why Data Science Matters*".
    """)

    st.markdown("""

📚 Glossary


**Data**: Information used to train machine learning models. In the context of the cars example, data refers to the examples shown to a computer to help it learn patterns.

**Patterns**: Regularities or trends observed i nthe data. In machine learning, computers learn by recognizing patterns in the provided data.

**Function**: in the context of AI and machine learning, a function represents the relationship between inputs and outputs of a model. It is the pathway through which the system processes data.

**Model**: A representation of a system or process. In the context of machine learning, a model is created to map inputs to outputs based on learned patterns from data.

**Features**: Distinct characteristics or attributes of data that are used by machine learning models to make predicitons or classifications.

**Target**: The output that a machine learning model is trying to predict.

**Layers**: Components of a machine learning model that process information at different levels of abstraction. Models often have multiple layers, with more layers leading to more complex pathways for data.

**Training**: The process of teaching a machine learning model by exposing it to example (data) so that it can learn and generalize patterns.

**Inference**: The process of drawing conclusions based on evidence or reasoning. In machine learning, an inference is made when the model predicts or classifies new data.

**Regression Model**: a machine learning model with a quantitative output, such as mileage per gallon of a car.

**Classification Model**: a machine learning mdoel with qualitative output, such as car or not a car.
****

    """)

    

    st.markdown("""

🎓 Resources

[Types of Machine Learning Models](https://www.geeksforgeeks.org/types-of-machine-learning/)

[Supervised Machine Learning](https://www.geeksforgeeks.org/supervised-machine-learning/)

[Unsupervised Machine Learning](https://www.geeksforgeeks.org/unsupervised-machine-learning-the-future-of-cybersecurity/)
    """)
    
with tab2:
    st.image("types of ML.png")
    
    st.markdown("""    
    Machine learning algorithms can be roughly grouped into two categories: **Supervised Machine Learning** and **Unsupervised Machine Learning**. The picture above shows an example of supervised machine learning. It requires both inputs and output and the model learns the pattern in the data (i.e., the relationship between inputs and output). On the other hand, unsupervised machine learning doesn't need an output; it instead discovers hidden patterns, similarities, or clusters within the data.

    Supervised Machine Learning can be further divided into **regression** and **classification** where regression models predict a quantity such as MPG of a car; and classification models predicts a label such as whether an image is a car or not.

    Unsupervised Machine Learning can be further divided into **clustering** and **anomaly detection**, where clustering focuses on grouping data points into clusters based on their similarity, and anomaly detection focuses on identifying outliers in the data that are different from the majority.
    """)
    
    st.markdown("""

📚 Glossary

**Regression Model**: a machine learning model with a quantitative output, such as mileage per gallon of a car.

**Classification Model**: A machine learning model with qualitative output, such as car or not a car.

**Clustering**: In machine learning, "clustering" is an unsupervised learning technique that involves grouping similar data points together into clusters based on their characteristics, allowing you to identify patterns and relationships within a dataset without any pre-defined labels, essentially organizing data into groups where points within a cluster are more similar to each other than to points in other clusters. 

**Anomaly Detection**: Anomaly detection is the process of identifying data points that are different from the norm or established pattern. It's also known as outlier detection. 


🎓 Resources

[Clustering](https://www.linkedin.com/pulse/what-clustering-machine-learning-avishek-patra-ap)

[Anomaly Detection](https://cnvrg.io/anomaly-detection-python/)

    """)

with tab3:
    
    st.markdown("""
        In this section, we will build a simple linear regression model. The target is *MPG*; the feature used is *Cylinders*. The goal is to fit a line that best explain the relationship between number of Cylinders and MPG.
    """)
    code = '''
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # read the data
    df = pd.read_csv("cars2020.csv", encoding = 'ISO-8859-1')

    # specify the target variable y and the independent variables X
    y = df["MPG"]
    X = df["Cylinders"].values.reshape(-1, 1)

    # train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # make predictions
    y_pred = model.predict(X)

    # plot the actual target "y" and predicted target values "y_pred"
    plt.plot(df["Cylinders"], y, 'bo')
    plt.plot(df["Cylinders"], y_pred)
    '''
    st.code(code, language = "python")


    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # read the data
    df = pd.read_csv("cars2020.csv", encoding = 'ISO-8859-1')

    # specify the target variable y and the independent variables X
    y = df["MPG"]
    X = df["Cylinders"].values.reshape(-1, 1)
    
    # train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # make predictions
    y_pred = model.predict(X)

    # plot the actual target "y" and predicted target values "y_pred"
    fig, ax = plt.subplots()
    ax.plot(df["Cylinders"], y, 'go')
    ax.plot(df["Cylinders"], y_pred)
    
    st.pyplot(fig)

    st.markdown("""
    In the plot above, the green dots represents the example data points in the training data; the blue line is one of the possible linear models that is trained from the example data points.
    """)
    
    st.markdown("""

    📚 Glossary

    - Target: the outcome that a model is trying to predict. 
    - Feature: aka independent variable or input variable.
    - Simple linear regression: a linear model that uses only one feature.
    - Multiple linear regression: a linear model that uses more than one feature.

    🎓 Resources

    [linear regression in Python](https://realpython.com/linear-regression-in-python/#what-is-regression)
    
    [Simple Linear Regression](https://www.sfu.ca/~mjbrydon/tutorials/BAinPy/09_regression.html)
    
    
    """)
                                  
with tab4:
    st.image("classification.png")
    code = '''
    # importing libraries
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models
    import matplotlib.pyplot as plt

    # loading data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(8,8))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays,
        #which is why we need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()
    '''
    st.markdown("""
   
    🎓 Resources

    [Image Classification using CNN for Beginners](https://www.kaggle.com/code/anandhuh/image-classification-using-cnn-for-beginners/notebook)
    
    []()
     """)
with tab5:
    st.image("clustering.png")
with tab6:
    st.image("anomaly.png")

