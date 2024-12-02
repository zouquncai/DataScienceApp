#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
st.markdown("# What is Data Science?")
st.sidebar.header("What is Data Science?")

st.image("AI_DS.png")


st.markdown(
    """
    Let's introduce some relevant terminologies first!
    
    **A) Artificial Intelligence (AI)**
    
    - We can define AI as computer behavior that mimics human decision making.

    - Most people may associate AI with some sort of futuristic-looking robots or a machine dominated world. 
    It might sound daunting or too distant from our everyday lives. 

    - AI can actually be as simple as a bunch of if-else statements to decide what to wear today. 
    
    - Smart speakers like Siri and Alexa are some examples of AI in our everyday lives.

    - More typically, we think of AI as machines that can make decisions based on their environment, like self driving cars. 
"""
)
st.image("auto_car.png")
st.markdown(
    """
    B) Machine Learning
    
    - Machine Learning is a subset of AI.  
    
    - It learns patterns from data (aka training data) to then make predictions or forecasts on something new.
    
    - For example, a retail store may want to forecast the sales volume for the Thanksgiving week.

    - One of the more ubiquitous examples of ML is Netflix’s recommender system.  Because I watched Orange is the New Black, 
    Netflix is suggesting I start on House of Cards.  It makes this decision based on what other viewers of Orange is 
    the New Black watched and similar characteristics between these two shows.  It has no information on what I think about House of Cards.  
    But it can make a prediction that I’ll probably enjoy it. 
"""
)

st.image("netflix.png")

st.markdown(
    """
    C) Deep Learning

    - Neural networks, aka artificial neural networks (ANNs) or simulated neural networks (SNNs), 
    are a subset of machine learning.
    
    -  Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.   
"""
)

st.image("nn.png")

st.markdown(
    """
    - Deep learning is a form of Machines Learning that is built upon neural networks. In other words, it is more sophisticated neural networks.

    - A typical example of neural network or deep learning is Google’s search engine. 
    
    D) Data Science
    
    - Data science doesn’t fit so nicely within the picture. That’s because while some data scientists do spend a lot of their time 
    training machine learning models, data scientists often have a broader skillsets that includes:
    
    - Machine Learning techniques
    - Data Visualization and reporting
    - Domain Expertise
    - Programming/coding
    - Experiment Design
    - data warehousing and structure

"""
)




# In[ ]:




