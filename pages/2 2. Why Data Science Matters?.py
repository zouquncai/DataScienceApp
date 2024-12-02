#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
st.markdown("# Why Data Science Matters?")
st.sidebar.header("Why Data Science Matters?")

st.image("why ds matters.png")


st.markdown(
    """
    Our generation is facing enormous amount of data. The blessing and curse of this world is that we have tons of data, 
    but not enough insight, and not enough people who care about extracting that insight. 
"""
)

st.markdown(
    """
    **Data can exist in different formats.**

    - Data may looks like a table in a relational database. We call this type of data *structured* since it follows a fixed format: 
    a row represents an **observation** and a column represents a **variable** or **feature**.
"""
)
st.image("Car_Fuel_Efficiency.png")

st.markdown(
    """
    - In contrast, the text data below is considered as *unstructured* since it doesn't follow a fixed format.
"""
)
st.image("text data.png")

st.markdown(
    """
    - In some cases, data scientists will work with image, video or audio data.
"""
)
st.image("image data.png")

st.markdown(
    """
    And that’s what data scientists do: 
    **they extract meaning from data. They help us understand the world and discover new things**.
"""
)

st.markdown(
    """
    ## Data Scientists extract meaning from data. They help us understand the world and discover new things.
"""
)
