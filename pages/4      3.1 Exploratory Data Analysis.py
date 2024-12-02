st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊")

st.markdown("# 🔍 Exploratory Data Analysis")
st.sidebar.header("🔍Exploratory Data Analysis")

st.write(
    """
    Now let's start with **Exploratory Data Analysis**. It is the process for analyzing data sets to identify patterns, relationships, and anomalies. A data scientist uses many tools to help understand the data, including:
    - check the distributions of each variable in the data
    - create charts (such as bar chart, line chart) to help visualize the data
    - analyze the correlations between variables in the data
    - build models to mathematically evaluate the patterns in the data
    """
)


@st.cache_data
def get_UN_data():
    # AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    # df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
    df = pd.read_csv("cars2020.csv", encoding = 'ISO-8859-1')
    return df
    # return df.set_index("Drive")

# @st.experimental_memo
def get_boxplot(source, x_var, y_var, use_container_width: bool):
    import altair as alt

    chart = alt.Chart(source).mark_boxplot(extent='min-max').encode(
        x=x_var,
        y=y_var
    )

    # tab1, tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

    # with tab1:
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
    # with tab2:
    #     st.altair_chart(chart, theme=None, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Data Details", "Chart 1", "Chart 2"])
try:
    df = get_UN_data()
    # drive = st.multiselect(
    #     "Choose Type of Drive", list(df.index), ["2-Wheel Drive, Front"]
    # )
    with tab1:
        n_row = st.number_input(label = "Enter the number of rows you would like to view (max 50):",
                            min_value = 1, 
                            max_value = 50, 
                            value = 5,
                           placeholder = "enter a number between 1 and 20")

        st.write(df.head(n = n_row))
        st.markdown("""
        By just reading the table above, what can you tell about the fuel efficiecy of each vehicle model? Probably not much. Data itself has a lot of information; but it needs some exploration. This is exactly what a data scientist does as part of his/her job: *data visualization* and *exploratory data analysis*. Click on the next tab "Chart 1" and "Chart 2" to learn more.
        """)
    with tab2:

        st.markdown("""
        Visualization makes it so much easier to glean insights from the data. the histogram on this page is the distribution of MPG across different car models. It is easy to tell that on average, a car can run ~25 miles with one gallon of gas. Some cars can achieve a high efficiency of close to 60 MPG; while other cars can be as low as 10 mpg.
        """)

        # histogram for "MPG"
        fig, ax = plt.subplots()
        ax.hist(df["MPG"], bins=20)
        ax.x_label = "Miles Per Gallon"
        ax.y_label = "frequency"
        st.pyplot(fig)

        # scatter plot for "MPG" by "Cylinders"
        fig, ax = plt.subplots()
        ax.plot(df["Cylinders"], df["MPG"], 'bo')
        st.pyplot(fig)
        
        
    with tab3:
        st.markdown("""
        The boxplots on this page provide additional insights. It is clear that fuel efficiency varies depending on the type of car/drive, number of cylinders, etc. For example, "2-wheel drive, Front" in general has higher MPG than "2-wheel drive, Rear"; and average fuel efficiency drops as number of cylinders increases.
        """)
        get_boxplot(df, "Model", "MPG", True)
        get_boxplot(df, "Drive", "MPG", True) 
        get_boxplot(df, "Cylinders", "MPG", True)
        get_boxplot(df, "Fuel injection", "MPG", True)
        get_boxplot(df, "Exhaust Valves Per Cyl", "MPG", True)
        get_boxplot(df, "Intake Valves Per Cyl2", "MPG", True)
        

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )

st.markdown("""
🎓 Resources

[Use Excel to create a chart from start to finish](https://support.microsoft.com/en-us/office/create-a-chart-from-start-to-finish-0baf399e-dd61-4e18-8a73-b3fd5d5680c2)

[Use Google Sheet to add of edit a chart](https://support.google.com/docs/answer/63824?hl=en&co=GENIE.Platform%3DDesktop)

[Use Python to produce charts](https://plotly.com/python/)

""")
# In[ ]:




