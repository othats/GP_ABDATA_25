import streamlit as st

st.set_page_config(
    page_title="Aigües de Barcelona - Water Consumption Modeling",
    layout="wide"
)

st.title("Aigües de Barcelona - Water Consumption Analytics")
st.markdown("""
Welcome!  
Use the menu on the left to navigate between:

1. **Data Exploration**  
2. **Consumption Prediction (with Weather)**  
3. **Consumption Prediction (without Weather)**  

This application uses 17M water consumption records merged with Barcelona weather data.
""")
