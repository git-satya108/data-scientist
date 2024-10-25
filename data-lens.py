import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from dotenv import load_dotenv

# Load secrets file from Streamlit
llm = OpenAI(api_token=st.secrets["OPENAI_API_KEY"])

# Module 1: Load and combine CSV/Excel files
@st.cache_data
def load_combined_data(files):
    combined_df = pd.DataFrame()
    for file in files:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

# Module 2: Handle natural language queries using OpenAI models via SmartDataframe
def handle_nlp_query(smart_df, query):
    try:
        result = smart_df.chat(query)  # Use the SmartDataframe's chat method for natural language queries
        return result
    except Exception as e:
        st.error(f"Error in processing query: {e}")
        return None

# Main Streamlit App Interface
def main():
    st.title("Gen AI-Powered Data Analysis with NLP")

    # Step 1: Upload Files
    uploaded_files = st.file_uploader("Upload your Excel or CSV files", type=["csv", "xlsx"], accept_multiple_files=True)

    # Step 2: Combine Files into a Single DataFrame
    if uploaded_files:
        combined_data = load_combined_data(uploaded_files)
        st.write("Combined Data Preview:", combined_data.head())

        # Step 3: Initialize OpenAI model with SmartDataframe
        llm = OpenAI()  # The API key is read from the environment variable
        smart_df = SmartDataframe(combined_data, config={"llm": llm})

        # Step 4: Enter Natural Language Query
        query = st.text_input("Chat with your Data!")

        if st.button("Run Query in Natural Language"):
            if query:
                result = handle_nlp_query(smart_df, query)

                if result is not None:
                    st.write("Query Result:", result)

# Run the Streamlit App
if __name__ == "__main__":
    main()
