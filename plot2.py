        
import os
import streamlit as st
import pandas as pd
from Streamlit_plot import StreamlitStaticalDataPloter


# Function to concatenate multiple CSV files
def concatenate_csv(files):
    # Initialize an empty DataFrame to store concatenated data
    concatenated_data = pd.DataFrame()

    # Loop through selected files and concatenate them vertically
    for file in files:
        data = pd.read_csv(file)
        concatenated_data = pd.concat([concatenated_data, data], axis=0)

    # Reset index
    concatenated_data.reset_index(drop=True, inplace=True)

    return concatenated_data
    
folder_list = [item for item in os.listdir() if os.path.isdir(item)]
        

  
folder_path = st.sidebar.selectbox("WELCOME! Please select your Strategy", folder_list)
df = pd.DataFrame()

if folder_path:
    folder_path = os.path.abspath(folder_path)
    
    # List all CSV files in the selected folder
    csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".csv")]
    
    on = st.sidebar.toggle('Multiple CSV files')

    if on:
        st.write('Please select Multiple CSV Files')
     
        # Display a multi-select box in Streamlit to choose CSV files
        selected_files = st.sidebar.multiselect("Select CSV files to combine", csv_files)
        
        if selected_files and st.sidebar.button('RUN'):

            # Concatenate selected CSV files into a single DataFrame
            df = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in selected_files], ignore_index=True)
    else:
        # Dropdown for selecting a specific CSV file
        selected_csv = st.sidebar.selectbox("Select a CSV file", csv_files)

        # Full path to the selected CSV file
        csv_file = os.path.join(folder_path, selected_csv)
        
        df = pd.read_csv(csv_file)
    
if not df.empty:
      
    df['p/l'] = (df['SPrice'] - df['BPrice']) * df["Qty"]

    # df['p/l'] = pd.to_numeric(self.df['p/l'], errors='coerce')
    df["Volume"] = (df["BPrice"] + df["SPrice"]) * df["Qty"]

    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

    # Convert 'Date' column to datetime type (if not already)
    #df['Date'] = pd.to_datetime(df['Date'])

    # Count unique dates and create a new column 'count'
    df['total_trade_per_day'] = df['Date'].map(df['Date'].value_counts())
        
    df["Expenses"] = df["Volume"] * 0.000925

    plotdata = StreamlitStaticalDataPloter(df)

    plotdata.runplot()
