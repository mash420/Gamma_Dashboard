import streamlit as st
from gammas import Gamma  # Import your Gamma class from gamma_class.py
import time
import threading

# Function to fetch data
@st.cache_resource
def fetch_data(symbol, days_to_expo):
    spx = Gamma(symbol, int(days_to_expo))
    path_chain, chain, path_vol = spx.get_same_day()  # Fetch data for the same day
    path, gex_df, path_vol_all = spx.get_all_chains()  # Fetch data for all expiration dates
    return path_chain, path, path_vol, path_vol_all

# Function to update the data every 5 minutes
def update_data():
    while True:
        time.sleep(300)
        st.experimental_rerun()  # Rerun the app to update the data

# Function to display the dashboard
def display_dashboard():
    symbol = st.sidebar.text_input('Enter Symbol', value='SPX')
    days_to_expo = st.sidebar.text_input('Days to Expiration', value='98')
    submit_button = st.sidebar.button('Submit')

    if submit_button:
        path_chain, path, path_vol, path_vol_all = fetch_data(symbol, days_to_expo)

        # Streamlit app title
        st.title('Options Analysis Dashboard')

        # Arrange components in two columns of two
        col1, col2 = st.columns(2)

        # Display same day gamma exposure plot and same day volume plot
        with col1:
            st.subheader('Same Day Gamma Exposure Plot')
            st.image(path_chain, use_column_width=True)

            st.subheader('Same Day Volume Plot')
            st.image(path_vol, use_column_width=True)

        # Display 98-day gamma exposure plot and 98-day volume plot
        with col2:
            st.subheader(f'{days_to_expo} Day Gamma Exposure Plot')
            st.image(path, use_column_width=True)

            st.subheader(f'{days_to_expo} Day Volume Plot')
            st.image(path_vol_all, use_column_width=True)

# Set page layout to wide mode
st.set_page_config(layout="wide")

# Start a thread to update the data
update_thread = threading.Thread(target=update_data)
update_thread.daemon = True
update_thread.start()

# Display the dashboard
display_dashboard()
