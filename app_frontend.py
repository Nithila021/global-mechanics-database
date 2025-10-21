
import streamlit as st
import requests
import pandas as pd

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")

st.title("Global Mechanics Data Platform")

# Sidebar for navigation
tab = st.sidebar.radio("Navigate", ["Ingestion & Normalization", "Database Viewer", "Predictive Maintenance"])

if tab == "Ingestion & Normalization":
    st.header("Data Ingestion & Normalization")

    uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True, type='csv')

    if uploaded_files:
        for file in uploaded_files:
            files = {"file": (file.name, file.getvalue(), "text/csv")}
            response = requests.post(f"{BACKEND_URL}/upload_csv/", files=files)
            if response.status_code == 200:
                st.success(f"Successfully uploaded {file.name}")
            else:
                st.error(f"Error uploading {file.name}: {response.text}")

    if st.button("Trigger BCNF Normalization"):
        response = requests.post(f"{BACKEND_URL}/normalize/")
        if response.status_code == 200:
            st.success("BCNF Normalization triggered successfully!")
        else:
            st.error(f"Error triggering normalization: {response.text}")

elif tab == "Database Viewer":
    st.header("Global Database Viewer")

    try:
        response = requests.get(f"{BACKEND_URL}/tables/")
        if response.status_code == 200:
            tables = response.json().get("tables", [])
            selected_table = st.selectbox("Select a table to view", tables)

            if selected_table:
                page = st.session_state.get(f"{selected_table}_page", 1)

                col1, col2 = st.columns(2)
                if col1.button("Previous Page"):
                    if page > 1:
                        page -= 1
                if col2.button("Next Page"):
                    page += 1
                
                st.session_state[f"{selected_table}_page"] = page

                data_response = requests.get(f"{BACKEND_URL}/data/{selected_table}/?page={page}")
                if data_response.status_code == 200:
                    data = data_response.json()
                    if data:
                        st.dataframe(pd.DataFrame(data))
                    else:
                        st.warning("No more data to display.")
                        if page > 1:
                            st.session_state[f"{selected_table}_page"] = page - 1 # Go back to last valid page
                else:
                    st.error(f"Error fetching data: {data_response.text}")
        else:
            st.error(f"Error fetching table list: {response.text}")
    except requests.exceptions.ConnectionError as e:
        st.error(f"Could not connect to the backend: {e}. Please ensure the backend is running.")

elif tab == "Predictive Maintenance":
    st.header("Predictive Maintenance")

    with st.form("prediction_form"):
        st.write("Fill in the vehicle details to predict maintenance needs. Leave fields blank for imputation.")
        vehicle_model = st.text_input("Vehicle Model")
        mileage = st.number_input("Mileage", min_value=0, value=0)
        maintenance_history = st.selectbox("Maintenance History", ["", "Good", "Average", "Poor"])
        reported_issues = st.number_input("Reported Issues", min_value=0, value=0)
        vehicle_age = st.number_input("Vehicle Age", min_value=0, value=0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            payload = {
                "Vehicle_Model": vehicle_model if vehicle_model else None,
                "Mileage": mileage if mileage > 0 else None,
                "Maintenance_History": maintenance_history if maintenance_history else None,
                "Reported_Issues": reported_issues if reported_issues > 0 else None,
                "Vehicle_Age": vehicle_age if vehicle_age > 0 else None
            }

            response = requests.post(f"{BACKEND_URL}/predict/", json=payload)

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                probability = result.get("probability")

                st.success(f"Maintenance Required: {'Yes' if prediction == 1 else 'No'}")
                st.info(f"Confidence: {probability:.2f}")
            else:
                st.error(f"Error making prediction: {response.text}")
