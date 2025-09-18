import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Ratio Prediction Model",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🚌 Transportation Ratio Prediction Model")
st.markdown("This application predicts the ratio based on transportation parameters using a Random Forest model.")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_joblib.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

if model is not None:
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input Parameters")
        
        # Input fields based on the parameter ranges
        jumlah_kebutuhan_unit = st.selectbox(
            "Jumlah Kebutuhan Unit",
            options=[9, 11, 10, 12, 8, 13, 14],
            index=2  # Default to 10
        )
        
        jumlah_keberangkatan_perhari = st.selectbox(
            "Jumlah Keberangkatan Per Hari",
            options=sorted([46, 59, 54, 50, 47, 60, 58, 43, 42, 41, 51, 45, 40, 56, 49]),
            index=7  # Default to middle value
        )
        
        okupansi = st.selectbox(
            "Okupansi",
            options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            index=4  # Default to 0.5
        )
    
    with col2:
        st.subheader("📝 Input Parameters (continued)")
        
        jam_keberangkatan_menit = st.selectbox(
            "Jam Keberangkatan (menit)",
            options=sorted([1110, 750, 720, 360, 480, 900, 540, 810, 390, 1020, 
                          690, 1050, 330, 870, 510, 1320, 780, 450, 1140, 1290, 
                          570, 1200, 960, 990, 300, 600, 1260, 420, 630, 930, 
                          1080, 660, 1230, 1170, 840]),
            format_func=lambda x: f"{x//60:02d}:{x%60:02d}"  # Convert minutes to HH:MM format
        )
        
        asal_kota = st.selectbox(
            "Asal Kota",
            options=['A', 'B']
        )
        
        # Create prediction button
        predict_button = st.button("🔮 Predict Ratio", type="primary", use_container_width=True)
    
    # Prediction section
    if predict_button:
        # Create input dataframe
        input_data = pd.DataFrame({
            'jumlah_kebutuhan_unit': [jumlah_kebutuhan_unit],
            'jumlah_keberangkatan_perhari': [jumlah_keberangkatan_perhari],
            'okupansi': [okupansi],
            'jam_keberangkatan_menit': [jam_keberangkatan_menit],
            'asal_kota': [asal_kota]
        })
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display prediction
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric(
                    label="Predicted Ratio",
                    value=f"{prediction:.4f}",
                    delta=None
                )
            
            # Store prediction in session state for visualization
            if 'predictions' not in st.session_state:
                st.session_state.predictions = []
            
            st.session_state.predictions.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'jumlah_kebutuhan_unit': jumlah_kebutuhan_unit,
                'jumlah_keberangkatan_perhari': jumlah_keberangkatan_perhari,
                'okupansi': okupansi,
                'jam_keberangkatan_menit': jam_keberangkatan_menit,
                'asal_kota': asal_kota
            })
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Visualization section
    st.markdown("---")
    st.subheader("📊 Prediction Trends and Analysis")
    
    if 'predictions' in st.session_state and len(st.session_state.predictions) > 0:
        # Convert predictions to DataFrame
        df_predictions = pd.DataFrame(st.session_state.predictions)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Prediction Timeline", "Parameter Impact", "Distribution", "Data Table"])
        
        with tab1:
            # Time series of predictions
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=df_predictions['timestamp'],
                y=df_predictions['prediction'],
                mode='lines+markers',
                name='Predictions',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8)
            ))
            fig_timeline.update_layout(
                title="Prediction Timeline",
                xaxis_title="Time",
                yaxis_title="Predicted Ratio",
                hovermode='x unified'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with tab2:
            # Parameter impact visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Okupansi vs Prediction
                fig_okupansi = px.scatter(
                    df_predictions, 
                    x='okupansi', 
                    y='prediction',
                    color='asal_kota',
                    size='jumlah_kebutuhan_unit',
                    title="Okupansi vs Predicted Ratio"
                )
                st.plotly_chart(fig_okupansi, use_container_width=True)
            
            with col2:
                # Keberangkatan vs Prediction
                fig_keberangkatan = px.scatter(
                    df_predictions, 
                    x='jumlah_keberangkatan_perhari', 
                    y='prediction',
                    color='asal_kota',
                    title="Keberangkatan per Hari vs Predicted Ratio"
                )
                st.plotly_chart(fig_keberangkatan, use_container_width=True)
        
        with tab3:
            # Distribution of predictions
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=df_predictions['prediction'],
                nbinsx=20,
                name='Prediction Distribution'
            ))
            fig_dist.update_layout(
                title="Distribution of Predictions",
                xaxis_title="Predicted Ratio",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df_predictions['prediction'].mean():.4f}")
            with col2:
                st.metric("Std Dev", f"{df_predictions['prediction'].std():.4f}")
            with col3:
                st.metric("Min", f"{df_predictions['prediction'].min():.4f}")
            with col4:
                st.metric("Max", f"{df_predictions['prediction'].max():.4f}")
        
        with tab4:
            # Display data table
            st.dataframe(
                df_predictions[['timestamp', 'prediction', 'jumlah_kebutuhan_unit', 
                               'jumlah_keberangkatan_perhari', 'okupansi', 
                               'jam_keberangkatan_menit', 'asal_kota']].sort_values('timestamp', ascending=False),
                use_container_width=True
            )
            
            # Clear predictions button
            if st.button("🗑️ Clear All Predictions"):
                st.session_state.predictions = []
                st.rerun()
    else:
        st.info("No predictions yet. Make a prediction to see trends and analysis.")
    
    # Additional analysis section
    with st.expander("🔍 Advanced Analysis"):
        st.subheader("Sensitivity Analysis")
        
        # Allow user to select a parameter to analyze
        param_to_analyze = st.selectbox(
            "Select parameter to analyze",
            ["okupansi", "jumlah_keberangkatan_perhari", "jumlah_kebutuhan_unit"]
        )
        
        if st.button("Run Sensitivity Analysis"):
            # Create a base input
            base_input = pd.DataFrame({
                'jumlah_kebutuhan_unit': [jumlah_kebutuhan_unit],
                'jumlah_keberangkatan_perhari': [jumlah_keberangkatan_perhari],
                'okupansi': [okupansi],
                'jam_keberangkatan_menit': [jam_keberangkatan_menit],
                'asal_kota': [asal_kota]
            })
            
            # Define ranges for sensitivity analysis
            if param_to_analyze == "okupansi":
                values = np.linspace(0.1, 1.0, 10)
            elif param_to_analyze == "jumlah_keberangkatan_perhari":
                values = range(40, 61, 2)
            else:  # jumlah_kebutuhan_unit
                values = range(8, 15)
            
            predictions = []
            for value in values:
                test_input = base_input.copy()
                test_input[param_to_analyze] = value
                pred = model.predict(test_input)[0]
                predictions.append(pred)
            
            # Plot sensitivity
            fig_sensitivity = go.Figure()
            fig_sensitivity.add_trace(go.Scatter(
                x=list(values),
                y=predictions,
                mode='lines+markers',
                name=f'Impact of {param_to_analyze}'
            ))
            fig_sensitivity.update_layout(
                title=f"Sensitivity Analysis: {param_to_analyze}",
                xaxis_title=param_to_analyze,
                yaxis_title="Predicted Ratio"
            )
            st.plotly_chart(fig_sensitivity, use_container_width=True)

else:
    st.error("Model could not be loaded. Please ensure 'model_joblib.pkl' is in the correct directory.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Random Forest Regression Model")