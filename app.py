import streamlit as st
import joblib
import pandas as pd
import numpy as np
from model import CascadingImputer, fill_stage_drained_with_unKnow

# Load the trained model
try:
    model_pipeline = joblib.load('model_pipeline.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Personality Prediction App",
    page_icon="🧠",
    layout="wide"
)

# App title and description
st.title("Personality Prediction App")
st.markdown("""
This app predicts personality traits (Introvert/Extrovert) based on social behavior patterns. \n
Adjust the sliders and selectors below and click 'Predict' to see the results. \n
Trained Model: 97% accuracy on untrained data https://www.kaggle.com/competitions/playground-series-s5e7 \n
Github repo: https://github.com/dsrrenCodes/Personality-Detector
""")

# Create a form for user input
with st.form("prediction_form"):
    st.header("Input Parameters")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Numerical inputs (0-10 scale)
        st.markdown("### Social Behavior Metrics (0-10 scale)")
        st.markdown("Rate each aspect of your social behavior on a scale from 0 to 10, where 0 is the minimum and 10 is the maximum.")
        
        time_spent_alone = st.slider(
            "Time Spent Alone", 
            min_value=0, 
            max_value=10, 
            value=5,
            help="How much time do you typically spend alone? 0 = Almost no time alone, 10 = Most of the time alone"
        )
        
        social_event_attendance = st.slider(
            "Social Event Attendance", 
            min_value=0, 
            max_value=10, 
            value=5,
            help="How often do you attend social events? 0 = Never attend, 5 = Occasionally, 10 = Very frequently"
        )
        
        going_outside = st.slider(
            "Outdoor Activity Level", 
            min_value=0, 
            max_value=10, 
            value=5,
            help="How often do you go outside for non-essential activities? 0 = Rarely leave home, 10 = Always out and about"
        )
        
        friends_circle_size = st.slider(
            "Social Circle Size", 
            min_value=0, 
            max_value=10, 
            value=5,
            help="How large is your circle of friends? 0 = Very small (1-2 close friends), 10 = Very large (many friends and acquaintances)"
        )
    
    with col2:
        st.markdown("### Online & Social Preferences")
        
        post_frequency = st.slider(
            "Social Media Activity", 
            min_value=0, 
            max_value=10, 
            value=5,
            help="How often do you post on social media? 0 = Never post, 5 = Occasionally, 10 = Multiple times daily"
        )
        
        st.markdown("### Social Comfort Levels")
        
        stage_fear = st.selectbox(
            "Stage/Public Speaking Comfort", 
            options=["Yes", "No"], 
            index=0,
            help="Select 'Yes' if you experience nervousness or fear when speaking in front of groups or on stage"
        )
        
        drained_after_socializing = st.selectbox(
            "Social Battery", 
            options=["Yes", "No"], 
            index=0,
            help="Select 'Yes' if you typically feel mentally or physically drained after social interactions"
        )
    
    # Submit button
    submitted = st.form_submit_button("Predict Personality")

# When the form is submitted
if submitted:
    try:
        # Create input data as a DataFrame
        input_data = pd.DataFrame({
            'Time_spent_Alone': [time_spent_alone],
            'Social_event_attendance': [social_event_attendance],
            'Going_outside': [going_outside],
            'Friends_circle_size': [friends_circle_size],
            'Post_frequency': [post_frequency],
            'Stage_fear': [stage_fear],
            'Drained_after_socializing': [drained_after_socializing]
        })
        
        # Make prediction
        prediction = model_pipeline.predict(input_data)[0]
        prediction_proba = model_pipeline.predict_proba(input_data)[0]
        
        # Map prediction to human-readable labels
        personality_map = {1: 'Extrovert', 0: 'Introvert'}
        prediction_label = personality_map.get(prediction, str(prediction))
        
        # Display results
        st.subheader("Prediction Results")
        
        # Create a nice progress bar for the prediction confidence
        confidence = max(prediction_proba) * 100
        st.metric("Predicted Personality", prediction_label)
        st.progress(int(confidence) / 100, text=f"Confidence: {confidence:.1f}%")
        
        # Show probability distribution with labels
        st.write("\n**Probability Distribution:**")
        proba_df = pd.DataFrame({
            'Personality': ['Introvert', 'Extrovert'],
            'Probability': [prediction_proba[0] * 100, prediction_proba[1] * 100]
        }).sort_values('Probability', ascending=False)
        
        # Create a bar chart with custom colors
        import plotly.express as px
        fig = px.bar(proba_df, 
                     x='Personality', 
                     y='Probability',
                     color='Personality',
                     color_discrete_map={'Introvert': '#1f77b4', 'Extrovert': '#ff7f0e'},
                     text_auto='.1f',
                     title='Personality Probability Distribution')
        fig.update_traces(textfont_size=12, textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title='Probability (%)')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add some styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stProgress > div > div > div {
        background-color: #e0e0e0;
    }
    .footer {
        margin-top: 3rem;
        padding: 1.5rem;
        text-align: center;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
    .social-links a {
        margin: 0 10px;
        color: #1f77b4;
        text-decoration: none;
    }
    .social-links a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Add footer with social links
st.markdown("""
<div class="footer">
    <h4>Connect with me</h4>
    <div class="social-links">
        <a href="https://github.com/dsrrenCodes" target="_blank">GitHub</a>
    </div>
    <p>© 2025 Personality Prediction App</p>
</div>
""", unsafe_allow_html=True)
