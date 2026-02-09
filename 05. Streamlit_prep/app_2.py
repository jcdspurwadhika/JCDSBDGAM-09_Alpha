import streamlit as st
import pandas as pd
import pickle
import dill
import numpy as np
import imblearn
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Customer Behavior Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 2rem 0;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding-bottom: 1rem;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    h3 {
        color: #7f8c8d;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎯 Customer Behavior Prediction Model")
st.markdown("---")

# Load the model and LIME explainer
@st.cache_resource
def load_model():
    with open('pipe_tuned_dtree20260209_1722.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_lime_explainer():
    with open('lime_explainer.dill', 'rb') as f:
        explainer = dill.load(f)
    return explainer

# Load model and explainer
try:
    model = load_model()
    lime_explainer = load_lime_explainer()
    st.success("✅ Model and LIME explainer loaded successfully!")
    
    # Display pipeline info for debugging
    with st.expander("🔧 Pipeline Information (for debugging)"):
        st.write("**Pipeline Steps:**")
        for name, step in model.named_steps.items():
            st.write(f"- `{name}`: {type(step).__name__}")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Create centered container for inputs
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("## 📝 Enter Customer Features")
    st.markdown("---")
    
    # Feature groups with expanders
    with st.expander("📈 Current Quarter Performance", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            num_orders = st.number_input("Number of Orders", min_value=0.0, value=10.0, step=1.0)
            total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, value=1000.0, step=10.0)
            total_freight = st.number_input("Total Freight ($)", min_value=0.0, value=100.0, step=5.0)
        with c2:
            avg_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=100.0, step=5.0)
            days_active_in_quarter = st.number_input("Days Active in Quarter", min_value=0, max_value=92, value=30, step=1)
            num_categories = st.number_input("Number of Categories", min_value=0, value=2, step=1)
    
    with st.expander("📅 Previous Quarter Metrics", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            prev_quarter_num_orders = st.number_input("Previous Quarter Orders", min_value=0.0, value=8.0, step=1.0)
            prev_quarter_total_revenue = st.number_input("Previous Quarter Revenue ($)", min_value=0.0, value=800.0, step=10.0)
        with c2:
            prev_quarter_total_freight = st.number_input("Previous Quarter Freight ($)", min_value=0.0, value=80.0, step=5.0)
    
    with st.expander("🔄 Changes from Previous Quarter", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            orders_change_from_prev = st.number_input("Orders Change", value=2.0, step=1.0)
        with c2:
            revenue_change_from_prev = st.number_input("Revenue Change ($)", value=200.0, step=10.0)
    
    with st.expander("📊 Historical Averages (Last 2 Quarters)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            avg_num_orders_last_2q = st.number_input("Avg Orders (Last 2Q)", min_value=0.0, value=9.0, step=1.0)
        with c2:
            avg_total_revenue_last_2q = st.number_input("Avg Revenue (Last 2Q) ($)", min_value=0.0, value=900.0, step=10.0)
    
    with st.expander("🏆 Lifetime Metrics", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            lifetime_orders = st.number_input("Lifetime Orders", min_value=0.0, value=50.0, step=1.0)
            lifetime_revenue = st.number_input("Lifetime Revenue ($)", min_value=0.0, value=5000.0, step=50.0)
            tenure_quarters = st.number_input("Tenure (Quarters)", min_value=0, value=5, step=1)
        with c2:
            quarters_since_first = st.number_input("Quarters Since First Order", min_value=0, value=4, step=1)
            num_previous_active_quarters = st.number_input("Previous Active Quarters", min_value=0, value=4, step=1)
    
    with st.expander("📉 Trend Indicators", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            is_growing = st.selectbox("Is Growing?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=1)
            is_declining = st.selectbox("Is Declining?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        with c2:
            consecutive_declines = st.number_input("Consecutive Declines", min_value=0, value=0, step=1)
    
    with st.expander("📆 Time-Based Features", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            quarter_of_year = st.selectbox("Quarter of Year", options=[1, 2, 3, 4], index=0)
        with c2:
            is_q4 = st.selectbox("Is Q4?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔮 PREDICT", use_container_width=True)

# Prediction and LIME explanation
if predict_button:
    # Create input dataframe
    input_data = pd.DataFrame({
        'num_orders': [num_orders],
        'total_revenue': [total_revenue],
        'total_freight': [total_freight],
        'avg_order_value': [avg_order_value],
        'days_active_in_quarter': [days_active_in_quarter],
        'num_categories': [num_categories],
        'prev_quarter_num_orders': [prev_quarter_num_orders],
        'prev_quarter_total_revenue': [prev_quarter_total_revenue],
        'prev_quarter_total_freight': [prev_quarter_total_freight],
        'orders_change_from_prev': [orders_change_from_prev],
        'revenue_change_from_prev': [revenue_change_from_prev],
        'avg_num_orders_last_2q': [avg_num_orders_last_2q],
        'avg_total_revenue_last_2q': [avg_total_revenue_last_2q],
        'lifetime_orders': [lifetime_orders],
        'lifetime_revenue': [lifetime_revenue],
        'tenure_quarters': [tenure_quarters],
        'quarters_since_first': [quarters_since_first],
        'num_previous_active_quarters': [num_previous_active_quarters],
        'is_growing': [is_growing],
        'is_declining': [is_declining],
        'consecutive_declines': [consecutive_declines],
        'quarter_of_year': [quarter_of_year],
        'is_q4': [is_q4]
    })
    
    try:
        # Make prediction - handle resampling step carefully
        # Resampling steps don't have transform() method, only fit_resample()
        # During prediction, we need to skip them
        
        # Check if we can use the full pipeline or need to bypass resampling
        try:
            # Try using full pipeline first (works if resampling has proper transform method)
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
        except TypeError as e:
            # If resampling step causes issues, bypass it
            if 'TARGET' in str(e) or 'fit_resample' in str(e):
                st.info("ℹ️ Bypassing resampling step for prediction (only used during training)")
                preprocessed = model.named_steps['preprocessing'].transform(input_data)
                prediction = model.named_steps['dtree'].predict(preprocessed)[0]
                prediction_proba = model.named_steps['dtree'].predict_proba(preprocessed)[0]
            else:
                raise
        
        # Display prediction
        with col2:
            st.markdown("## 🎯 Prediction Result")
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-box positive">
                        ✅ POSITIVE CLASS (Class 1)<br>
                        Probability: {prediction_proba[1]:.2%}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-box negative">
                        ❌ NEGATIVE CLASS (Class 0)<br>
                        Probability: {prediction_proba[0]:.2%}
                    </div>
                """, unsafe_allow_html=True)
            
            # Display probabilities
            st.markdown("### 📊 Prediction Probabilities")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("Class 0 Probability", f"{prediction_proba[0]:.2%}")
            with prob_col2:
                st.metric("Class 1 Probability", f"{prediction_proba[1]:.2%}")
        
        # LIME Explanation
        with col2:
            st.markdown("---")
            st.markdown("## 🔍 LIME Explanation")
            st.markdown("Understanding which features influenced this prediction")
            
            with st.spinner("Generating LIME explanation..."):
                # Create a custom predict function for LIME
                # This must handle the resampling step which doesn't work during prediction
                def predict_fn_for_lime(instances):
                    """
                    Predict function for LIME that bypasses resampling step
                    instances: numpy array of shape (n_samples, n_features)
                    """
                    try:
                        # Convert numpy array back to DataFrame with original feature names
                        feature_names = input_data.columns.tolist()
                        df = pd.DataFrame(instances, columns=feature_names)
                        
                        # Apply only preprocessing and model steps (skip resampling)
                        preprocessed = model.named_steps['preprocessing'].transform(df)
                        
                        # Get predictions directly from the decision tree
                        predictions = model.named_steps['dtree'].predict_proba(preprocessed)
                        
                        return predictions
                    except Exception as e:
                        st.error(f"Error in LIME predict function: {str(e)}")
                        raise
                
                # Generate LIME explanation using raw input
                input_array = input_data.values[0]
                
                try:
                    exp = lime_explainer.explain_instance(
                        input_array, 
                        predict_fn_for_lime,
                        num_features=10
                    )
                    
                    # Display LIME plot
                    fig = exp.as_pyplot_figure()
                    fig.set_size_inches(10, 6)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show feature importance list
                    st.markdown("### 📋 Feature Contributions")
                    lime_list = exp.as_list()
                    
                    # Create a dataframe for better display
                    lime_df = pd.DataFrame(lime_list, columns=['Feature', 'Weight'])
                    lime_df['Impact'] = lime_df['Weight'].apply(lambda x: '🟢 Positive' if x > 0 else '🔴 Negative')
                    lime_df['Absolute Weight'] = lime_df['Weight'].abs()
                    lime_df = lime_df.sort_values('Absolute Weight', ascending=False)
                    
                    st.dataframe(
                        lime_df[['Feature', 'Weight', 'Impact']].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True
                    )
                except Exception as e:
                    st.error(f"❌ Error generating LIME explanation: {str(e)}")
                    st.info("💡 This may be due to pipeline configuration. The prediction above is still valid.")
                    with st.expander("See full error details"):
                        st.exception(e)
                
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p>🚀 Powered by Decision Tree Classification Model</p>
        <p>📊 LIME Explanations for Model Interpretability</p>
    </div>
""", unsafe_allow_html=True)
