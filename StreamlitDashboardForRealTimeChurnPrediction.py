# Streamlit Dashboard for Real-Time Customer Churn Prediction
# Save as: streamlit_app.py
# Run with: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e1e5e9;
}
.churn-high {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
    padding: 1rem;
    border-radius: 0.25rem;
}
.churn-low {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
    padding: 1rem;
    border-radius: 0.25rem;
}
.header-style {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

class StreamlitChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.feature_importance = None
    
    def create_sample_model_data(self):
        """Create sample model and data for demonstration"""
        # This would normally be loaded from saved model files
        # For demo purposes, we'll create sample feature importance data
        sample_features = [
            'Monthly Charge', 'Tenure in Months', 'Total Charges', 'Contract_Month-to-Month',
            'Internet Service_Fiber Optic', 'Payment Method_Electronic check', 'Senior Citizen',
            'Paperless Billing', 'Phone Service', 'Multiple Lines', 'Online Security',
            'Online Backup', 'Device Protection Plan', 'Tech Support', 'Streaming TV'
        ]
        
        sample_importance = pd.DataFrame({
            'feature': sample_features,
            'coefficient': [0.89, -0.67, 0.45, 0.78, 0.56, 0.34, 0.23, 0.19, -0.15, 0.12, -0.34, -0.28, -0.21, -0.18, 0.16],
            'abs_coefficient': [0.89, 0.67, 0.45, 0.78, 0.56, 0.34, 0.23, 0.19, 0.15, 0.12, 0.34, 0.28, 0.21, 0.18, 0.16]
        }).sort_values('abs_coefficient', ascending=False)
        
        return sample_importance

def main():
    st.markdown('<h1 class="header-style">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize predictor
    predictor = StreamlitChurnPredictor()
    
    # Sidebar for navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Predict Churn", "📈 Model Insights", "📊 Analytics Dashboard"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Predict Churn":
        show_prediction_page()
    elif page == "📈 Model Insights":
        show_insights_page(predictor)
    elif page == "📊 Analytics Dashboard":
        show_analytics_page()

def show_home_page():
    st.header("Welcome to the Customer Churn Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📈 Model Accuracy",
            value="87.3%",
            delta="2.1%"
        )
    
    with col2:
        st.metric(
            label="🎯 Precision",
            value="84.7%",
            delta="1.5%"
        )
    
    with col3:
        st.metric(
            label="🔄 Recall",
            value="82.1%",
            delta="0.8%"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Overview")
        st.write("""
        This system uses **Logistic Regression** to predict customer churn in the telecom industry.
        
        **Key Features:**
        - Real-time churn prediction
        - Interactive customer input form
        - Detailed probability scores
        - Business insights and recommendations
        - Feature importance analysis
        
        **Use Cases:**
        - Identify high-risk customers
        - Targeted retention campaigns
        - Customer lifetime value optimization
        - Proactive customer success
        """)
    
    with col2:
        st.subheader("📊 Model Performance")
        
        # Create a sample performance chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        scores = [87.3, 84.7, 82.1, 83.4, 89.2]
        
        fig = go.Figure(data=go.Bar(
            x=metrics,
            y=scores,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            yaxis_title="Score (%)",
            xaxis_title="Metrics",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_page():
    st.header("🔮 Customer Churn Prediction")
    st.write("Enter customer details below to predict churn probability:")
    
    # Create input form
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📋 Basic Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 80, 45)
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            married = st.selectbox("Married", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
        with col2:
            st.subheader("📞 Service Information")
            tenure = st.slider("Tenure (Months)", 1, 72, 24)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber Optic"])
            contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
            payment_method = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])
        
        with col3:
            st.subheader("💰 Billing Information")
            monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
            total_charges = st.slider("Total Charges ($)", 20.0, 8000.0, 1500.0)
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        
        # Submit button
        submitted = st.form_submit_button("🚀 Predict Churn", use_container_width=True)
        
        if submitted:
            # Calculate prediction (demo calculation)
            # In real implementation, this would use the trained model
            risk_factors = 0
            
            # Simple risk calculation for demo
            if contract == "Month-to-Month":
                risk_factors += 0.3
            if monthly_charges > 80:
                risk_factors += 0.25
            if tenure < 12:
                risk_factors += 0.2
            if payment_method == "Electronic Check":
                risk_factors += 0.15
            if senior_citizen == "Yes":
                risk_factors += 0.1
            
            churn_probability = min(risk_factors, 0.95)
            churn_prediction = churn_probability > 0.5
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if churn_prediction:
                    st.markdown(f"""
                    <div class="churn-high">
                        <h3>⚠️ HIGH CHURN RISK</h3>
                        <p><strong>Prediction:</strong> Customer likely to churn</p>
                        <p><strong>Probability:</strong> {churn_probability:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="churn-low">
                        <h3>✅ LOW CHURN RISK</h3>
                        <p><strong>Prediction:</strong> Customer unlikely to churn</p>
                        <p><strong>Probability:</strong> {churn_probability:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = churn_probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommended Actions")
            
            if churn_prediction:
                st.error("🚨 **IMMEDIATE ACTION REQUIRED**")
                recommendations = [
                    "Contact customer within 24 hours with retention offer",
                    "Offer 15-20% discount on next 3 months",
                    "Provide free premium support for 6 months",
                    "Switch to annual contract with discount",
                    "Assign dedicated customer success manager"
                ]
            else:
                st.success("✅ **CUSTOMER IN GOOD STANDING**")
                recommendations = [
                    "Continue regular engagement",
                    "Consider upselling additional services",
                    "Invite to loyalty program",
                    "Send satisfaction survey",
                    "Offer referral incentives"
                ]
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

def show_insights_page(predictor):
    st.header("📈 Model Insights & Feature Importance")
    
    # Get sample feature importance data
    feature_importance = predictor.create_sample_model_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔝 Top Churn Drivers")
        churn_drivers = feature_importance[feature_importance['coefficient'] > 0].head(8)
        
        fig = px.bar(
            churn_drivers, 
            x='coefficient', 
            y='feature',
            orientation='h',
            color='coefficient',
            color_continuous_scale='Reds',
            title="Features that Increase Churn Risk"
        )
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🛡️ Top Retention Factors")
        retention_factors = feature_importance[feature_importance['coefficient'] < 0].head(8)
        retention_factors['abs_coef'] = abs(retention_factors['coefficient'])
        
        fig = px.bar(
            retention_factors, 
            x='abs_coef', 
            y='feature',
            orientation='h',
            color='abs_coef',
            color_continuous_scale='Greens',
            title="Features that Reduce Churn Risk"
        )
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    st.subheader("📊 Complete Feature Importance Analysis")
    
    # Add interpretation column
    feature_importance['impact'] = feature_importance['coefficient'].apply(
        lambda x: "Increases Churn" if x > 0 else "Reduces Churn"
    )
    
    feature_importance['strength'] = feature_importance['abs_coefficient'].apply(
        lambda x: "Strong" if x > 0.5 else "Moderate" if x > 0.3 else "Weak"
    )
    
    st.dataframe(
        feature_importance[['feature', 'coefficient', 'impact', 'strength']].round(4),
        use_container_width=True
    )
    
    # Business insights
    st.markdown("---")
    st.subheader("💼 Key Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 High-Risk Customer Profile:**
        - Month-to-month contract customers
        - High monthly charges (>$80)
        - Short tenure (<12 months)
        - Electronic check payment method
        - Senior citizens
        - Fiber optic internet users
        """)
    
    with col2:
        st.markdown("""
        **✅ Loyal Customer Profile:**
        - Long-term contract holders
        - Longer tenure customers
        - Additional service subscribers
        - Online security users
        - Tech support users
        - Multiple service bundles
        """)

def show_analytics_page():
    st.header("📊 Analytics Dashboard")
    
    # Create sample data for visualization
    np.random.seed(42)
    
    # Sample customer data
    sample_data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Total_Customers': [7043, 7234, 7156, 7398, 7521, 7445],
        'Churned_Customers': [1521, 1634, 1456, 1598, 1621, 1545],
        'Churn_Rate': [21.6, 22.6, 20.3, 21.6, 21.5, 20.8]
    }
    
    df_analytics = pd.DataFrame(sample_data)
    
    # Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Total Customers", "7,445", "52 from last month")
    with col2:
        st.metric("📉 Churned This Month", "1,545", "-76 from last month")
    with col3:
        st.metric("📊 Churn Rate", "20.8%", "-0.7% from last month")
    with col4:
        st.metric("💰 Revenue at Risk", "$2.1M", "-$150K from last month")
    
    st.markdown("---")
    
    # Row 2: Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Monthly Churn Trend")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_analytics['Month'],
            y=df_analytics['Churn_Rate'],
            mode='lines+markers',
            name='Churn Rate (%)',
            line=dict(color='#ff6b6b', width=3)
        ))
        
        fig.update_layout(
            title="Churn Rate Over Time",
            yaxis_title="Churn Rate (%)",
            xaxis_title="Month",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Distribution")
        
        risk_data = {
            'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Customers': [4234, 2156, 1055],
            'Colors': ['#2ecc71', '#f39c12', '#e74c3c']
        }
        
        fig = px.pie(
            values=risk_data['Customers'],
            names=risk_data['Risk Level'],
            color_discrete_sequence=risk_data['Colors'],
            title="Customer Risk Distribution"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: Detailed Analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💼 Churn by Contract Type")
        
        contract_data = {
            'Contract Type': ['Month-to-Month', 'One Year', 'Two Year'],
            'Customers': [3875, 2456, 1114],
            'Churn Rate': [42.7, 11.3, 2.8]
        }
        
        fig = px.bar(
            x=contract_data['Contract Type'],
            y=contract_data['Churn Rate'],
            color=contract_data['Churn Rate'],
            color_continuous_scale='Reds',
            title="Churn Rate by Contract Type"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📱 Churn by Service Type")
        
        service_data = {
            'Service': ['Phone Only', 'Internet Only', 'Phone + Internet', 'Full Bundle'],
            'Churn Rate': [15.2, 28.6, 31.4, 18.9]
        }
        
        fig = px.bar(
            x=service_data['Service'],
            y=service_data['Churn Rate'],
            color=service_data['Churn Rate'],
            color_continuous_scale='Blues',
            title="Churn Rate by Service Type"
        )
        
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Actionable insights
    st.markdown("---")
    st.subheader("🎯 Actionable Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📞 Focus Area: Month-to-Month Customers**
        - 42.7% churn rate
        - Immediate retention campaigns needed
        - Offer contract incentives
        """)
    
    with col2:
        st.warning("""
        **💻 Service Strategy: Internet Services**
        - High churn in internet-only customers
        - Bundle promotion opportunities
        - Improve service quality
        """)
    
    with col3:
        st.success("""
        **🏆 Success Story: Two-Year Contracts**
        - Only 2.8% churn rate
        - Expand two-year promotions
        - Replicate success factors
        """)

# Additional utility functions
def load_model():
    """Load the trained model (placeholder for real implementation)"""
    # In real implementation, you would load the pickled model
    # with open('churn_model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # return model
    pass

def save_prediction_log(customer_data, prediction, probability):
    """Save prediction to log file for tracking"""
    # In real implementation, you would save to database
    pass

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 Customer Churn Prediction System v1.0</p>
        <p>Built with Streamlit, Python, and Scikit-learn</p>
        <p>For support, contact: data-science-team@company.com</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()

# Instructions to run:
"""
To run this Streamlit dashboard:

1. Save this code as 'streamlit_app.py'
2. Install required packages:
   pip install streamlit plotly pandas numpy

3. Run the dashboard:
   streamlit run streamlit_app.py

4. For production deployment:
   - Connect to your trained model (replace sample calculations)
   - Add database connectivity for logging predictions
   - Implement user authentication if needed
   - Add more sophisticated analytics and reporting
"""