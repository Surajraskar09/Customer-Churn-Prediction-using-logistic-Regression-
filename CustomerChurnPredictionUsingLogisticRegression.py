# Customer Churn Prediction using Logistic Regression
# Complete Project Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_and_explore_data(self, file_path):
        """Load data and perform initial exploration"""
        print("="*60)
        print("STEP 1: DATA LOADING & EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Load data
        self.df = pd.read_csv("D:\telco.csv")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        # Basic info
        print(f"\nDataset Info:")
        print(self.df.info())
        
        # Check for missing values
        print(f"\nMissing Values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        # Target variable analysis
        print(f"\nTarget Variable Distribution:")
        if 'Churn Label' in self.df.columns:
            target_col = 'Churn Label'
        elif 'Churn' in self.df.columns:
            target_col = 'Churn'
        else:
            print("Target column not found. Please check column names.")
            return
            
        print(self.df[target_col].value_counts())
        print(f"Churn percentage: {self.df[target_col].value_counts(normalize=True) * 100}")
        
        return self.df
    
    def visualize_data(self):
        """Create visualizations for EDA"""
        print("\n" + "="*60)
        print("DATA VISUALIZATION")
        print("="*60)
        
        # Determine target column
        if 'Churn Label' in self.df.columns:
            target_col = 'Churn Label'
        else:
            target_col = 'Churn'
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Target distribution
        self.df[target_col].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Churn Distribution')
        axes[0,0].set_xlabel('Churn Status')
        axes[0,0].set_ylabel('Count')
        
        # 2. Monthly charges distribution by churn
        if 'Monthly Charge' in self.df.columns:
            sns.boxplot(data=self.df, x=target_col, y='Monthly Charge', ax=axes[0,1])
            axes[0,1].set_title('Monthly Charges by Churn Status')
        
        # 3. Tenure distribution by churn
        if 'Tenure in Months' in self.df.columns:
            sns.histplot(data=self.df, x='Tenure in Months', hue=target_col, 
                        bins=30, ax=axes[1,0], alpha=0.7)
            axes[1,0].set_title('Tenure Distribution by Churn Status')
        
        # 4. Contract type vs Churn
        if 'Contract' in self.df.columns:
            pd.crosstab(self.df['Contract'], self.df[target_col]).plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Contract Type vs Churn')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap for numerical features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.df[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap of Numerical Features')
            plt.show()
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Handle target variable
        if 'Churn Label' in df_processed.columns:
            target_col = 'Churn Label'
        else:
            target_col = 'Churn'
        
        # Convert target to binary if needed
        if df_processed[target_col].dtype == 'object':
            df_processed[target_col] = (df_processed[target_col] == 'Yes').astype(int)
        
        # Handle missing values in Total Charges (common issue in telecom data)
        if 'Total Charges' in df_processed.columns:
            df_processed['Total Charges'] = pd.to_numeric(df_processed['Total Charges'], errors='coerce')
            df_processed['Total Charges'].fillna(df_processed['Total Charges'].median(), inplace=True)
        
        # Select relevant features (exclude ID columns and target)
        exclude_cols = ['Customer ID', 'customerID', target_col, 'Customer Status', 'Churn Category', 'Churn Reason']
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        
        # Separate numerical and categorical features
        numerical_features = df_processed[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df_processed[feature_cols].select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Handle categorical variables with Label Encoding
        for col in categorical_features:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Prepare features and target
        X = df_processed[feature_cols]
        y = df_processed[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if numerical_features:
            X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
            X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        self.feature_columns = feature_cols
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the Logistic Regression model"""
        print("\n" + "="*60)
        print("STEP 3: MODEL BUILDING")
        print("="*60)
        
        # Initialize and train the model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train, y_train)
        
        print("Logistic Regression model trained successfully!")
        return self.model
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate the model performance"""
        print("\n" + "="*60)
        print("STEP 4: MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_test_pred))
        
        # Visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        
        # Feature Importance (Coefficients)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_[0]
        })
        feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False).head(10)
        
        axes[2].barh(feature_importance['feature'], feature_importance['coefficient'])
        axes[2].set_title('Top 10 Feature Coefficients')
        axes[2].set_xlabel('Coefficient Value')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    def analyze_feature_importance(self):
        """Analyze which features contribute most to churn"""
        print("\n" + "="*60)
        print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get feature coefficients
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': abs(self.model.coef_[0])
        })
        
        # Sort by absolute coefficient value
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        print("Top 10 Most Important Features for Churn Prediction:")
        print(feature_importance.head(10))
        
        # Visualize top features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
        plt.barh(top_features['feature'], top_features['coefficient'], color=colors, alpha=0.7)
        plt.xlabel('Coefficient Value')
        plt.title('Top 15 Feature Coefficients (Red: Increases Churn, Green: Decreases Churn)')
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def predict_single_customer(self, customer_data):
        """Predict churn for a single customer"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Process the input data similar to training data
        processed_data = customer_data.copy()
        
        # Apply label encoders to categorical variables
        for col, encoder in self.label_encoders.items():
            if col in processed_data:
                try:
                    processed_data[col] = encoder.transform([processed_data[col]])[0]
                except:
                    # Handle unknown categories
                    processed_data[col] = 0
        
        # Create DataFrame with all required features
        input_df = pd.DataFrame([processed_data])
        
        # Ensure all features are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Scale numerical features
        if self.scaler:
            numerical_features = input_df.select_dtypes(include=[np.number]).columns
            input_df[numerical_features] = self.scaler.transform(input_df[numerical_features])
        
        # Make prediction
        prediction = self.model.predict(input_df[self.feature_columns])[0]
        probability = self.model.predict_proba(input_df[self.feature_columns])[0][1]
        
        return prediction, probability
    
    def generate_business_insights(self, feature_importance):
        """Generate business insights and recommendations"""
        print("\n" + "="*60)
        print("STEP 6: BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Top churn drivers (positive coefficients)
        churn_drivers = feature_importance[feature_importance['coefficient'] > 0].head(5)
        print("TOP CHURN DRIVERS (Features that increase churn probability):")
        for _, row in churn_drivers.iterrows():
            print(f"• {row['feature']}: {row['coefficient']:.4f}")
        
        # Top retention factors (negative coefficients)
        retention_factors = feature_importance[feature_importance['coefficient'] < 0].head(5)
        print(f"\nTOP RETENTION FACTORS (Features that decrease churn probability):")
        for _, row in retention_factors.iterrows():
            print(f"• {row['feature']}: {row['coefficient']:.4f}")
        
        print(f"\n📋 BUSINESS RECOMMENDATIONS:")
        print("1. 🎯 TARGETED RETENTION CAMPAIGNS:")
        print("   • Focus on customers with high-risk profiles")
        print("   • Offer personalized discounts to month-to-month customers")
        print("   • Provide better customer service to dissatisfied customers")
        
        print("\n2. 💰 PRICING STRATEGIES:")
        print("   • Review pricing for customers with high monthly charges")
        print("   • Introduce flexible pricing tiers")
        print("   • Offer bundle discounts for multiple services")
        
        print("\n3. 📞 PROACTIVE CUSTOMER SUCCESS:")
        print("   • Implement early warning system using this model")
        print("   • Regular check-ins with high-risk customers")
        print("   • Improve customer onboarding process")
        
        print("\n4. 📊 PRODUCT IMPROVEMENTS:")
        print("   • Address service quality issues")
        print("   • Enhance customer support experience")
        print("   • Develop loyalty programs for long-term customers")

# Example usage and demo
def main():
    print("🚀 CUSTOMER CHURN PREDICTION PROJECT")
    print("Using Logistic Regression for Telecom Dataset")
    print("="*60)
    
    # Initialize the predictor
    churn_predictor = ChurnPredictor()
    
    # Note: Replace with actual dataset path
    print("📁 Please place your dataset file in the same directory")
    print("💡 Expected file: 'telco_customer_churn.csv'")
    print("\n🔧 To run this project:")
    print("1. Download the dataset from Kaggle")
    print("2. Update the file_path variable below")
    print("3. Run the complete pipeline")
    
    # STEP 1: UPDATE THE FILE PATH HERE 👇
    # Replace with your actual CSV file path
    file_path = "telco_customer_churn.csv"  # 🔧 CHANGE THIS PATH
    
    # Alternative path examples:
    # file_path = "data/telco_customer_churn.csv"                    # If in data folder
    # file_path = "C:/Users/YourName/Desktop/telco_customer_churn.csv"  # Windows absolute path
    # file_path = "/home/username/datasets/telco_customer_churn.csv"    # Linux/Mac absolute path
    
    print(f"📁 Looking for dataset at: {file_path}")
    
    # Check if file exists before proceeding
    import os
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        print("💡 Please download the dataset and update the file_path variable")
        print("📥 Dataset URL: https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3")
        return
    
    # Load and explore data
    df = churn_predictor.load_and_explore_data(file_path)
    
    # Visualize data
    churn_predictor.visualize_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = churn_predictor.preprocess_data()
    
    # Train model
    model = churn_predictor.train_model(X_train, y_train)
    
    # Evaluate model
    metrics = churn_predictor.evaluate_model(X_train, X_test, y_train, y_test)
    
    # Analyze feature importance
    feature_importance = churn_predictor.analyze_feature_importance()
    
    # Generate business insights
    churn_predictor.generate_business_insights(feature_importance)
    
    # Example prediction for a single customer
    sample_customer = {
        'Gender': 'Male',
        'Age': 45,
        'Tenure in Months': 12,
        'Monthly Charge': 75.50,
        'Contract': 'Month-to-Month',
        'Internet Service': 'Yes',
        'Phone Service': 'Yes'
        # Add more features as needed
    }
    
    prediction, probability = churn_predictor.predict_single_customer(sample_customer)
    print(f"\n🔮 PREDICTION FOR SAMPLE CUSTOMER:")
    print(f"Churn Prediction: {'Will Churn' if prediction == 1 else 'Will Not Churn'}")
    print(f"Churn Probability: {probability:.2%}")
if __name__ == "__main__":
    main()