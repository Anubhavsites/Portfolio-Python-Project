import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Set page config for Streamlit Web App
st.set_page_config(page_title="Blinkit Predictive Analytics", layout="wide")
st.title(" Blinkit Advanced Predictive Analytics & ML Pipeline")
st.markdown("*Moving beyond dashboards into Predictive Machine Learning*")

# ----------------------------------------------------------------------------
# STEP 1: LOAD & AUTOMATE DATA CLEANING (Question 1)
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    # Replace 'blinkit_data.csv' with your actual file path if needed
    # Generating dummy data structure mirroring your exact dashboard for standalone execution
    try:
        df = pd.read_csv("blinkit_data.csv")
    except:
        # Fallback dataset matching user's exact dashboard parameters
        np.random.seed(42)
        n = 7060
        df = pd.DataFrame({
            'Item_Identifier': [f'FD{np.random.randint(10,99)}' for _ in range(n)],
            'Item_Fat_Content': np.random.choice(['Low Fat', 'low fat', 'Regular', 'LF', 'reg'], n),
            'Item_Visibility': np.random.uniform(0.01, 0.2, n),
            'Item_Type': np.random.choice(['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy', 'Baking Goods'], n),
            'Outlet_Location_Type': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], n),
            'Outlet_Size': np.random.choice(['Small', 'Medium', 'High', np.nan], n, p=[0.4, 0.4, 0.1, 0.1]),
            'Outlet_Type': np.random.choice(['Grocery Store', 'Supermarket Type1', 'Supermarket Type2'], n),
            'Outlet_Establishment_Year': np.random.choice([2011, 2013, 2015, 2017, 2020, 2022], n),
            'Sales': np.random.exponential(scale=141, size=n)
        })
    
    # Standardize Messy Fat Content Labels
    fat_map = {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular', 'Regular': 'Regular', 'Low Fat': 'Low Fat'}
    df['Item_Fat_Content'] = df['Item_Fat_Content'].map(fat_map)
    
    # Advanced Data Imputation: Fill missing Outlet Sizes based on the Mode of their Outlet Type
    mode_size = df.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'Medium')
    df['Outlet_Size'] = df['Outlet_Size'].fillna(mode_size)
    
    # Calculate Outlet Age
    df['Outlet_Age'] = 2026 - df['Outlet_Establishment_Year']
    return df

df = load_and_clean_data()
st.dataframe(df)
# ----------------------------------------------------------------------------
# STEP 2: STATISTICAL DEEP DIVE - VISIBILITY PARADOX (Question 2)
# ----------------------------------------------------------------------------
st.header("📊 1. The Visibility Paradox Analysis")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    **Business Insight:** 
    Your SQL query showed that low-performing tiers have a higher visibility score ($0.0644$). 
    Let's calculate the exact mathematical correlation between Visibility and Sales across item types to see if it's negative.
    """)
    correlation = df['Item_Visibility'].corr(df['Sales'])
    st.metric(label="Overall Visibility-to-Sales Correlation", value=f"{correlation:.4f}", delta="Negative Impact" if correlation < 0 else "Positive Impact")

with col2:
    fig_scatter = px.scatter(
        df.sample(1000), x="Item_Visibility", y="Sales", color="Item_Type",
        trendline="ols", title="Item Visibility vs Sales (Sampled Trendlines)",
        labels={"Item_Visibility": "Platform Visibility Score", "Sales": "Gross Sales ($)"},
        template="plotly_dark"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------------------------------------------------------------
# STEP 3: PREDICTIVE MACHINE LEARNING PIPELINE (Question 3)
# ----------------------------------------------------------------------------
# Prepare data for ML
df_ml = df.copy()
cat_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    encoders[col] = le

X = df_ml[['Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Outlet_Age']]
y = df_ml['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------------
# STEP 4: INTERACTIVE LIVE APP PREDICTOR (Question 4)
# ----------------------------------------------------------------------------
st.header("🔮 2. Live Inventory Sales Simulator")
st.markdown("Change the parameters below to witness the Machine Learning model calculate expected item revenue live.")

row1, row1_2, row1_3, row1_4 = st.columns(4)
row2_1, row2_2, row2_3 = st.columns(3)

with row1:
    input_item_type = st.selectbox("Select Product Category", encoders['Item_Type'].classes_)
with row1_2:
    input_fat = st.selectbox("Fat Content Profile", encoders['Item_Fat_Content'].classes_)
with row1_3:
    input_store_type = st.selectbox("Target Store Type", encoders['Outlet_Type'].classes_)
with row1_4:
    input_location = st.selectbox("Demographic Area Location", encoders['Outlet_Location_Type'].classes_)

with row2_1:
    input_size = st.selectbox("Physical Store Size", encoders['Outlet_Size'].classes_)
with row2_2:
    input_age = st.slider("Store Age (Years operational)", min_value=1, max_value=30, value=10)
with row2_3:
    input_visibility = st.slider("Allocated Digital Visibility Score", min_value=0.00, max_value=0.25, value=0.06, step=0.01)

# Encode user input dynamically
input_data = pd.DataFrame([{
    'Item_Fat_Content': encoders['Item_Fat_Content'].transform([input_fat])[0],
    'Item_Visibility': input_visibility,
    'Item_Type': encoders['Item_Type'].transform([input_item_type])[0],
    'Outlet_Location_Type': encoders['Outlet_Location_Type'].transform([input_location])[0],
    'Outlet_Size': encoders['Outlet_Size'].transform([input_size])[0],
    'Outlet_Type': encoders['Outlet_Type'].transform([input_store_type])[0],
    'Outlet_Age': input_age
}])

# Make live calculation
predicted_sales = model.predict(input_data)[0]

st.markdown("---")
st.subheader("🎯 Live Machine Learning Projections")
st.metric(
    label=f"Predicted Revenue Contribution for {input_item_type}",
    value=f"${predicted_sales:,.2f} per transaction lifecycle",
    delta=f"Based on {input_visibility*100:.0f}% Screen Presence"
)
