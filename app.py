import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta

# --- PAGE SETUP ---
st.set_page_config(page_title="Bakery Daily Prep Forecaster", page_icon="🍰", layout="centered")

st.title("🍰 Daily Bakery Prep Forecaster")
st.write("Select tomorrow's date and enter the last 7 days of sales to generate your baking schedule.")

# --- LOAD TRAINED MODEL ---
@st.cache_resource
def load_model():
    return joblib.load('bakery_model.pkl')

artifact = load_model()
model = artifact['model']
feature_cols = artifact['feature_cols']
threshold = artifact['threshold']

# --- CAKE CONFIGURATION ---
CAKE_MENU = {
    'Fresh Fruit Cake':       {'Price': 700.00},
    'Rasmalai Cake':          {'Price': 750.00},
    'Blueberry Cake':         {'Price': 510.00},
    'Black Forest Cake':      {'Price': 610.00},
    'Chocolate Kitkat Cake':  {'Price': 800.00}
}

# --- INPUT FORM ---
st.subheader("1. Date Selection")
target_date = st.date_input("Target Date for Forecast", value=date.today() + timedelta(days=1))

st.subheader("2. Recent Demand Signals")
st.caption("Enter the total units sold in the last 7 days for each cake:")

cake_inputs = {}
for cake_name, details in CAKE_MENU.items():
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.write(f"**{cake_name}** (₹{details['Price']:.0f})")
    with col2:
        sales_7d = st.number_input(f"Sales 7 Days", min_value=0, max_value=20, value=0, key=f"{cake_name}_7d")
    with col3:
        same_day = st.selectbox(f"Sold 7 Days Ago?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", key=f"{cake_name}_sd")
    
    cake_inputs[cake_name] = {
        'Price': details['Price'],
        'Sales_Last_7_Days': sales_7d,
        'Same_Day_Last_Week': same_day
    }

# --- PREDICTION LOGIC ---
if st.button("🚀 Generate Baking Schedule", type="primary"):
    target_dt = pd.to_datetime(target_date)
    rows = []
    
    for cake_name, stats in cake_inputs.items():
        row = {
            'Price': stats['Price'],
            'Day_of_Week': target_dt.dayofweek,
            'Is_Weekend': 1 if target_dt.dayofweek in [5, 6] else 0,
            'Is_Payday_Period': 1 if target_dt.day in [1, 2, 3, 4, 5, 28, 29, 30, 31] else 0,
            'Sales_Last_7_Days': stats['Sales_Last_7_Days'],
            'Same_Day_Last_Week': stats['Same_Day_Last_Week']
        }
        
        # Match One-Hot Encoded features
        for col in feature_cols:
            if col.startswith('Item_Name_'):
                clean_col_item = col.replace('Item_Name_', '').strip()
                row[col] = 1 if clean_col_item == cake_name else 0
                
        rows.append((cake_name, row))
        
    input_df = pd.DataFrame([r[1] for r in rows])[feature_cols]
    probabilities = model.predict_proba(input_df)[:, 1]
    
    # Format Results
    results = []
    for (cake_name, _), p in zip(rows, probabilities):
        is_prep = p >= threshold
        results.append({
            'Cake Item': cake_name,
            'Price': f"₹{CAKE_MENU[cake_name]['Price']:.0f}",
            'Demand Probability': f"{p*100:.1f}%",
            'Recommendation': "🟢 PREP / BAKE" if is_prep else "🔴 BAKE TO ORDER"
        })
        
    results_df = pd.DataFrame(results)
    
    st.divider()
    st.subheader(f"📋 Prep Plan for {target_date.strftime('%A, %b %d, %Y')}")
    
    # Custom colored metric output table
    st.dataframe(results_df, use_container_width=True, hide_index=True)