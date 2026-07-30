import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date, timedelta

# --- PAGE SETUP ---
st.set_page_config(page_title="Bakery Prep Decision Engine", page_icon="🍰", layout="centered")

st.title("🍰 Bakery Prep Decision Engine")
st.write("Enter target date and recent demand inputs to generate instant **Go / No-Go** baking recommendations.")

# --- LOAD TRAINED MODEL ARTIFACT (ROBUST PATH RESOLUTION) ---
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'bakery_model.pkl')
    return joblib.load(model_path)

try:
    artifact = load_model()
    model = artifact['model']
    feature_cols = artifact['feature_cols']
    threshold = artifact['threshold']
except Exception as e:
    st.error(f"Error loading model file ('bakery_model.pkl'): {e}")
    st.info("Ensure 'bakery_model.pkl' is uploaded directly to the root of your GitHub repository.")
    st.stop()

# --- CAKE MENU CONFIGURATION ---
CAKE_MENU = {
    'Fresh Fruit Cake':       {'Price': 700.00},
    'Rasmalai Cake':          {'Price': 750.00},
    'Blueberry Cake':         {'Price': 510.00},
    'Black Forest Cake':      {'Price': 610.00},
    'Chocolate Kitkat Cake':  {'Price': 800.00}
}

# --- INPUT FORM ---
st.subheader("1. Select Forecast Date")
target_date = st.date_input("Target Date", value=date.today() + timedelta(days=1))

st.subheader("2. Enter Recent Demand Inputs")
st.caption("Provide total units sold in the last 7 days and indicate if the item was sold exactly 7 days ago:")

cake_inputs = {}
for cake_name, details in CAKE_MENU.items():
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.write(f"**{cake_name}**  \n:gray[₹{details['Price']:.0f}]")
    with col2:
        sales_7d = st.number_input("Sales (Last 7 Days)", min_value=0, max_value=20, value=0, key=f"{cake_name}_7d")
    with col3:
        same_day = st.selectbox("Sold 7 Days Ago?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=f"{cake_name}_sd")
    
    cake_inputs[cake_name] = {
        'Price': details['Price'],
        'Sales_Last_7_Days': sales_7d,
        'Same_Day_Last_Week': same_day
    }

st.divider()

# --- PREDICTION ENGINE LOGIC ---
if st.button("🚀 Generate Baking Decision Schedule", type="primary", use_container_width=True):
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
        
        # Match One-Hot Encoded features dynamically
        for col in feature_cols:
            if col.startswith('Item_Name_'):
                clean_col_item = col.replace('Item_Name_', '').strip()
                row[col] = 1 if clean_col_item == cake_name else 0
                
        rows.append((cake_name, row))
        
    input_df = pd.DataFrame([r[1] for r in rows])[feature_cols]
    probabilities = model.predict_proba(input_df)[:, 1]
    
    # Format Results
    results = []
    go_count = 0
    
    for (cake_name, _), p in zip(rows, probabilities):
        is_go = p >= threshold
        if is_go:
            go_count += 1
            
        results.append({
            'Cake Item': cake_name,
            'Price': f"₹{CAKE_MENU[cake_name]['Price']:.0f}",
            'Prob. Score (%)': f"{p*100:.1f}%",
            'Go / No-Go Signal': "🟢 GO (Prep / Bake)" if is_go else "🔴 NO-GO (Bake to Order)"
        })
        
    results_df = pd.DataFrame(results)
    
    # Executive Summary Metrics
    st.subheader(f"📋 Prep Decision Schedule ({target_date.strftime('%A, %b %d, %Y')})")
    
    m1, m2 = st.columns(2)
    m1.metric("Total GO Signals (Prep Ahead)", f"{go_count} Cakes")
    m2.metric("Total NO-GO Signals (Hold)", f"{len(CAKE_MENU) - go_count} Cakes")
    
    # Output Display Table
    st.dataframe(
        results_df, 
        column_config={
            "Cake Item": st.column_config.TextColumn("Cake Item"),
            "Price": st.column_config.TextColumn("Price"),
            "Prob. Score (%)": st.column_config.TextColumn("Prob. Score (%)"),
            "Go / No-Go Signal": st.column_config.TextColumn("Go / No-Go Signal")
        },
        use_container_width=True, 
        hide_index=True
    )
