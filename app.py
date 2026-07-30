import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date, timedelta

# --- PAGE SETUP ---
# --- PAGE SETUP ---
st.set_page_config(
    page_title="Decision Engine | Bake & Cook House", 
    page_icon="🧁", 
    layout="centered"
)

# Custom Header Branding
st.title("🍰 Decision Engine")
st.subheader("Bake & Cook House")
st.caption("Powered by **Verostat** | Demand Forecasting & Kitchen Prep System")

st.divider()
# --- LOAD TRAINED MODEL ARTIFACT ---
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'bakery_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, 'bakery_forecasting_model.pkl')
    return joblib.load(model_path)

try:
    artifact = load_model()
    model = artifact['model']
    feature_cols = artifact['feature_cols']
    threshold = artifact.get('threshold', 0.45)
except Exception as e:
    st.error(f"Error loading model file: {e}")
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
        # Initialize feature schema
        row = {col: 0 for col in feature_cols}
        
        # Populate features dynamically
        if 'Price' in row: row['Price'] = stats['Price']
        if 'Day_of_Week' in row: row['Day_of_Week'] = target_dt.dayofweek
        if 'Is_Weekend' in row: row['Is_Weekend'] = 1 if target_dt.dayofweek in [5, 6] else 0
        if 'Sales_Last_7_Days' in row: row['Sales_Last_7_Days'] = stats['Sales_Last_7_Days']
        if 'Same_Day_Last_Week' in row: row['Same_Day_Last_Week'] = stats['Same_Day_Last_Week']
        
        # One-Hot Encoding match
        for col in feature_cols:
            if col.startswith('Item_Name_'):
                clean_col_item = col.replace('Item_Name_', '').strip().lower()
                if clean_col_item == cake_name.strip().lower():
                    row[col] = 1
                
        rows.append((cake_name, row))
        
    input_df = pd.DataFrame([r[1] for r in rows])[feature_cols]
    raw_probabilities = model.predict_proba(input_df)[:, 1]
    
    results = []
    go_count = 0
    
    for (cake_name, _), p in zip(rows, raw_probabilities):
        stats = cake_inputs[cake_name]
        
        # 🛡️ OPERATIONAL GUARDRAIL: Dampen zero-demand items
        if stats['Sales_Last_7_Days'] == 0 and stats['Same_Day_Last_Week'] == 0:
            final_p = p * 0.35  # Apply 65% penalty for zero sales momentum
        else:
            final_p = p
            
        is_go = final_p >= threshold
        if is_go:
            go_count += 1
            
        results.append({
            'Cake Item': cake_name,
            'Price': f"₹{CAKE_MENU[cake_name]['Price']:.0f}",
            'Prob. Score (%)': f"{final_p*100:.1f}%",
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
