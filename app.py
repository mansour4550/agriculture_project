import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime, timedelta
import random
# Plotting Functions Section

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# Function to style Matplotlib plots consistently
def style_plot(fig, ax, title):
    fig.patch.set_facecolor('#3b5e2a')
    ax.set_facecolor('#3b5e2a')
    ax.set_title(title, color='#fffbdb', fontsize=14, pad=15)
    ax.tick_params(axis='both', colors='#fffbdb')
    ax.xaxis.label.set_color('#fffbdb')
    ax.yaxis.label.set_color('#fffbdb')
    for spine in ax.spines.values():
        spine.set_color('#93c249')
    ax.grid(True, color='#93c249', alpha=0.3)

# Plot for Top 3 Recommended Crops (Bar Plot with Matplotlib)
@st.cache_data
def plot_top_3_crops(crops, probabilities):
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(crops, probabilities, color='#93c249', edgecolor='#2a451c')
        ax.set_xlabel("Suitability Score (%)")
        ax.set_ylabel("Crop")
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                    ha='left', va='center', color='#fffbdb', x=width+1)
        style_plot(fig, ax, "Top 3 Recommended Crops")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting top 3 crops: {str(e)}")

# Plot for Sustainability Metrics (Bar Plot with Matplotlib)
@st.cache_data
def plot_sustainability_metrics(crop, sustainability_metrics):
    try:
        avg_metrics = {'water_usage': 1500, 'carbon_reduction': 0.5}
        labels = ['Water Usage (liters/kg)', 'Carbon Reduction (kg CO2/kg)']
        crop_values = [sustainability_metrics['water_usage'], sustainability_metrics['carbon_reduction']]
        avg_values = [avg_metrics['water_usage'], avg_metrics['carbon_reduction']]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width/2, crop_values, width, label=crop, color='#93c249', edgecolor='#2a451c')
        ax.bar(x + width/2, avg_values, width, label='Average', color='#fffbdb', edgecolor='#2a451c')

        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend()
        style_plot(fig, ax, f"Sustainability Metrics for {crop}")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting sustainability metrics: {str(e)}")

# Plot for IoT Data Comparison (Scatter Plot with Matplotlib)
@st.cache_data
def plot_iot_data_comparison(crop, iot_temp, iot_humidity, iot_ph, iot_rainfall, conditions):
    try:
        parameters = ['Temperature (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)']
        current_values = [iot_temp, iot_humidity, iot_ph, iot_rainfall]
        ideal_min = [conditions['temperature'][0], conditions['humidity'][0], conditions['ph'][0], conditions['rainfall'][0]]
        ideal_max = [conditions['temperature'][1], conditions['humidity'][1], conditions['ph'][1], conditions['rainfall'][1]]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(parameters))

        # Plot ideal ranges as shaded areas
        for i in range(len(parameters)):
            ax.fill_between([i-0.4, i+0.4], ideal_min[i], ideal_max[i], color='#93c249', alpha=0.2, label='Ideal Range' if i == 0 else "")

        # Plot current values as scatter points
        ax.scatter(x, current_values, color='#fffbdb', s=100, edgecolor='#2a451c', label='Current Values')
        for i, val in enumerate(current_values):
            ax.text(i, val + (ideal_max[i] - ideal_min[i])*0.05, f'{val:.1f}', ha='center', color='#fffbdb')

        ax.set_xticks(x)
        ax.set_xticklabels(parameters)
        ax.set_ylabel("Value")
        ax.legend()
        style_plot(fig, ax, f"IoT Data vs Ideal Conditions for {crop}")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting IoT data comparison: {str(e)}")

# Plot for Plant Growth Timeline (Step Plot with Matplotlib)
@st.cache_data
def plot_plant_growth_timeline(plant_name, current_stage):
    try:
        stages = PLANT_GROWTH_STAGES[plant_name.lower()]['stages']
        stage_names = [stage['name'] for stage in stages]
        stage_durations = [stage['duration_days'] for stage in stages]
        cumulative_days = np.cumsum([0] + stage_durations[:-1])

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.step(cumulative_days, range(len(stages)), where='post', color='#93c249', lw=2)
        ax.scatter(cumulative_days, range(len(stages)), color='#93c249', s=50)

        current_stage_idx = stage_names.index(current_stage)
        ax.axvline(x=sum(stage_durations[:current_stage_idx]), color='#fffbdb', linestyle='--', label='Current Stage')

        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels(stage_names)
        ax.set_xlabel("Days Since Planting")
        ax.legend()
        style_plot(fig, ax, f"Growth Timeline for {plant_name}")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting plant growth timeline: {str(e)}")

# Plot for Cumulative Carbon Reduction (Line Plot with Matplotlib)
@st.cache_data
def plot_cumulative_carbon_reduction(timestamps, cumulative_carbon):
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(timestamps, cumulative_carbon, color='#93c249', lw=2, marker='o', markersize=5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Carbon Reduction (kg CO2)")
        ax.tick_params(axis='x', rotation=45)
        style_plot(fig, ax, "Cumulative Carbon Footprint Reduction Over Time")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting cumulative carbon reduction: {str(e)}")

# Plot for Community Dashboard (Heatmap with Matplotlib)
@st.cache_data
def plot_community_dashboard(community_data):
    try:
        community_data_numeric = community_data[['Temperature', 'Humidity', 'pH', 'Rainfall']]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            community_data_numeric.T,
            cmap='YlGn',
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Value'},
            ax=ax
        )
        ax.set_xticklabels(community_data['Timestamp'], rotation=45, ha='right')
        style_plot(fig, ax, "Environmental Conditions in Recent Predictions")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting community dashboard: {str(e)}")

# Plot for Sustainability Leaderboard (Bar Plot with Matplotlib)
@st.cache_data
def plot_sustainability_leaderboard(leaderboard_data):
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            leaderboard_data['Crop'],
            leaderboard_data['Points'],
            color='#93c249',
            edgecolor='#2a451c'
        )
        ax.set_ylabel("Sustainability Points")
        ax.set_xlabel("Crop")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{int(yval)}', ha='center', color='#fffbdb')
        style_plot(fig, ax, "Top Contributors to Sustainability")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting sustainability leaderboard: {str(e)}")

# Function to train and save the model
@st.cache_data
def train_model(file_path):
    try:
        data = pd.read_csv(file_path)
        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])
        X = data.drop('label', axis=1)
        y = data['label']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        joblib.dump(model, 'crop_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        return model, scaler, label_encoder, X.columns, accuracy
    except FileNotFoundError:
        st.error(f"Dataset file not found at {file_path}. Please ensure the file exists.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('crop_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Function to save prediction to SQLite database
def save_prediction_to_db(inputs, crop, points):
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS predictions
                          (timestamp TEXT, N INTEGER, P INTEGER, K INTEGER, temperature FLOAT,
                           humidity FLOAT, ph FLOAT, rainfall FLOAT, crop TEXT, points INTEGER)''')
        timestamp = datetime.now().strftime("%Y-%m-d %H:%M:%S")
        cursor.execute('''INSERT INTO predictions (timestamp, N, P, K, temperature, humidity, ph, rainfall, crop, points)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (timestamp, inputs['N'], inputs['P'], inputs['K'], inputs['temperature'],
                        inputs['humidity'], inputs['ph'], inputs['rainfall'], crop, points))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error saving prediction to database: {str(e)}")

# Function to estimate water usage and carbon footprint reduction (simplified)
def estimate_sustainability_metrics(crop):
    sustainability_data = {
        'rice': {'water_usage': 2500, 'carbon_reduction': 0.5},
        'mango': {'water_usage': 1000, 'carbon_reduction': 1.2},
        'wheat': {'water_usage': 1500, 'carbon_reduction': 0.8},
        'maize': {'water_usage': 1200, 'carbon_reduction': 0.7},
        'potato': {'water_usage': 900, 'carbon_reduction': 0.6},
        'tomato': {'water_usage': 1100, 'carbon_reduction': 0.9}
    }
    return sustainability_data.get(crop.lower(), {'water_usage': 1500, 'carbon_reduction': 0.5})

# Function to calculate sustainability points for gamification
def calculate_sustainability_points(sustainability_metrics):
    water_usage = sustainability_metrics['water_usage']
    carbon_reduction = sustainability_metrics['carbon_reduction']
    points = (2000 - water_usage) / 100 + carbon_reduction * 10
    return max(0, int(points))

# Function to generate a PDF report
def generate_pdf_report(crop, inputs, sustainability_metrics, rotation_plan, points):
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, "Crop Recommendation Report")
        c.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-d %H:%M:%S')}")
        c.drawString(100, 700, f"Recommended Crop: {crop}")
        c.drawString(100, 680, "Input Parameters:")
        c.drawString(120, 660, f"Nitrogen (N): {inputs['N']}")
        c.drawString(120, 640, f"Phosphorus (P): {inputs['P']}")
        c.drawString(120, 620, f"Potassium (K): {inputs['K']}")
        c.drawString(120, 600, f"Temperature (°C): {inputs['temperature']}")
        c.drawString(120, 580, f"Humidity (%): {inputs['humidity']}")
        c.drawString(120, 560, f"pH: {inputs['ph']}")
        c.drawString(120, 540, f"Rainfall (mm): {inputs['rainfall']}")
        c.drawString(100, 510, "Sustainability Metrics:")
        c.drawString(120, 490, f"Estimated Water Usage: {sustainability_metrics['water_usage']} liters/kg")
        c.drawString(120, 470, f"Carbon Footprint Reduction: {sustainability_metrics['carbon_reduction']} kg CO2/kg")
        c.drawString(120, 450, f"Sustainability Points Earned: {points}")
        c.drawString(100, 420, "Crop Rotation Plan:")
        c.drawString(120, 400, f"Season 1 (Current): {crop}")
        c.drawString(120, 380, f"Season 2: {rotation_plan[0]}")
        c.drawString(120, 360, f"Season 3: {rotation_plan[1]}")
        c.drawString(100, 330, "Recommendation:")
        c.drawString(120, 310, f"Growing {crop} in urban settings can reduce carbon emissions")
        c.drawString(120, 290, "and optimize water usage, contributing to sustainable cities.")
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return None

# Crop rotation planner
crop_families = {
    'rice': 'Poaceae', 'mango': 'Anacardiaceae', 'wheat': 'Poaceae',
    'maize': 'Poaceae', 'potato': 'Solanaceae', 'tomato': 'Solanaceae'
}

def suggest_rotation_plan(crop):
    current_family = crop_families.get(crop.lower(), 'Unknown')
    rotation_plan = []
    available_crops = [c for c, f in crop_families.items() if f != current_family]
    rotation_plan.append(available_crops[0] if available_crops else 'Diverse Crop')
    rotation_plan.append(available_crops[1] if len(available_crops) > 1 else 'Diverse Crop')
    return rotation_plan

# Define growth stages and needs for each plant
PLANT_GROWTH_STAGES = {
    'rice': {
        'stages': [
            {'name': 'Germination', 'duration_days': 10, 'needs': {'water': '5-10 cm flooding', 'temperature': '20-35°C', 'sunlight': '6 hours', 'nutrients': 'Low NPK'}},
            {'name': 'Vegetative', 'duration_days': 60, 'needs': {'water': '5-10 cm flooding', 'temperature': '20-35°C', 'sunlight': '6-8 hours', 'nutrients': 'High N, Low P, Low K'}},
            {'name': 'Flowering', 'duration_days': 30, 'needs': {'water': '5 cm flooding', 'temperature': '20-30°C', 'sunlight': '6-8 hours', 'nutrients': 'Low N, High P, High K'}},
            {'name': 'Grain Filling', 'duration_days': 30, 'needs': {'water': '2-5 cm flooding', 'temperature': '20-30°C', 'sunlight': '6 hours', 'nutrients': 'Low N, High P, High K'}}
        ],
        'total_duration_days': 130
    },
    'mango': {
        'stages': [
            {'name': 'Seedling', 'duration_days': 365, 'needs': {'water': '1-2 liters every 2-3 days', 'temperature': '20-30°C', 'sunlight': '6-8 hours', 'nutrients': 'Balanced NPK (10-10-10)'}},
            {'name': 'Vegetative', 'duration_days': 730, 'needs': {'water': '2-3 liters every 2-3 days', 'temperature': '20-30°C', 'sunlight': '6-8 hours', 'nutrients': 'High N, Low P, Low K'}},
            {'name': 'Flowering', 'duration_days': 90, 'needs': {'water': '1-2 liters every 2 days', 'temperature': '20-25°C', 'sunlight': '6-8 hours', 'nutrients': 'Low N, High P, High K'}},
            {'name': 'Fruiting', 'duration_days': 90, 'needs': {'water': '2 liters every 2 days', 'temperature': '20-25°C', 'sunlight': '6-8 hours', 'nutrients': 'Low N, High P, High K'}}
        ],
        'total_duration_days': 1275
    },
    'wheat': {
        'stages': [
            {'name': 'Germination', 'duration_days': 7, 'needs': {'water': '0.5 liters daily', 'temperature': '15-25°C', 'sunlight': '6 hours', 'nutrients': 'Low NPK'}},
            {'name': 'Vegetative', 'duration_days': 60, 'needs': {'water': '0.5-1 liter every 2 days', 'temperature': '15-25°C', 'sunlight': '6-8 hours', 'nutrients': 'High N, Low P, Low K'}},
            {'name': 'Flowering', 'duration_days': 30, 'needs': {'water': '0.5 liters daily', 'temperature': '15-20°C', 'sunlight': '6-8 hours', 'nutrients': 'Low N, High P, High K'}},
            {'name': 'Grain Filling', 'duration_days': 30, 'needs': {'water': '0.3 liters daily', 'temperature': '15-20°C', 'sunlight': '6 hours', 'nutrients': 'Low N, High P, High K'}}
        ],
        'total_duration_days': 127
    },
    'maize': {
        'stages': [
            {'name': 'Germination', 'duration_days': 7, 'needs': {'water': '0.5 liters daily', 'temperature': '20-30°C', 'sunlight': '6 hours', 'nutrients': 'Low NPK'}},
            {'name': 'Vegetative', 'duration_days': 50, 'needs': {'water': '1 liter every 2 days', 'temperature': '20-30°C', 'sunlight': '6-8 hours', 'nutrients': 'High N, Low P, Low K'}},
            {'name': 'Flowering', 'duration_days': 30, 'needs': {'water': '1 liter daily', 'temperature': '20-25°C', 'sunlight': '6-8 hours', 'nutrients': 'Low N, High P, High K'}},
            {'name': 'Grain Filling', 'duration_days': 40, 'needs': {'water': '0.5 liters daily', 'temperature': '20-25°C', 'sunlight': '6 hours', 'nutrients': 'Low N, High P, High K'}}
        ],
        'total_duration_days': 127
    },
    'potato': {
        'stages': [
            {'name': 'Sprouting', 'duration_days': 14, 'needs': {'water': '0.5 liters every 2 days', 'temperature': '15-20°C', 'sunlight': '6 hours', 'nutrients': 'Low NPK'}},
            {'name': 'Vegetative', 'duration_days': 40, 'needs': {'water': '1 liter every 2 days', 'temperature': '15-20°C', 'sunlight': '6-8 hours', 'nutrients': 'High N, Low P, Low K'}},
            {'name': 'Tuber Formation', 'duration_days': 30, 'needs': {'water': '1 liter daily', 'temperature': '15-20°C', 'sunlight': '6-8 hours', 'nutrients': 'Low N, High P, High K'}},
            {'name': 'Maturation', 'duration_days': 20, 'needs': {'water': '0.5 liters every 2 days', 'temperature': '15-20°C', 'sunlight': '6 hours', 'nutrients': 'Low NPK'}}
        ],
        'total_duration_days': 104
    },
    'tomato': {
        'stages': [
            {'name': 'Germination', 'duration_days': 10, 'needs': {'water': '0.3 liters daily', 'temperature': '20-25°C', 'sunlight': '6 hours', 'nutrients': 'Low NPK'}},
            {'name': 'Vegetative', 'duration_days': 40, 'needs': {'water': '0.5 liters every 2 days', 'temperature': '20-25°C', 'sunlight': '6-8 hours', 'nutrients': 'High N, Low P, Low K'}},
            {'name': 'Flowering', 'duration_days': 20, 'needs': {'water': '0.5 liters daily', 'temperature': '20-25°C', 'sunlight': '6-8 hours', 'nutrients': 'Low N, High P, High K'}},
            {'name': 'Fruiting', 'duration_days': 30, 'needs': {'water': '1 liter daily', 'temperature': '20-25°C', 'sunlight': '6-8 hours', 'nutrients': 'Low N, High P, High K'}}
        ],
        'total_duration_days': 100
    }
}

# Function to save a plant to the tracking database
def save_plant_to_db(plant_name, planting_date):
    try:
        conn = sqlite3.connect('plant_tracker.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS plant_tracker
                          (id INTEGER PRIMARY KEY AUTOINCREMENT, plant_name TEXT, planting_date TEXT, current_stage TEXT)''')
        cursor.execute('''INSERT INTO plant_tracker (plant_name, planting_date, current_stage)
                          VALUES (?, ?, ?)''',
                       (plant_name, planting_date, 'Germination'))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error saving plant to database: {str(e)}")

# Function to get all tracked plants
def get_tracked_plants():
    try:
        conn = sqlite3.connect('plant_tracker.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, plant_name, planting_date, current_stage FROM plant_tracker")
        plants = cursor.fetchall()
        conn.close()
        return plants
    except Exception as e:
        st.error(f"Error retrieving plants from database: {str(e)}")
        return []

# Function to determine the current growth stage of a plant
def get_current_stage(plant_name, planting_date_str):
    try:
        planting_date = datetime.strptime(planting_date_str, "%Y-%m-d")
        days_since_planting = (datetime.now() - planting_date).days
        plant_data = PLANT_GROWTH_STAGES.get(plant_name.lower())
        if not plant_data:
            return "Unknown", None

        cumulative_days = 0
        for stage in plant_data['stages']:
            cumulative_days += stage['duration_days']
            if days_since_planting <= cumulative_days:
                return stage['name'], stage['needs']
        last_stage = plant_data['stages'][-1]
        return last_stage['name'], last_stage['needs']
    except Exception as e:
        st.error(f"Error calculating plant stage: {str(e)}")
        return "Unknown", None

# Function to update the current stage of a plant in the database
def update_plant_stage(plant_id, current_stage):
    try:
        conn = sqlite3.connect('plant_tracker.db')
        cursor = conn.cursor()
        cursor.execute("UPDATE plant_tracker SET current_stage = ? WHERE id = ?",
                       (current_stage, plant_id))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error updating plant stage: {str(e)}")

# Hardcoded responses for the chatbot
HARDCODED_RESPONSES = {
    "how do i grow mango in a rooftop garden": """
    Growing a mango tree in a rooftop garden is a fantastic idea for urban farming! It’s a bit of a long-term project, but with the right steps, you can enjoy fresh mangoes while adding greenery to your city space. Here’s a step-by-step guide to help you get started:

    1. **Choose the Right Mango Sapling**: Start with a grafted mango sapling, as these grow faster and produce fruit sooner—usually within 3-5 years—compared to growing from seed. You can find grafted saplings at a local nursery.

    2. **Pick a Large Container**: Mango trees need space for their roots, so use a large container that’s at least 50 cm deep and wide. Make sure it has drainage holes to prevent waterlogging, which can harm the tree.

    3. **Prepare the Soil**: Fill the container with well-drained soil that has a pH between 5.5 and 7.5. A good mix is potting soil, sand, and compost in equal parts. This ensures the soil is light, fertile, and drains well.

    4. **Plant the Sapling**: Dig a hole in the soil deep enough to cover the sapling’s root ball. Place the sapling in the hole, cover the roots with soil, and press down gently. Water it immediately to help the roots settle—about 1-2 liters of water should do.

    5. **Find a Sunny Spot**: Mango trees love sunlight, so place the container in a spot on your rooftop that gets 6-8 hours of direct sunlight daily. If your rooftop is windy, choose a corner that’s sheltered to protect the tree.

    6. **Water Regularly**: Keep the soil moist but not soggy. Water the tree every 2-3 days with about 1-2 liters of water, depending on the weather. In hotter months, you might need to water more often, but always check the soil first—if it’s still wet, wait a day.

    7. **Support the Tree**: Rooftops can be windy, so use stakes or supports to keep the tree stable. Tie the trunk loosely to the stakes with soft ties to prevent it from snapping in strong winds.

    8. **Fertilize Monthly**: Feed your mango tree with a balanced fertilizer, like a 10-10-10 NPK (nitrogen, phosphorus, potassium), once a month. Follow the package instructions, but typically, a small handful sprinkled around the base and watered in is enough.

    9. **Prune for Growth**: Once a year, prune any dead or damaged branches to encourage healthy growth. This also helps the tree maintain a manageable size for your rooftop.

    **Sustainable Tip**: To make your mango growing more eco-friendly, use drip irrigation to save water. You can set up a simple system with a small water tank and drip lines. Also, add a layer of organic mulch—like dried leaves or straw—around the base of the tree to retain soil moisture and reduce watering needs. This way, you’re growing mangoes sustainably while contributing to a greener city!
    """,
    "how do i grow rice in urban settings": """
    Growing rice in urban settings is a unique challenge, but it’s definitely possible with the right setup! Rice needs a lot of water and sunlight, so we’ll simulate a mini paddy field on your balcony or rooftop. Here’s a step-by-step guide:

    1. **Soak the Seeds**: Start by soaking rice seeds in water for 24 hours. This encourages germination and helps the seeds sprout faster. You can use any rice variety, but short-grain varieties are often easier for small-scale growing.

    2. **Choose a Watertight Container**: Rice grows best in flooded conditions, so use a large, watertight container like a plastic tub. It should be at least 20 cm deep to hold water and soil.

    3. **Prepare the Soil**: Fill the container with 10-15 cm of clay-rich soil or a mix of garden soil and compost. Rice prefers soil with a pH of 5.0 to 6.5, which retains water well.

    4. **Plant the Seeds**: After soaking, plant the germinated seeds 2-3 cm deep in the soil, spacing them about 15 cm apart. This gives each plant enough room to grow.

    5. **Flood the Container**: Add 5-10 cm of water to the container to create a flooded environment, similar to a paddy field. Maintain this water level throughout the growing season by topping up as needed.

    6. **Place in a Sunny Spot**: Rice needs at least 6 hours of direct sunlight daily, so place the container in a sunny spot on your balcony or rooftop. The ideal temperature is between 20°C and 35°C.

    7. **Monitor Water Levels**: Check the water level daily, especially in hot weather, as it can evaporate quickly. Keep the soil flooded but not overflowing—top up with about 1-2 liters of water as needed.

    8. **Harvest the Rice**: After 3-6 months, when the rice grains are hard and golden, it’s time to harvest. Drain the water, let the plants dry out for a few days, then cut the stalks and thresh the grains.

    **Sustainable Tip**: Use rainwater harvesting to supply water for your rice container. You can collect rainwater in a barrel and use it to flood the container. Also, recycle water from household use—like water used to rinse vegetables—to reduce waste. This makes your urban rice farming more sustainable!
    """,
    "advise me": """
    Urban farming is a great way to grow your own food, reduce your carbon footprint, and contribute to a sustainable city! Here are some general tips to get started:

    1. **Start Small**: If you’re new to urban farming, begin with easy crops like herbs (basil, mint) or leafy greens (lettuce, spinach). They grow quickly and don’t need much space.

    2. **Choose the Right Spot**: Find a spot on your balcony, rooftop, or even a sunny windowsill that gets at least 6 hours of sunlight daily. Most crops need plenty of light to thrive.

    3. **Use Containers**: In urban settings, space is limited, so use containers like pots, grow bags, or even recycled buckets. Make sure they have drainage holes to prevent waterlogging.

    4. **Focus on Soil Health**: Use a good potting mix with compost to provide nutrients for your plants. You can make compost from kitchen scraps like vegetable peels to enrich the soil naturally.

    5. **Water Wisely**: Overwatering can harm plants, so water only when the top inch of soil feels dry. Use a watering can or set up a simple drip irrigation system to save water.

    6. **Practice Crop Rotation**: To keep your soil healthy and prevent pests, rotate crops each season. For example, if you grow tomatoes this season, plant a different family like beans or wheat next season.

    7. **Use Sustainable Practices**: Harvest rainwater in a barrel to use for irrigation, and use natural pest control methods like neem oil or companion planting (e.g., plant marigolds to deter pests).

    If you have a specific crop in mind, let me know, and I can provide more detailed advice!
    """,
    "best crops for small spaces": """
    If you're working with small spaces, like a balcony or a tiny rooftop, you’ll want to choose crops that don’t need much room and can thrive in containers. Here are some of the best crops for small spaces in urban farming:

    1. **Herbs (Basil, Mint, Parsley)**: Herbs are perfect for small spaces because they grow well in small pots and don’t need much room. A pot as small as 15 cm in diameter can work. They also grow quickly—basil can be harvested in 4-6 weeks.

    2. **Leafy Greens (Lettuce, Spinach, Kale)**: These greens are compact and can be grown in shallow containers (20-30 cm deep). You can harvest them multiple times by cutting the outer leaves and letting the plant continue to grow.

    3. **Radishes**: Radishes are small, fast-growing root vegetables that mature in just 3-4 weeks. They don’t need much space—a container 15 cm deep is enough—and you can plant them close together.

    4. **Cherry Tomatoes**: These are smaller than regular tomatoes and can be grown in a pot or hanging basket. Use a container at least 30 cm deep, and provide a small trellis or stake for support. They’ll give you a steady supply of fruit with enough sunlight.

    5. **Microgreens**: Microgreens, like pea shoots or sunflower sprouts, are tiny, nutrient-packed plants that you can grow in trays. They’re ready to harvest in 1-2 weeks and only need a shallow container (5-10 cm deep).

    **Tips for Small Spaces**:
    - Use vertical space by stacking pots or using hanging planters to maximize your area.
    - Choose containers with good drainage to prevent root rot in small spaces.
    - Place your plants in a spot with at least 4-6 hours of sunlight daily.

    **Sustainable Tip**: Use a small compost bin to recycle kitchen scraps into fertilizer for your plants. This reduces waste and keeps your small garden thriving sustainably!
    """,
    "how to reduce water usage": """
    Reducing water usage in urban farming is key to making your garden sustainable, especially in a city where resources can be limited. Here are some practical steps to help you save water while keeping your plants healthy:

    1. **Use Drip Irrigation**: Set up a simple drip irrigation system with a small water tank and drip lines. This delivers water directly to the roots, reducing waste. You can even make a DIY version using a plastic bottle with tiny holes, buried near the plant’s base.

    2. **Mulch Your Plants**: Add a layer of organic mulch—like dried leaves, straw, or wood chips—around the base of your plants. This helps retain soil moisture, so you’ll need to water less often. Aim for a 5 cm layer of mulch.

    3. **Water at the Right Time**: Water your plants early in the morning or late in the evening when temperatures are cooler. This reduces evaporation, ensuring more water reaches the roots. Avoid watering in the middle of the day when the sun is strongest.

    4. **Harvest Rainwater**: Set up a small barrel or bucket on your balcony or rooftop to collect rainwater. Use this to water your plants instead of relying on tap water. Even a small container can collect enough for a few days of watering.

    5. **Choose Drought-Tolerant Crops**: Grow plants that need less water, like herbs (rosemary, thyme) or succulents. These are great for urban gardens and can survive with minimal watering—once every 4-5 days.

    6. **Check Soil Moisture**: Before watering, stick your finger 2-3 cm into the soil. If it feels moist, wait a day or two. Overwatering wastes water and can harm your plants.

    7. **Group Plants by Water Needs**: Place plants with similar water requirements together. For example, keep thirsty plants like rice separate from drought-tolerant ones like rosemary, so you don’t overwater some while underwatering others.

    **Sustainable Tip**: Reuse household water—like water used to rinse vegetables—to irrigate your plants. Just make sure it’s free of soap or chemicals. This reduces your overall water usage while keeping your garden green!
    """,
    "urban composting techniques": """
    Composting in an urban setting is a fantastic way to recycle kitchen waste into nutrient-rich fertilizer for your plants, even if you have limited space. Here are some urban composting techniques that work well in small spaces like balconies or apartments:

    1. **Vermicomposting (Worm Composting)**:
       - **What You Need**: A small bin (30 cm x 30 cm x 30 cm), red worms (like Eisenia fetida), and kitchen scraps (vegetable peels, coffee grounds, etc.).
       - **Steps**:
         1. Drill small holes in the sides of the bin for ventilation and the bottom for drainage. Place a tray underneath to catch any liquid.
         2. Add a layer of shredded newspaper or cardboard as bedding for the worms.
         3. Add your worms (about 500 grams for a small bin) and start adding kitchen scraps in small amounts.
         4. Keep the bin in a shaded spot and cover the scraps with more bedding to prevent odors.
         5. After 2-3 months, the worms will turn the scraps into compost. Harvest the compost and use the liquid (worm tea) as a natural fertilizer.
       - **Tip**: Avoid adding meat, dairy, or oily foods, as they can attract pests and create odors.

    2. **Bokashi Composting**:
       - **What You Need**: A small airtight bucket and Bokashi bran (a mix of microbes to ferment the waste).
       - **Steps**:
         1. Place a layer of kitchen scraps (including meat and dairy) in the bucket.
         2. Sprinkle a handful of Bokashi bran over the scraps to start fermentation.
         3. Press down the scraps to remove air, then seal the bucket tightly to create an anaerobic environment.
         4. Repeat the process, adding layers until the bucket is full.
         5. Let it ferment for 2 weeks, draining any liquid (Bokashi juice) every few days—this can be diluted and used as fertilizer.
         6. After 2 weeks, bury the fermented waste in soil or a larger compost bin to finish breaking down (another 2-4 weeks).
       - **Tip**: Bokashi is great for small spaces because it’s odorless when sealed properly.

    3. **DIY Tumbler Composting**:
       - **What You Need**: A small plastic container with a lid (like a 5-gallon bucket) and a mix of green (kitchen scraps) and brown (dry leaves, cardboard) materials.
       - **Steps**:
         1. Drill holes in the container for aeration.
         2. Add a mix of 1 part green materials (like vegetable peels) to 2 parts brown materials (like shredded paper).
         3. Seal the container and roll it gently every few days to mix the contents and speed up decomposition.
         4. Keep the mix moist but not soggy—add water if it’s too dry, or more brown materials if it’s too wet.
         5. In 4-6 weeks, you’ll have compost ready to use.
       - **Tip**: Place the tumbler in a shaded spot to avoid overheating, which can slow down decomposition.

    **General Tips for Urban Composting**:
    - Keep your compost bin in a shaded, well-ventilated area to prevent odors and overheating.
    - Chop scraps into smaller pieces to speed up decomposition.
    - If you’re worried about pests, use a bin with a tight lid and avoid adding meat or dairy (unless using Bokashi).

    **Sustainable Tip**: Use your compost to fertilize your urban garden, reducing the need for chemical fertilizers. This closes the loop, turning your waste into a resource for growing more food!
    """
}

# Function to get a chatbot response
def get_chatbot_response(question):
    try:
        if not question:
            return "Please ask a question about urban farming!"

        question = question.lower().strip()
        for key in HARDCODED_RESPONSES:
            if key in question:
                return f"**Answer:** {HARDCODED_RESPONSES[key]}"
        return """
        **Answer:** I’m not sure how to answer that specific question, but here’s some general advice for urban farming:

        Urban farming is a great way to grow your own food in the city! Start by choosing a sunny spot on your balcony or rooftop that gets at least 6 hours of sunlight daily. Use containers with good drainage, and fill them with a mix of potting soil and compost. Start with easy crops like herbs or leafy greens, and water them wisely—only when the soil feels dry. To be sustainable, use rainwater for irrigation and compost kitchen scraps to enrich your soil. If you have a specific crop in mind, try asking something like 'How do I grow mango in a rooftop garden?' for more detailed steps!
        """
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit app
def main():
    st.set_page_config(page_title="AgriTunisia SmartCrop: AI-Powered Urban Farming", layout="wide")
    
    # Title section with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image('C:\\Users\\HP\\Desktop\\trc 2.0\\new_venv\\agrinexa_logo.png.png', width=100)  # Ensure the logo file is in the correct directory
    with col2:
        st.markdown("###  AgriTunisia SmartCrop: AI-Powered Urban Farming")
    
    st.markdown("Optimize urban farming for sustainable cities by finding the best crops to grow! Powered by AI technology.")

    # Custom CSS for sliders
    st.markdown(
    """
    <style>
    /* Slider track */
    .stSlider [type="range"] {
        -webkit-appearance: none;
        width: 100%;
        height: 8px;
        background: #3b5e2a;
        border-radius: 5px;
        outline: none;
    }

    /* Slider thumb (handle) */
    .stSlider [type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 16px;
        height: 16px;
        background: #fffbdb;
        border-radius: 50%;
        cursor: pointer;
        border: 1px solid #3b5e2a;
    }
    
    .stSlider [type="range"]::-moz-range-thumb {
        width: 16px;
        height: 16px;
        background: #fffbdb;
        border-radius: 50%;
        cursor: pointer;
        border: 1px solid #3b5e2a;
    }

    /* Slider progress (filled portion) */
    .stSlider [type="range"]::-webkit-slider-runnable-track {
        height: 8px;
        background: #3b5e2a;
        border-radius: 5px;
    }
    
    .stSlider [type="range"]::-moz-range-progress {
        height: 8px;
        background: #fffbdb;
        border-radius: 5px;
    }

    /* Active state for the thumb */
    .stSlider [type="range"]::-webkit-slider-thumb:active {
        background: #fffbdb;
        box-shadow: 0 0 5px #fffbdb;
    }
    </style>
    """,
        unsafe_allow_html=True
    )

    # File path for the dataset
    DATA_PATH = r"C:\Users\HP\Desktop\trc 2.0\new_venv\Crop_recommendation.csv"
    
    # Load or train model
    if os.path.exists('crop_model.pkl') and os.path.exists('scaler.pkl') and os.path.exists('label_encoder.pkl'):
        model, scaler, label_encoder = load_model()
        if model is None:
            return
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        accuracy = None
    else:
        st.warning("Model not found. Training a new model...")
        model, scaler, label_encoder, feature_names, accuracy = train_model(DATA_PATH)
        if model is None:
            return
        st.success(f"Model trained with accuracy: {accuracy:.4f}")

    # Create two columns: left for main content, right for chatbot
    col1, col2 = st.columns([3, 1])

    # Left column: Main content
    with col1:
        # Sidebar for inputs
        st.sidebar.header("Input Parameters")
        with st.sidebar:
            n = st.slider("Nitrogen (N)", 0, 140, 50, help="Nitrogen content in soil (0-140)")
            p = st.slider("Phosphorus (P)", 5, 145, 50, help="Phosphorus content in soil (5-145)")
            k = st.slider("Potassium (K)", 5, 205, 50, help="Potassium content in soil (5-205)")
            st.markdown("**Temperature (°C)**")
            temp_min, temp_max = 8.0, 43.0
            temp = st.slider("", min_value=temp_min, max_value=temp_max, value=25.0, step=0.1, key="temperature_slider", help="Temperature in Celsius (8-43°C)")
            st.markdown(f'<div class="current-value">{temp:.2f}</div>', unsafe_allow_html=True)
            col1_inner, col2_inner, col3_inner = st.columns([1, 3, 1])
            with col1_inner:
                st.markdown(f'<div class="min-max-label">{temp_min}</div>', unsafe_allow_html=True)
            with col2_inner:
                pass
            with col3_inner:
                st.markdown(f'<div class="min-max-label">{temp_max}</div>', unsafe_allow_html=True)
            humidity = st.slider("Humidity (%)", 14.0, 99.0, 50.0, step=0.1, help="Humidity percentage (14-99%)")
            ph = st.slider("pH", 3.5, 9.9, 6.5, step=0.1, help="Soil pH level (3.5-9.9)")
            rainfall = st.slider("Rainfall (mm)", 20.0, 298.0, 100.0, step=0.1, help="Rainfall in millimeters (20-298 mm)")

        # Prepare input data
        inputs = {
            'N': n, 'P': p, 'K': k, 'temperature': temp,
            'humidity': humidity, 'ph': ph, 'rainfall': rainfall
        }
        input_data = pd.DataFrame(
            [[n, p, k, temp, humidity, ph, rainfall]],
            columns=feature_names
        )
        input_scaled = scaler.transform(input_data)

        # Prediction
        if st.button("Predict Crop"):
            try:
                prediction_probs = model.predict_proba(input_scaled)[0]
                top_3_indices = np.argsort(prediction_probs)[-3:][::-1]
                top_3_crops = label_encoder.inverse_transform(top_3_indices)
                top_3_probabilities = prediction_probs[top_3_indices] * 100

                st.subheader("Top 3 Recommended Crops")
                for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_probabilities), 1):
                    st.write(f"**{i}. {crop}** - Suitability Score: {prob:.2f}%")
                
                # Add plot for top 3 crops
                plot_top_3_crops(top_3_crops, top_3_probabilities)

                prediction = model.predict(input_scaled)
                crop = label_encoder.inverse_transform(prediction)[0]
                st.success(f"Selected Crop for Detailed Recommendations: **{crop}**")
                st.write("This recommendation optimizes resource use for sustainable urban farming!")

                st.session_state['predicted_crop'] = crop

                sustainability_metrics = estimate_sustainability_metrics(crop)
                points = calculate_sustainability_points(sustainability_metrics)
                st.subheader("Sustainability Impact")
                st.write(f"**Estimated Water Usage:** {sustainability_metrics['water_usage']} liters/kg")
                st.write(f"**Carbon Footprint Reduction:** {sustainability_metrics['carbon_reduction']} kg CO2/kg")
                st.write(f"**Sustainability Points Earned:** {points}")

                # Add plot for sustainability metrics
                plot_sustainability_metrics(crop, sustainability_metrics)

                rotation_plan = suggest_rotation_plan(crop)
                st.subheader("Crop Rotation Plan for Sustainability")
                st.write(f"**Season 1 (Current):** {crop}")
                st.write(f"**Season 2:** {rotation_plan[0]}")
                st.write(f"**Season 3:** {rotation_plan[1]}")

                st.subheader("Growing Instructions for Urban Farming")
                growing_instructions = {
                    'rice': """
                    **How to Grow Rice in Urban Settings**:
                    - **Container**: Use a watertight container (at least 20 cm deep) to simulate a paddy field.
                    - **Soil**: Use clay-rich soil or a mix of garden soil and compost (pH 5.0-6.5).
                    - **Watering**: Maintain 5-10 cm of water above the soil throughout the growing season. Top up daily (1-2 liters as needed).
                    - **Sunlight**: Needs at least 6 hours of direct sunlight daily.
                    - **Temperature**: Ideal range is 20-35°C.
                    - **Challenges**: Watch for pests like rice weevils. Use neem oil as a natural pest repellent.
                    - **Tip**: Use rainwater harvesting to supply water sustainably.
                    """,
                    'mango': """
                    **How to Grow Mango in a Rooftop Garden**:
                    - **Container**: Use a large container (at least 50 cm deep and wide) with drainage holes.
                    - **Soil**: Well-drained soil with pH 5.5-7.5 (mix potting soil, sand, and compost).
                    - **Watering**: Water every 2-3 days with 1-2 liters, keeping soil moist but not soggy.
                    - **Sunlight**: Requires 6-8 hours of direct sunlight daily.
                    - **Temperature**: Ideal range is 20-30°C.
                    - **Challenges**: Protect from strong winds on rooftops by using stakes. Watch for anthracnose (fungal disease); use organic fungicides if needed.
                    - **Tip**: Use drip irrigation to save water and mulch to retain moisture.
                    """,
                    'wheat': """
                    **How to Grow Wheat in Urban Settings**:
                    - **Container**: Use a container at least 15 cm deep with good drainage.
                    - **Soil**: Loamy soil with pH 6.0-7.0, enriched with compost.
                    - **Watering**: Water every 2 days with 0.5-1 liter, keeping soil moist but not waterlogged.
                    - **Sunlight**: Needs 6-8 hours of direct sunlight daily.
                    - **Temperature**: Ideal range is 15-25°C.
                    - **Challenges**: Susceptible to rust (fungal disease). Ensure good air circulation and avoid overwatering.
                    - **Tip**: Rotate with legumes like beans to improve soil health.
                    """,
                    'maize': """
                    **How to Grow Maize in Urban Settings**:
                    - **Container**: Use a container at least 30 cm deep with drainage holes.
                    - **Soil**: Well-drained, fertile soil with pH 5.8-7.0.
                    - **Watering**: Water every 2 days with 1 liter during vegetative stage, then daily during flowering.
                    - **Sunlight**: Requires 6-8 hours of direct sunlight daily.
                    - **Temperature**: Ideal range is 20-30°C.
                    - **Challenges**: Watch for corn earworms. Use organic pest control like neem oil.
                    - **Tip**: Plant in blocks (not rows) to improve pollination in small spaces.
                    """,
                    'potato': """
                    **How to Grow Potato in Urban Settings**:
                    - **Container**: Use a deep container or grow bag (at least 40 cm deep) with drainage holes.
                    - **Soil**: Loose, well-drained soil with pH 5.0-6.0, mixed with compost.
                    - **Watering**: Water every 2 days with 0.5-1 liter, increasing to daily during tuber formation.
                    - **Sunlight**: Needs 6-8 hours of sunlight daily.
                    - **Temperature**: Ideal range is 15-20°C.
                    - **Challenges**: Watch for potato blight. Ensure good air circulation and avoid wet foliage.
                    - **Tip**: Use a hilling technique (pile soil around the base) to encourage more tuber growth.
                    """,
                    'tomato': """
                    **How to Grow Tomato in Urban Settings**:
                    - **Container**: Use a pot at least 30 cm deep with drainage holes.
                    - **Soil**: Well-drained, fertile soil with pH 6.0-6.8, enriched with compost.
                    - **Watering**: Water every 2 days with 0.5 liters during vegetative stage, then daily during fruiting.
                    - **Sunlight**: Requires 6-8 hours of direct sunlight daily.
                    - **Temperature**: Ideal range is 20-25°C.
                    - **Challenges**: Watch for blossom-end rot (due to calcium deficiency). Add crushed eggshells to the soil.
                    - **Tip**: Use a stake or cage to support the plant as it grows.
                    """
                }
                st.markdown(growing_instructions.get(crop.lower(), "No detailed growing instructions available for this crop."))

                st.subheader("Personalized Recommendations")
                ideal_conditions = {
                    'rice': {'temperature': (20, 35), 'humidity': (70, 90), 'ph': (5.0, 6.5), 'rainfall': (150, 300)},
                    'mango': {'temperature': (20, 30), 'humidity': (50, 70), 'ph': (5.5, 7.5), 'rainfall': (50, 150)},
                    'wheat': {'temperature': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.0), 'rainfall': (50, 100)},
                    'maize': {'temperature': (20, 30), 'humidity': (60, 80), 'ph': (5.8, 7.0), 'rainfall': (50, 150)},
                    'potato': {'temperature': (15, 20), 'humidity': (60, 80), 'ph': (5.0, 6.0), 'rainfall': (50, 100)},
                    'tomato': {'temperature': (20, 25), 'humidity': (60, 80), 'ph': (6.0, 6.8), 'rainfall': (50, 150)}
                }
                conditions = ideal_conditions.get(crop.lower(), {'temperature': (0, 100), 'humidity': (0, 100), 'ph': (0, 14), 'rainfall': (0, 500)})
                
                warnings = []
                if 'iot_temp' in st.session_state:
                    if st.session_state['iot_temp'] < conditions['temperature'][0]:
                        warnings.append(f"Warning: Temperature ({st.session_state['iot_temp']}°C) is too low for {crop}. Ideal range is {conditions['temperature'][0]}-{conditions['temperature'][1]}°C. Increase temperature by {conditions['temperature'][0] - st.session_state['iot_temp']:.1f}°C.")
                    elif st.session_state['iot_temp'] > conditions['temperature'][1]:
                        warnings.append(f"Warning: Temperature ({st.session_state['iot_temp']}°C) is too high for {crop}. Ideal range is {conditions['temperature'][0]}-{conditions['temperature'][1]}°C. Decrease temperature by {st.session_state['iot_temp'] - conditions['temperature'][1]:.1f}°C.")
                
                if 'iot_humidity' in st.session_state:
                    if st.session_state['iot_humidity'] < conditions['humidity'][0]:
                        warnings.append(f"Warning: Humidity ({st.session_state['iot_humidity']}%) is too low for {crop}. Ideal range is {conditions['humidity'][0]}-{conditions['humidity'][1]}%. Increase humidity.")
                    elif st.session_state['iot_humidity'] > conditions['humidity'][1]:
                        warnings.append(f"Warning: Humidity ({st.session_state['iot_humidity']}%) is too high for {crop}. Ideal range is {conditions['humidity'][0]}-{conditions['humidity'][1]}%. Decrease humidity.")
                
                if 'iot_ph' in st.session_state:
                    if st.session_state['iot_ph'] < conditions['ph'][0]:
                        warnings.append(f"Warning: pH ({st.session_state['iot_ph']}) is too low for {crop}. Ideal range is {conditions['ph'][0]}-{conditions['ph'][1]}. Add lime to increase pH.")
                    elif st.session_state['iot_ph'] > conditions['ph'][1]:
                        warnings.append(f"Warning: pH ({st.session_state['iot_ph']}) is too high for {crop}. Ideal range is {conditions['ph'][0]}-{conditions['ph'][1]}. Add sulfur to decrease pH.")
                
                if 'iot_rainfall' in st.session_state:
                    if st.session_state['iot_rainfall'] < conditions['rainfall'][0]:
                        warnings.append(f"Warning: Rainfall ({st.session_state['iot_rainfall']} mm) is too low for {crop}. Ideal range is {conditions['rainfall'][0]}-{conditions['rainfall'][1]} mm. Supplement with irrigation.")
                    elif st.session_state['iot_rainfall'] > conditions['rainfall'][1]:
                        warnings.append(f"Warning: Rainfall ({st.session_state['iot_rainfall']} mm) is too high for {crop}. Ideal range is {conditions['rainfall'][0]}-{conditions['rainfall'][1]} mm. Ensure proper drainage.")
                
                if warnings:
                    for warning in warnings:
                        st.warning(warning)
                else:
                    st.success(f"Current conditions are ideal for growing {crop}!")

                st.subheader("Add to Plant Growth Tracker")
                if st.button("Add Predicted Crop to Tracker"):
                    planting_date_str = datetime.now().strftime("%Y-%m-d")
                    save_plant_to_db(crop, planting_date_str)
                    st.success(f"Added {crop} to the Plant Growth Tracker! Planted on {planting_date_str}.")

                save_prediction_to_db(inputs, crop, points)

                pdf_buffer = generate_pdf_report(crop, inputs, sustainability_metrics, rotation_plan, points)
                if pdf_buffer:
                    st.download_button(
                        label="Download Report",
                        data=pdf_buffer,
                        file_name=f"crop_recommendation_{crop}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

        # IoT Simulation with Sliders
        st.subheader("IoT Simulation for Real-Time Data")
        st.write("Adjust the sliders to simulate IoT sensor data:")

        st.markdown("**Temperature (°C)**")
        iot_temp_min, iot_temp_max = 15.0, 35.0
        iot_temp = st.slider("", min_value=iot_temp_min, max_value=iot_temp_max, value=25.0, step=0.1, key="iot_temperature_slider", help="Simulated temperature in Celsius (15-35°C)")
        st.markdown(f'<div class="current-value">{iot_temp:.2f}</div>', unsafe_allow_html=True)
        col1_inner, col2_inner, col3_inner = st.columns([1, 3, 1])
        with col1_inner:
            st.markdown(f'<div class="min-max-label">{iot_temp_min}</div>', unsafe_allow_html=True)
        with col2_inner:
            pass
        with col3_inner:
            st.markdown(f'<div class="min-max-label">{iot_temp_max}</div>', unsafe_allow_html=True)

        st.markdown("**Humidity (%)**")
        iot_humidity_min, iot_humidity_max = 20.0, 90.0
        iot_humidity = st.slider("", min_value=iot_humidity_min, max_value=iot_humidity_max, value=55.0, step=0.1, key="iot_humidity_slider", help="Simulated humidity percentage (20-90%)")
        st.markdown(f'<div class="current-value">{iot_humidity:.2f}</div>', unsafe_allow_html=True)
        col1_inner, col2_inner, col3_inner = st.columns([1, 3, 1])
        with col1_inner:
            st.markdown(f'<div class="min-max-label">{iot_humidity_min}</div>', unsafe_allow_html=True)
        with col2_inner:
            pass
        with col3_inner:
            st.markdown(f'<div class="min-max-label">{iot_humidity_max}</div>', unsafe_allow_html=True)

        st.markdown("**pH**")
        iot_ph_min, iot_ph_max = 5.0, 8.0
        iot_ph = st.slider("", min_value=iot_ph_min, max_value=iot_ph_max, value=6.5, step=0.1, key="iot_ph_slider", help="Simulated soil pH level (5-8)")
        st.markdown(f'<div class="current-value">{iot_ph:.2f}</div>', unsafe_allow_html=True)
        col1_inner, col2_inner, col3_inner = st.columns([1, 3, 1])
        with col1_inner:
            st.markdown(f'<div class="min-max-label">{iot_ph_min}</div>', unsafe_allow_html=True)
        with col2_inner:
            pass
        with col3_inner:
            st.markdown(f'<div class="min-max-label">{iot_ph_max}</div>', unsafe_allow_html=True)

        st.markdown("**Rainfall (mm)**")
        iot_rainfall_min, iot_rainfall_max = 50.0, 200.0
        iot_rainfall = st.slider("", min_value=iot_rainfall_min, max_value=iot_rainfall_max, value=125.0, step=0.1, key="iot_rainfall_slider", help="Simulated rainfall in millimeters (50-200 mm)")
        st.markdown(f'<div class="current-value">{iot_rainfall:.2f}</div>', unsafe_allow_html=True)
        col1_inner, col2_inner, col3_inner = st.columns([1, 3, 1])
        with col1_inner:
            st.markdown(f'<div class="min-max-label">{iot_rainfall_min}</div>', unsafe_allow_html=True)
        with col2_inner:
            pass
        with col3_inner:
            st.markdown(f'<div class="min-max-label">{iot_rainfall_max}</div>', unsafe_allow_html=True)

        if st.button("Predict with IoT Data"):
            try:
                iot_data = {
                    'temperature': iot_temp,
                    'humidity': iot_humidity,
                    'ph': iot_ph,
                    'rainfall': iot_rainfall
                }
                inputs.update(iot_data)
                input_data = pd.DataFrame(
                    [[inputs['N'], inputs['P'], inputs['K'], inputs['temperature'],
                      inputs['humidity'], inputs['ph'], inputs['rainfall']]],
                    columns=feature_names
                )
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                crop = label_encoder.inverse_transform(prediction)[0]
                st.write(f"**Recommended Crop based on IoT Data:** {crop}")
                sustainability_metrics = estimate_sustainability_metrics(crop)
                points = calculate_sustainability_points(sustainability_metrics)
                st.write(f"**Carbon Footprint Reduction:** {sustainability_metrics['carbon_reduction']} kg CO2/kg")
                save_prediction_to_db(inputs, crop, points)

                # Add plot for IoT data comparison
                ideal_conditions = {
                    'rice': {'temperature': (20, 35), 'humidity': (70, 90), 'ph': (5.0, 6.5), 'rainfall': (150, 300)},
                    'mango': {'temperature': (20, 30), 'humidity': (50, 70), 'ph': (5.5, 7.5), 'rainfall': (50, 150)},
                    'wheat': {'temperature': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.0), 'rainfall': (50, 100)},
                    'maize': {'temperature': (20, 30), 'humidity': (60, 80), 'ph': (5.8, 7.0), 'rainfall': (50, 150)},
                    'potato': {'temperature': (15, 20), 'humidity': (60, 80), 'ph': (5.0, 6.0), 'rainfall': (50, 100)},
                    'tomato': {'temperature': (20, 25), 'humidity': (60, 80), 'ph': (6.0, 6.8), 'rainfall': (50, 150)}
                }
                conditions = ideal_conditions.get(crop.lower(), {'temperature': (0, 100), 'humidity': (0, 100), 'ph': (0, 14), 'rainfall': (0, 500)})
                plot_iot_data_comparison(crop, iot_temp, iot_humidity, iot_ph, iot_rainfall, conditions)
            except Exception as e:
                st.error(f"Error during IoT prediction: {str(e)}")

        # IoT Simulator for Predicted Crop
        if 'predicted_crop' in st.session_state:
            crop = st.session_state['predicted_crop']
            st.subheader(f"IoT Simulator for Predicted Crop: {crop}")
            st.write(f"Simulate real-time conditions for {crop} and get immediate feedback:")

            st.markdown("**Temperature (°C)**")
            pred_iot_temp_min, pred_iot_temp_max = 10.0, 40.0
            pred_iot_temp = st.slider("", min_value=pred_iot_temp_min, max_value=pred_iot_temp_max, value=25.0, step=0.1, key="pred_iot_temperature_slider", help="Simulated temperature in Celsius for the predicted crop")
            st.markdown(f'<div class="current-value">{pred_iot_temp:.2f}</div>', unsafe_allow_html=True)
            col1_inner, col2_inner, col3_inner = st.columns([1, 3, 1])
            with col1_inner:
                st.markdown(f'<div class="min-max-label">{pred_iot_temp_min}</div>', unsafe_allow_html=True)
            with col2_inner:
                pass
            with col3_inner:
                st.markdown(f'<div class="min-max-label">{pred_iot_temp_max}</div>', unsafe_allow_html=True)

            st.markdown("**Humidity (%)**")
            pred_iot_humidity_min, pred_iot_humidity_max = 20.0, 90.0
            pred_iot_humidity = st.slider("", min_value=pred_iot_humidity_min, max_value=pred_iot_humidity_max, value=55.0, step=0.1, key="pred_iot_humidity_slider", help="Simulated humidity percentage for the predicted crop")
            st.markdown(f'<div class="current-value">{pred_iot_humidity:.2f}</div>', unsafe_allow_html=True)
            col1_inner, col2_inner, col3_inner = st.columns([1, 3, 1])
            with col1_inner:
                st.markdown(f'<div class="min-max-label">{pred_iot_humidity_min}</div>', unsafe_allow_html=True)
            with col2_inner:
                pass
            with col3_inner:
                st.markdown(f'<div class="min-max-label">{pred_iot_humidity_max}</div>', unsafe_allow_html=True)

            st.markdown("**pH**")
            pred_iot_ph_min, pred_iot_ph_max = 4.0, 9.0
            pred_iot_ph = st.slider("", min_value=pred_iot_ph_min, max_value=pred_iot_ph_max, value=6.5, step=0.1, key="pred_iot_ph_slider", help="Simulated soil pH level for the predicted crop")
            st.markdown(f'<div class="current-value">{pred_iot_ph:.2f}</div>', unsafe_allow_html=True)
            col1_inner, col2_inner, col3_inner = st.columns([1, 3, 1])
            with col1_inner:
                st.markdown(f'<div class="min-max-label">{pred_iot_ph_min}</div>', unsafe_allow_html=True)
            with col2_inner:
                pass
            with col3_inner:
                st.markdown(f'<div class="min-max-label">{pred_iot_ph_max}</div>', unsafe_allow_html=True)

            st.markdown("**Rainfall (mm)**")
            pred_iot_rainfall_min, pred_iot_rainfall_max = 20.0, 300.0
            pred_iot_rainfall = st.slider("", min_value=pred_iot_rainfall_min, max_value=pred_iot_rainfall_max, value=125.0, step=0.1, key="pred_iot_rainfall_slider", help="Simulated rainfall in millimeters for the predicted crop")
            st.markdown(f'<div class="current-value">{pred_iot_rainfall:.2f}</div>', unsafe_allow_html=True)
            col1_inner, col2_inner, col3_inner = st.columns([1, 3, 1])
            with col1_inner:
                st.markdown(f'<div class="min-max-label">{pred_iot_rainfall_min}</div>', unsafe_allow_html=True)
            with col2_inner:
                pass
            with col3_inner:
                st.markdown(f'<div class="min-max-label">{pred_iot_rainfall_max}</div>', unsafe_allow_html=True)

            st.session_state['iot_temp'] = pred_iot_temp
            st.session_state['iot_humidity'] = pred_iot_humidity
            st.session_state['iot_ph'] = pred_iot_ph
            st.session_state['iot_rainfall'] = pred_iot_rainfall

            ideal_conditions = {
                'rice': {'temperature': (20, 35), 'humidity': (70, 90), 'ph': (5.0, 6.5), 'rainfall': (150, 300)},
                'mango': {'temperature': (20, 30), 'humidity': (50, 70), 'ph': (5.5, 7.5), 'rainfall': (50, 150)},
                'wheat': {'temperature': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.0), 'rainfall': (50, 100)},
                'maize': {'temperature': (20, 30), 'humidity': (60, 80), 'ph': (5.8, 7.0), 'rainfall': (50, 150)},
                'potato': {'temperature': (15, 20), 'humidity': (60, 80), 'ph': (5.0, 6.0), 'rainfall': (50, 100)},
                'tomato': {'temperature': (20, 25), 'humidity': (60, 80), 'ph': (6.0, 6.8), 'rainfall': (50, 150)}
            }
            conditions = ideal_conditions.get(crop.lower(), {'temperature': (0, 100), 'humidity': (0, 100), 'ph': (0, 14), 'rainfall': (0, 500)})
            
            # Add plot for IoT data comparison for predicted crop
            plot_iot_data_comparison(crop, pred_iot_temp, pred_iot_humidity, pred_iot_ph, pred_iot_rainfall, conditions)

            feedback = []
            if pred_iot_temp < conditions['temperature'][0]:
                feedback.append(f"Temperature ({pred_iot_temp}°C) is too low. Ideal range is {conditions['temperature'][0]}-{conditions['temperature'][1]}°C.")
            elif pred_iot_temp > conditions['temperature'][1]:
                feedback.append(f"Temperature ({pred_iot_temp}°C) is too high. Ideal range is {conditions['temperature'][0]}-{conditions['temperature'][1]}°C.")
            
            if pred_iot_humidity < conditions['humidity'][0]:
                feedback.append(f"Humidity ({pred_iot_humidity}%) is too low. Ideal range is {conditions['humidity'][0]}-{conditions['humidity'][1]}%.")
            elif pred_iot_humidity > conditions['humidity'][1]:
                feedback.append(f"Humidity ({pred_iot_humidity}%) is too high. Ideal range is {conditions['humidity'][0]}-{conditions['humidity'][1]}%.")
            
            if pred_iot_ph < conditions['ph'][0]:
                feedback.append(f"pH ({pred_iot_ph}) is too low. Ideal range is {conditions['ph'][0]}-{conditions['ph'][1]}.")
            elif pred_iot_ph > conditions['ph'][1]:
                feedback.append(f"pH ({pred_iot_ph}) is too high. Ideal range is {conditions['ph'][0]}-{conditions['ph'][1]}.")
            
            if pred_iot_rainfall < conditions['rainfall'][0]:
                feedback.append(f"Rainfall ({pred_iot_rainfall} mm) is too low. Ideal range is {conditions['rainfall'][0]}-{conditions['rainfall'][1]} mm.")
            elif pred_iot_rainfall > conditions['rainfall'][1]:
                feedback.append(f"Rainfall ({pred_iot_rainfall} mm) is too high. Ideal range is {conditions['rainfall'][0]}-{conditions['rainfall'][1]} mm.")
            
            if feedback:
                st.subheader("Feedback on Simulated Conditions")
                for msg in feedback:
                    st.warning(msg)
            else:
                st.success("Simulated conditions are ideal for growing your predicted crop!")

        # Plant Growth Tracker Section
        st.subheader("Plant Growth Tracker")
        st.write("Track the growth stages of your predicted crops and get care recommendations!")
        st.write("Plants are automatically added when you predict a crop and click 'Add Predicted Crop to Tracker'.")

        st.write("**Your Tracked Plants**")
        tracked_plants = get_tracked_plants()
        if tracked_plants:
            plant_data = []
            for plant in tracked_plants:
                plant_id, plant_name, planting_date, current_stage = plant
                new_stage, needs = get_current_stage(plant_name, planting_date)
                if new_stage != current_stage:
                    update_plant_stage(plant_id, new_stage)
                    current_stage = new_stage

                planting_date_dt = datetime.strptime(planting_date, "%Y-%m-d")
                days_since_planting = (datetime.now() - planting_date_dt).days
                total_duration = PLANT_GROWTH_STAGES[plant_name.lower()]['total_duration_days']
                progress = min(100, (days_since_planting / total_duration) * 100)

                needs_str = f"Water: {needs['water']}, Temperature: {needs['temperature']}, Sunlight: {needs['sunlight']}, Nutrients: {needs['nutrients']}" if needs else "N/A"

                plant_data.append({
                    "Plant Name": plant_name,
                    "Planted On": planting_date,
                    "Current Stage": current_stage,
                    "Needs": needs_str,
                    "Progress (%)": progress
                })

            plant_df = pd.DataFrame(plant_data)
            st.dataframe(plant_df, use_container_width=True)

            for plant in plant_data:
                st.write(f"**{plant['Plant Name']} Progress** (Planted on {plant['Planted On']})")
                st.progress(plant['Progress (%)'] / 100.0)
                st.write(f"Current Stage: {plant['Current Stage']}")
                st.write(f"Current Needs: {plant['Needs']}")
                # Add plot for plant growth timeline
                plot_plant_growth_timeline(plant['Plant Name'], plant['Current Stage'])
        else:
            st.write("No plants are being tracked yet. Predict a crop and add it to the tracker to get started!")

        # Cumulative Carbon Reduction
        try:
            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, crop FROM predictions")
            predictions = cursor.fetchall()
            timestamps = []
            cumulative_carbon = []
            total_carbon = 0
            for timestamp, crop in predictions:
                timestamps.append(timestamp)
                carbon_reduction = estimate_sustainability_metrics(crop)['carbon_reduction']
                total_carbon += carbon_reduction
                cumulative_carbon.append(total_carbon)
            conn.close()
            st.subheader("Cumulative Impact")
            st.write(f"**Total Carbon Footprint Reduction:** {total_carbon:.2f} kg CO2")
            # Add plot for cumulative carbon reduction
            plot_cumulative_carbon_reduction(timestamps, cumulative_carbon)
        except Exception as e:
            st.error(f"Error calculating cumulative impact: {str(e)}")

        # Community Dashboard
        try:
            st.subheader("Community Dashboard")
            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, crop, N, P, K, temperature, humidity, ph, rainfall FROM predictions ORDER BY timestamp DESC LIMIT 5")
            recent_predictions = cursor.fetchall()
            conn.close()
            if recent_predictions:
                community_data = pd.DataFrame(
                    recent_predictions,
                    columns=['Timestamp', 'Crop', 'N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
                )
                st.write("**Recent Community Predictions:**")
                st.dataframe(community_data, use_container_width=True)
                # Add plot for community dashboard
                plot_community_dashboard(community_data)
            else:
                st.write("No community predictions yet. Be the first to share!")
        except Exception as e:
            st.error(f"Error displaying community dashboard: {str(e)}")

        # Leaderboard
        try:
            st.subheader("Sustainability Leaderboard")
            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, crop, points FROM predictions WHERE points IS NOT NULL ORDER BY points DESC LIMIT 5")
            leaderboard = cursor.fetchall()
            conn.close()
            if leaderboard:
                leaderboard_data = pd.DataFrame(
                    leaderboard,
                    columns=['Timestamp', 'Crop', 'Points']
                )
                st.write("**Top Contributors to Sustainability:**")
                st.dataframe(leaderboard_data, use_container_width=True)
                # Add plot for sustainability leaderboard
                plot_sustainability_leaderboard(leaderboard_data)
            else:
                st.write("No entries in the leaderboard yet. Start predicting to earn points!")
        except Exception as e:
            st.error(f"Error displaying leaderboard: {str(e)}")

    # Right column: Chatbot
    with col2:
        st.subheader("💬 Urban Farming Chatbot")
        st.write("Ask me anything about urban farming!")
        # Note: For a more advanced chatbot, consider integrating Google Dialogflow for natural language processing.
        # Example: Replace hardcoded responses with Dialogflow API calls (requires setup in Google Cloud Console).

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        user_input = st.text_input("Your question:", key="chat_input", placeholder="e.g., How do I grow mango in a rooftop garden?")

        if st.button("Ask"):
            if user_input:
                response = get_chatbot_response(user_input)
                st.session_state['chat_history'].append({"user": user_input, "bot": response})

        if st.session_state['chat_history']:
            for chat in st.session_state['chat_history']:
                st.markdown(f"**You:** {chat['user']}")
                st.markdown(f"**Bot:** {chat['bot']}")
                st.markdown("---")


if __name__ == "__main__":
    main()