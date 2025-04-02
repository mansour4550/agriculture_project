AgriTunisia SmartCrop: AI-Powered Urban Farming
Overview
AgriTunisia SmartCrop is an AI-powered web application designed to optimize urban farming for sustainable cities. The application recommends the best crops to grow based on environmental conditions and provides detailed guidance for urban farmers. It features:

Crop Recommendation Engine: Uses machine learning to suggest optimal crops

Sustainability Analytics: Estimates water usage and carbon footprint reduction

IoT Integration: Simulates real-time environmental monitoring

Plant Growth Tracker: Helps users monitor crop development stages

Community Features: Leaderboards and shared data for collaborative farming

Educational Chatbot: Provides urban farming advice

Features
1. AI-Powered Crop Recommendation
Random Forest classifier trained on agricultural data

Recommends top 3 suitable crops with probability scores

Visualizes recommendations with interactive charts

2. Sustainability Analysis
Estimates water usage and carbon footprint reduction

Calculates sustainability points for gamification

Tracks cumulative environmental impact over time

3. IoT Simulation
Interactive sliders to simulate environmental sensors

Real-time feedback on ideal growing conditions

Comparative analysis of current vs ideal parameters

4. Plant Growth Tracking
Monitors growth stages with timeline visualization

Provides stage-specific care instructions

Progress tracking with percentage completion

5. Community Features
Leaderboard of top sustainability contributors

Recent community predictions dashboard

Shared environmental data visualization

6. Educational Chatbot
Answers common urban farming questions

Provides crop-specific growing advice

Offers sustainability tips and techniques

Technical Implementation
Backend
Machine Learning Model: Random Forest Classifier

Data Processing: StandardScaler for feature normalization

Persistence: SQLite database for prediction history

Report Generation: PDF reports with ReportLab

Frontend
Streamlit: Python web framework for interactive UI

Data Visualization: Matplotlib and Seaborn for charts

Interactive Elements: Custom-styled sliders and buttons

Data Flow
User inputs environmental parameters

System scales inputs and makes prediction

Results are stored in database

Visualizations are generated

User receives recommendations and reports

Setup Instructions
Prerequisites
Python 3.7+

Required Python packages (install via pip install -r requirements.txt):

Copy
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
sqlite3
reportlab
Installation
Clone the repository:

bash
Copy
git clone https://github.com/yourusername/agritunisia-smartcrop.git
cd agritunisia-smartcrop
Install dependencies:

bash
Copy
pip install -r requirements.txt
Download the dataset and place in project directory:

Dataset file: Crop_recommendation.csv

Running the Application
bash
Copy
streamlit run app.py
The application will launch in your default web browser at http://localhost:8501

File Structure
Copy
agritunisia-smartcrop/
├── app.py                  # Main application code
├── Crop_recommendation.csv  # Training dataset
├── crop_model.pkl          # Trained model (generated)
├── scaler.pkl              # Feature scaler (generated)
├── label_encoder.pkl       # Label encoder (generated)
├── predictions.db          # Prediction database (generated)
├── plant_tracker.db        # Plant tracking database (generated)
├── requirements.txt        # Python dependencies
└── README.md               # This file
Customization
Model Training
To retrain the model with different parameters:

Modify the train_model() function in app.py

Adjust RandomForestClassifier parameters:

python
Copy
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
Data Sources
Replace Crop_recommendation.csv with your own dataset

Ensure the dataset has the same feature columns:

N, P, K, temperature, humidity, ph, rainfall, label

UI Customization
Modify the Streamlit components in main()

Adjust color schemes in the custom CSS section

Add new visualizations by extending the plotting functions

Usage Examples
Getting Crop Recommendations:

Adjust the sliders for your environmental conditions

Click "Predict Crop" to see recommendations

View detailed sustainability metrics

Tracking Plant Growth:

After getting a recommendation, add to tracker

Monitor progress through growth stages

Get care instructions for each stage

Community Features:

View recent community predictions

Compare your sustainability score on the leaderboard

Learn from others' growing conditions

Troubleshooting
Issue: Model files not found
Solution: Run the training process by removing existing .pkl files or setting force_train=True

Issue: Database errors
Solution: Delete corrupt .db files to regenerate fresh databases

Issue: Visualization errors
Solution: Check matplotlib/seaborn versions match requirements

Future Enhancements
Real IoT Integration: Connect to actual sensor hardware

Mobile App: Native iOS/Android versions

Advanced Chatbot: Integrate with Dialogflow for NLP

Multi-language Support: Localize for different regions

Marketplace Features: Connect urban farmers with local markets

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Streamlit for the web framework

Scikit-learn for machine learning tools

ReportLab for PDF generation

The urban farming community for inspiration
