# Climate Future Regression

This project analyzes climate change trends using machine learning regression models to predict **future temperature anomalies** based on historical CO₂ emissions and global temperature data.

---

## 📁 Dataset

- `climate_data.csv`: Contains yearly data on CO₂ emissions and global temperature changes.
- Columns used: `year`, `co2`, `temperature_change`

---

## 📊 Models Implemented

1. **Linear Regression**
2. **Polynomial Regression**
3. **Random Forest Regression**
4. **Support Vector Regression (SVR)**

Each model:
- Trains on historical data
- Predicts temperature anomalies for future years (2030, 2040, 2050)
- Shows prediction results both as **text output** and **graphical plots**

---

## 🧪 How to Run

1. Make sure you have Python 3+ installed.
2. Install required libraries:

```bash
pip install pandas matplotlib scikit-learn


Run the script:
python climate_future_regression.py



📈 Output

The script:

Displays future predictions in the terminal

Plots line graphs showing how temperature is expected to change based on CO₂ trends


🤔 What is Temperature Anomaly?

Temperature Anomaly means how much the global temperature deviates from a long-term average baseline (usually the 1951–1980 average).
Positive values = warmer than average
Negative values = cooler than average


📌 Purpose

To highlight the potential future effects of rising CO₂ emissions using data-driven predictions. Useful for students, researchers, and climate activists.


👤 Author

Okasha Shehbaz Mueed
Cyber Security | Machine Learning | Environmental Enthusiast