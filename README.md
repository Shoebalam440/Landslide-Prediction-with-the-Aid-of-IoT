# Landslide Prediction with the Aid of IoT

End-to-end demo that combines wireless-sensor data, a RandomForest classifier, and an interactive Streamlit dashboard to estimate landslide risk in real time. The dashboard highlights the most important IoT signals (rainfall, soil moisture, pore-water pressure, seismic activity) and lets users run manual “what-if” scenarios, compare presets, and export saved simulations.

## Project Modules

1. **IoT Sensing Layer (Arduino / WSN)**  
   - Collect rainfall, soil-moisture, and optional temperature / microseismic readings.  
   - Stream values via serial, Wi-Fi, or LoRa to a gateway script.
2. **Data Bridge**  
   - Python script ingests live readings into `wsn_landslide_data.csv` or a database.  
   - Historical dataset (sample provided) is used to train and benchmark the model.
3. **ML & Dashboard**  
   - `app.py` trains a RandomForest on demand and serves the Streamlit UI with tabs for Overview, Model Insights, and Prediction Studio.

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\activate        # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

The app loads `wsn_landslide_data.csv`, fits the model, and opens a browser window at `http://localhost:8501`.

## Dashboard Highlights

- **Overview tab**: KPIs, data-quality snapshot, and feature distributions.
- **Model Insights**: Accuracy/precision/recall metrics, ROC curve, confusion heatmap, and feature-importance chart.
- **Prediction Studio**:
  - Manual form with tooltips explaining each sensor field.
  - Sidebar presets for typical sites (mountain village, coastal highway, etc.).
  - Probability gauge, local factor explanation, scenario history, and CSV export.
  - “What-if” slider to see how changing one feature shifts risk.

## Adapting to Your Sensor Stack

1. **Match Features**: If your Arduino setup only captures rainfall + moisture, subset those columns in the dataset and retrain to keep the model realistic.
2. **Data Feed**: Append new readings to the CSV (or expose an API) so the dashboard reflects recent conditions.
3. **Deployment**: Package with Docker or run on a small edge PC (Raspberry Pi, Intel NUC) colocated with the gateway.

## Repository Structure

```
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── wsn_landslide_data.csv  # Sample high-resolution dataset (sensor + contextual features)
└── .streamlit/config.toml  # Custom theme for Streamlit
```

## Future Enhancements

- Automate retraining when new labeled events arrive.
- Add alerting (email/SMS) when probability crosses a threshold.
- Integrate SHAP for per-scenario explanations.
- Replace the CSV bridge with MQTT / REST ingestion for true real-time updates.

## License

MIT Certified


