# VacSmart: AI-Powered Vaccine Prediction System

## Overview
VacSmart is a machine learning-based web application that predicts an individual's likelihood of getting H1N1 and seasonal flu vaccines. The system uses demographic and health-related features to provide personalized vaccination recommendations.

## Features
- **Real-time Prediction**: Get instant vaccination recommendations
- **Dual Prediction**: Separate predictions for H1N1 and seasonal flu vaccines
- **Confidence Metrics**: Visual gauge charts showing prediction confidence
- **Interactive FAQ**: Comprehensive vaccine information
- **User-friendly Interface**: Clean, modern design with intuitive controls

## Technical Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **ML Models**: XGBoost
- **Data Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## Project Structure
```
analyticaZ/
├── src/
│   ├── app.py              # Main application file
│   ├── data/               # Dataset directory
│   │   └── training_set_features.csv
│   └── models/             # Trained models
│       ├── h1n1_model.pkl
│       └── seasonal_model.pkl
├── requirements.txt
└── README.md
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/vacsmart.git
cd vacsmart
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Start the Streamlit app:
```bash
cd src
streamlit run app.py
```

2. Open your browser and navigate to:
```
http://localhost:8501
```

## Model Features
The system uses 11 key predictive features:
- Age Group
- Education Level
- Race/Ethnicity
- Gender
- Income Level
- Health Insurance Status
- H1N1 Risk Opinion
- Vaccine Effectiveness Opinion
- Chronic Medical Conditions
- Healthcare Worker Status
- Marital Status

## Data Source
The model is trained on the National 2009 H1N1 Flu Survey (NHFS) dataset, which includes:
- 35 total features
- Demographic information
- Behavioral indicators
- Health-related variables

## Performance Metrics
- H1N1 Vaccine Prediction:
  - Accuracy: ~80%
  - AUC-ROC: 0.85

- Seasonal Flu Prediction:
  - Accuracy: ~75%
  - AUC-ROC: 0.80

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Data source: National 2009 H1N1 Flu Survey (NHFS)
- Streamlit for the web framework
- XGBoost for the machine learning models

## Contact
Your Name - [@yourusername](https://twitter.com/yourusername)
Project Link: [https://github.com/yourusername/vacsmart](https://github.com/yourusername/vacsmart)