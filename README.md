# ML Prediction Platform

An enterprise-grade machine learning platform for training, managing, and deploying predictive models. Built with Streamlit and modern ML libraries, it provides an intuitive interface for end-to-end machine learning workflows.

## 🚀 Features

### Model Training
- Multiple algorithms supported:
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
  - Gradient Boosting
  - Stacking Ensemble
- Automated feature engineering
- Hyperparameter optimization via Grid Search
- Cross-validation
- Interactive model training progress

### Model Management
- Version control for trained models
- Model metadata tracking
- Performance metrics comparison
- Easy model selection
- One-click model deletion
- Export functionality

### Data Processing
- Automated preprocessing
- Feature importance analysis
- Missing value handling
- Date feature extraction
- Categorical encoding
- Outlier detection

### Visualization
- Interactive performance metrics
- Error analysis dashboards
- Feature importance plots
- Prediction vs actual comparisons
- Residual analysis
- Model comparison charts

### User Interface
- Clean, modern design
- Responsive layout
- Interactive components
- Real-time updates
- Progress tracking
- Error handling

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository
```bash
git clone https://github.com/bomino/SpendPredic.git
cd ml-prediction-platform
```

2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 📊 Usage

1. Start the application
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

### Data Requirements

Your CSV file should contain:
- Date column (e.g., transaction_date)
- At least one categorical column (e.g., department)
- Target column (numerical value to predict)

Example format:
```csv
Date,Category,Amount
2023-01-01,Electronics,1500
2023-01-01,Clothing,800
2023-01-02,Electronics,2000
```

## 📁 Project Structure

```
ml-prediction-platform/
├── .gitignore            # Git ignore rules
├── app.py                # Main application entry point
├── requirements.txt      # Project dependencies
├── README.md            # Project documentation
├── LICENSE              # MIT license
├── DOCUMENTATION.md     # Comprehensive documentation
├── model.py             # ML model implementations
├── model_management.py  # Model versioning and storage
├── preprocessing.py     # Data preprocessing utilities
│
├── config/              # Configuration files
│   └── styles.py       # UI styling and theme
│
├── pages/              # Additional pages
│   ├── About.py       # About page
│   └── QuickStart.py  # Quick start guide
│
└── models/            # Directory for saved models
    └── model_metadata.json
```

## 🔧 Configuration

Key configurations are managed in `config/styles.py`:
- UI theme and styling
- Color schemes
- Component layouts
- Visual elements

## 📦 Dependencies

```
streamlit
streamlit-card
pandas
scikit-learn
plotly
matplotlib
xgboost
lightgbm
catboost
```

## 🚀 Quick Start

1. **Prepare Data**
   - Ensure CSV format
   - Clean data
   - Verify required columns

2. **Train Model**
   - Upload data
   - Select columns
   - Choose model type
   - Start training

3. **Make Predictions**
   - Select trained model
   - Upload new data
   - Download predictions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📫 Support

For support, please open an issue in the GitHub repository or contact the maintainers.