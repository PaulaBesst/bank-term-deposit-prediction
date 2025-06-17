# 🏦 Term Deposit Prediction Model

A machine learning project that predicts whether a bank client will subscribe to a term deposit based on marketing campaign data. Built with Python, Scikit-learn, and deployed with Streamlit.

## Project Overview

This project analyzes bank marketing campaign data to predict client subscription to term deposits using machine learning algorithms. The model helps banks optimize their marketing strategies by identifying clients most likely to subscribe.

## Dataset

- **Records**: 41,188 client interactions
- **Original Features**:  20 input variables plus target variable 
- **Target**: Binary classification (subscribe/not subscribe)
- **Time Period**: Banking campaign data
- **Class Distribution**: Imbalanced dataset (~11.3% positive class)

## Features

- **Data Preprocessing**: Handles missing values and categorical encoding
- **Class Imbalance**: Uses SMOTE oversampling technique
- **Multiple Models**: Random Forest, XGBoost, and Logistic Regression
- **Feature Engineering**: Focused selection of key banking features
- **Interactive Dashboard**: Streamlit web application for predictions

## Technologies Used

- **Python 3.8+**
- **Machine Learning**: Scikit-learn, XGBoost, Imbalanced-learn
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit
- **Deployment**: Streamlit Community Cloud 

 ## Bank Marketing Dataset - Data Dictionary

### Client Information
| Variable | Type | Description |
|----------|------|-------------|
| `age` | Numeric | Client's age in years | 
| `job` | Categorical | Type of job/occupation | 
| `marital` | Categorical | Marital status | 
| `education` | Categorical | Education level | 
| `default` | Binary | Has credit in default? | 
| `housing` | Binary | Has housing loan? | 
| `loan` | Binary | Has personal loan? |

### Contact Information
| Variable | Type | Description | 
|----------|------|-------------|
| `contact` | Categorical | Contact communication type | 
| `month` | Categorical | Last contact month of year | 
| `day_of_week` | Categorical | Last contact day of the week |
| `duration` | Numeric | Last contact duration in seconds | 

### Campaign Information
| Variable | Type | Description | 
|----------|------|-------------|
| `campaign` | Numeric | Number of contacts performed during this campaign |
| `pdays` | Numeric | Number of days since client was last contacted from previous campaign |
| `previous` | Numeric | Number of contacts performed before this campaign | 
| `poutcome` | Categorical | Outcome of previous marketing campaign | 

### Economic Context Attributes
| Variable | Type | Description | 
|----------|------|-------------|
| `emp.var.rate` | Numeric | Employment variation rate (quarterly indicator) | 
| `cons.price.idx` | Numeric | Consumer price index (monthly indicator) | 
| `cons.conf.idx` | Numeric | Consumer confidence index (monthly indicator) | 
| `euribor3m` | Numeric | Euribor 3 month rate (daily indicator) | 
| `nr.employed` | Numeric | Number of employees (quarterly indicator) | 

### Target Variable
| Variable | Type | Description | 
|----------|------|-------------|
| `y` | Binary | **TARGET**: Has the client subscribed to a term deposit? | **no, yes** |

---

## Project Structure

```
term-deposit-prediction/
├── term_deposit_prediction.py    # Main analysis script
├── app.py              # Streamlit web application
├── best_bank_model.pkl           # Trained model (best performing)
├── feature_columns.pkl           # Feature columns for preprocessing
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── data/
    └── bank-additional-full.csv  # Dataset (if included)
```

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/term-deposit-prediction.git
cd term-deposit-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Analysis
```bash
python term_deposit_prediction.py
```

### 5. Launch Streamlit App
```bash
streamlit run app.py
```

## Model Performance

The project evaluates three machine learning models:

| Model | Features | Technique |
|-------|----------|-----------|
| Random Forest | Ensemble method with class balancing | `class_weight='balanced'` |
| XGBoost | Gradient boosting with scale balancing | `scale_pos_weight` |
| Logistic Regression | Linear model with class weights | `class_weight='balanced'` |

**Best Model Selection**: Automatically selects the model with highest ROC AUC score.

## Key Features Used

The model focuses on 13 key banking features:

- **Client Demographics**: Age, job, marital status, education
- **Campaign Data**: Duration, campaign contacts, previous contacts
- **Economic Indicators**: Employment rate, consumer price index, Euribor rate
- **Previous Outcomes**: Results from previous campaigns

## Results & Insights

### Feature Importance
The analysis reveals the most predictive features for term deposit subscription:

1. **Call Duration** - Most important predictor
2. **Economic Indicators** - Significant impact on decisions
3. **Previous Campaign Results** - Strong predictive power
4. **Client Demographics** - Age and job category influence

### Business Recommendations

1. **Focus on Call Quality**: Longer, meaningful conversations increase conversion
2. **Economic Timing**: Align campaigns with favorable economic conditions
3. **Leverage History**: Use previous interaction data for targeting

## 🌐 Live Demo

<!-- Add your deployed app links here -->
- **Streamlit Cloud**: [https://bank-term-deposit-prediction-7jgw7a89zirhwjc2ckverg.streamlit.app/]

### Streamlit Web Application
![Streamlit App](images/streamlit_app.png)
*Interactive web interface for making real-time predictions*

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/term-prediction`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add term prediction'`)
5. Push to the branch (`git push origin feature/prediction`)
6. Open a Pull Request


## 👨‍💻 Author

**Paula Obeng-Bioh**
- LinkedIn: [Paula Obeng-Bioh](www.linkedin.com/in/paula-obeng-bioh-38a58a190)

