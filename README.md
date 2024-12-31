# AlphaCare Insurance Analysis

## **Overview**
Welcome to the AlphaCare Insurance Analysis project! This repository showcases advanced data analysis and machine learning techniques applied to real-world problems in the insurance industry. The focus of this project is to analyze car insurance claim data and develop solutions to optimize marketing strategies and identify low-risk customer segments for AlphaCare Insurance Solutions (ACIS) in South Africa.

## **Business Objective**
AlphaCare Insurance Solutions (ACIS) aims to revolutionize risk assessment and predictive analytics in the insurance domain. This project addresses key challenges by:
- Enhancing marketing strategies for targeted campaigns.
- Identifying low-risk customers for premium reductions.
- Building predictive models to guide data-driven decision-making.

---

## **Project Highlights**
### **Exploratory Data Analysis (EDA)**
- Detailed summarization of key metrics like `TotalPremium` and `TotalClaims`.
- Data quality checks to handle missing values and inconsistencies.
- Advanced visualizations to uncover geographic and demographic trends.
- Outlier detection using statistical and graphical methods.

### **Statistical and Predictive Modeling**
- Statistical insights to validate actionable recommendations.
- Development of predictive models to estimate claim probabilities and premium optimization.
- Feature importance analysis to understand customer segmentation.

---

## **Tools and Technologies**
### **Programming and Analysis**
- Python: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels.
### **Version Control and CI/CD**
- Git and GitHub for transparent version management.
- GitHub Actions for automated testing and deployment pipelines.
### **Visualization**
- Comprehensive graphical analyses using Matplotlib and Seaborn.

---

## **Folder Structure**
```plaintext
AlphaCare-Insurance-Analysis/
├── .dvc/
│   ├── cache/
│   │   ├── files/
│   │   │   ├── md5/
│   │   │   │   └── 09/
│   └── tmp/
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .vscode/
│   └── settings.json
├── src/
│   ├── __init__.py
│   ├── preprocess_data.py
│   ├── train_model.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_evaluation.ipynb
│   ├── A/
│   ├── data/
│   │   ├── data_cleaned/
│   └── plots/
├── tests/
│   ├── __init__.py
│   └── test_models.py
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── visualize_results.py
│   └── __pycache__/
├── data/
│   ├── raw/
│   │   └── insurance_data.csv
│   ├── processed/
│       └── cleaned_data.csv
├── output/
├── reports/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## **Performance Overview**
### **Models Implemented**
- **Linear Regression**: R² Score: 0.7845
- **Random Forest Regressor**: R² Score: 0.8934  

### **Feature Importance**
- **Top Predictors**: `SumInsured`, `CustomValueEstimate`, `TotalPremium`

---

## **Usage Guide**
### Prerequisites
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: `env\Scripts\activate`
   pip install -r requirements.txt
   ```

### Run the Project
1. **Load Data**: Use the `scripts/preprocess_data.py` to clean and preprocess data.
2. **Perform EDA**: Explore the data through Jupyter notebooks in the `notebooks/` directory.
3. **Train Models**: Use `src/train_model.py` to train and evaluate predictive models.

---

## **Insights and Recommendations**
### **Key Findings**
1. Random Forest outperforms Linear Regression with an R² score of 0.8934.
2. `SumInsured` is the strongest predictor of claim amounts, highlighting its critical role in risk assessment.
3. Vehicle-related features significantly impact claim predictions.

### **Actionable Recommendations**
1. **Adopt Predictive Models**: Deploy Random Forest for premium optimization and risk scoring.
2. **Focus on Key Features**: Enhance data collection for high-impact variables like `VehicleAge` and `PolicyDuration`.
3. **Automate Risk Assessment**: Use model outputs to automate underwriting and claim predictions.

---

## **Author**
**Yonatan Abrham**  
- Email: [email2yonatan@gmail.com](mailto:email2yonatan@gmail.com)  
- LinkedIn: [Yonatan Abrham](https://www.linkedin.com/in/yonatan-abrham1/)  
- GitHub: [YonInsights](https://github.com/YonInsights)  
Feel free to connect for collaborations or queries.

---

## **Acknowledgements**
- Heartfelt thanks to 10 Academy for providing an excellent internship opportunity.
- Appreciation for the open-source tools and the community that made this project possible.

---

> **"Data is the lifeblood of decision-making, and this project brings it to life."**
