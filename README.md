association-mining-analysis
A Streamlit-based data mining application that performs Association Rule Mining using techniques such as Apriori and FP-Growth algorithms to discover association rules, frequent itemsets, and purchasing patterns from transaction datasets.

An interactive Streamlit web application for performing Market Basket Analysis using:

- Apriori Algorithm
- FP-Growth Algorithm
- Association Rule Mining
  
# Features

- One-Hot Encoding of transaction data
- Frequent Itemset Generation
- Apriori Algorithm
- FP-Growth Algorithm
- Association Rule Mining
- Rule Filtering using:
  - Support
  - Confidence
  - Lift
- Interactive Threshold Selection
- Data Visualization using Matplotlib
- Streamlit-based Interactive Dashboard

# Concepts Used

- Data Mining
- Association Rule Mining
- Frequent Pattern Analysis
- Market Basket Analysis
- Support, Confidence and Lift Metrics

# Technologies Used

- Python
- Streamlit
- Pandas
- mlxtend
- Matplotlib

# How to Run the Project

1. Clone the Repository
git clone <your-repository-link>

2. Open Project Folder
cd market-basket-analysis

3. Install Dependencies
pip install -r requirements.txt

4. Run Streamlit App
streamlit run app.py


---

# Algorithms Implemented

1.  Apriori Algorithm

Used to identify frequent itemsets based on minimum support threshold.

2. FP-Growth Algorithm

Efficient frequent pattern mining algorithm without candidate generation.

3. Association Rules

Rules are generated using confidence threshold and evaluated using lift.

---

# Evaluation Metrics

a) Support
Measures how frequently an itemset appears in the dataset.

b) Confidence
Measures the likelihood of occurrence of consequent items.

c) Lift
Measures strength of association between itemsets.

---



# Future Improvements

- CSV File Upload Support
- Interactive Plotly Visualizations
- Dynamic Dataset Handling
- Rule Export Feature
- Advanced Dashboard UI

---

Author
Hiranyaa Vignesh
