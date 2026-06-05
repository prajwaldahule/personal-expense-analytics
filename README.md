# Personal Expense Analytics

**Intern ID: CITS1288**  
**Name: Prajwal Jitendra Dahule**  
**Organization: Codtech IT Solutions Pvt. Ltd**  
**Internship Period: 20 May 2026 - 17 June 2026**

---

This is my first internship project. I did analysis on personal expense data to find spending patterns and also made a prediction model for next month expenses.

---

## Tools and Libraries Used

- Python 3
- Pandas (for data handling)
- Matplotlib and Seaborn (for charts)
- Scikit-learn (for prediction model)

---

## Dataset

I used a dummy dataset that I created with 6 months of expense data (January to June 2024). It has 114 records and 5 columns.

Columns in the dataset:
- Date
- Category (Food, Shopping, Transport, Health, Entertainment, Utilities)
- Amount (in rupees)
- Payment_Method (UPI, Cash, Credit Card, Net Banking)
- Description

---

## Steps I Followed

**1. Data Loading**
Loaded the CSV file using pandas and checked the shape and first few rows.

**2. Data Cleaning**
- Converted Date column from string to datetime format
- Extracted month name and day of week from the date
- Checked for null values — none were found
- Removed duplicate rows

**3. Analysis**
- Calculated total expense for 6 months
- Found monthly totals and category wise breakdown
- Checked which payment method was used the most

**4. Visualization**
Made 6 charts:
1. Monthly expense trend (line chart)
2. Category wise spending (bar chart)
3. Category distribution (pie chart)
4. Payment method usage (donut chart)
5. Day wise spending pattern
6. Month vs category heatmap

**5. Prediction Model**
Used Linear Regression to predict July 2024 expense based on previous 6 months data.

---

## Results

- Total expense in 6 months: Rs. 82,108
- Average monthly expense: Rs. 13,685
- Highest spending category: Shopping
- Most used payment method: Credit Card
- Predicted expense for July 2024: Rs. 13,774
- Model R2 Score: 0.0018

---

## Files in this Project

- expense_data.csv — dataset
- expense_analytics.py — main python code
- expense_visualizations.png — all 6 charts
- expense_prediction.png — prediction chart
- README.md — this file

---

## How to Run

Install the required libraries first:
```
pip install pandas matplotlib seaborn scikit-learn
```

Then run the python file:
```
python expense_analytics.py
```
