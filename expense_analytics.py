# Personal Expense Analytics
# done by me for internship project 1
# dataset is dummy data i made for 6 months

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# load the data first
df = pd.read_csv('expense_data.csv')
print("data loaded, shape is:", df.shape)
print(df.head())


# ------- data cleaning -------

# convert date column to proper datetime
df['Date'] = pd.to_datetime(df['Date'])

# extract month and day info from date
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')
df['Day_of_Week'] = df['Date'].dt.day_name()

# check for nulls
print("\nnull values check:")
print(df.isnull().sum())

# no nulls found so moving ahead
# removing duplicates just in case
df.drop_duplicates(inplace=True)
print("after removing duplicates, rows:", len(df))


# ------- basic analysis -------

total = df['Amount'].sum()
print("\ntotal expense for 6 months:", total)

monthly_total = df.groupby('Month')['Amount'].sum()
print("\nmonthly totals:")
print(monthly_total)

category_total = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
print("\ncategory wise:")
print(category_total)

payment_split = df.groupby('Payment_Method')['Amount'].sum()
print("\npayment method split:")
print(payment_split)


# ------- visualizations -------

# setting up the figure with 6 subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 17))
fig.suptitle('Personal Expense Analytics', fontsize=18, fontweight='bold')

colors = ['#4472C4', '#ED7D31', '#A9D18E', '#FF6B6B', '#9C27B0', '#00BCD4']

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# chart 1 - monthly trend line chart
ax1 = axes[0, 0]
ax1.plot(month_labels, monthly_total.values, marker='o', color='#4472C4',
         linewidth=2, markersize=7)
ax1.fill_between(range(6), monthly_total.values, alpha=0.1, color='#4472C4')
for i, v in enumerate(monthly_total.values):
    ax1.text(i, v + 150, f'Rs.{v}', ha='center', fontsize=8)
ax1.set_title('Monthly Expense Trend')
ax1.set_ylabel('Amount (Rs.)')
ax1.set_xticks(range(6))
ax1.set_xticklabels(month_labels)
ax1.grid(True, linestyle='--', alpha=0.5)

# chart 2 - category bar chart
ax2 = axes[0, 1]
ax2.bar(category_total.index, category_total.values, color=colors)
ax2.set_title('Spending by Category')
ax2.set_ylabel('Amount (Rs.)')
ax2.tick_params(axis='x', rotation=15)
for i, v in enumerate(category_total.values):
    ax2.text(i, v + 100, str(v), ha='center', fontsize=8)
ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

# chart 3 - pie chart for category
ax3 = axes[1, 0]
ax3.pie(category_total.values, labels=category_total.index,
        autopct='%1.1f%%', colors=colors, startangle=90,
        wedgeprops={'edgecolor': 'white'})
ax3.set_title('Category Wise Distribution')

# chart 4 - payment method donut
ax4 = axes[1, 1]
ax4.pie(payment_split.values, labels=payment_split.index,
        autopct='%1.1f%%', startangle=90,
        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
        wedgeprops={'width': 0.5, 'edgecolor': 'white'})
ax4.set_title('Payment Method Usage')

# chart 5 - day wise spending
ax5 = axes[2, 0]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_data = df.groupby('Day_of_Week')['Amount'].sum().reindex(day_order, fill_value=0)
bar_colors = ['#FF6B6B' if d in ['Saturday', 'Sunday'] else '#4472C4' for d in day_order]
ax5.bar([d[:3] for d in day_order], day_data.values, color=bar_colors)
ax5.set_title('Day wise Spending')
ax5.set_ylabel('Amount (Rs.)')
ax5.grid(True, axis='y', linestyle='--', alpha=0.4)
p1 = mpatches.Patch(color='#4472C4', label='Weekday')
p2 = mpatches.Patch(color='#FF6B6B', label='Weekend')
ax5.legend(handles=[p1, p2])

# chart 6 - heatmap
ax6 = axes[2, 1]
pivot_data = df.pivot_table(values='Amount', index='Category',
                            columns='Month_Name', aggfunc='sum', fill_value=0)
# reorder months
ordered_months = [m for m in ['January','February','March','April','May','June'] if m in pivot_data.columns]
pivot_data = pivot_data[ordered_months]
sns.heatmap(pivot_data, ax=ax6, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.5)
ax6.set_title('Month vs Category Heatmap')
ax6.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('expense_visualizations.png', dpi=150, bbox_inches='tight')
plt.show()
print("charts saved")


# ------- prediction model -------

# using linear regression to predict next month expense
# took month number as feature and monthly total as target

X = monthly_total.index.values.reshape(-1, 1)
y = monthly_total.values

model = LinearRegression()
model.fit(X, y)

# predict july (month 7)
july_pred = model.predict([[7]])[0]

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print(f"\nModel Results:")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"Predicted expense for July: Rs.{july_pred:.0f}")

# prediction chart
fig2, ax = plt.subplots(figsize=(9, 5))
ax.scatter(X, y, color='#4472C4', s=70, zorder=5, label='Actual')
ax.plot(X, y_pred, color='#ED7D31', linewidth=2, label='Regression Line')
ax.scatter([7], [july_pred], color='green', s=120, zorder=6, marker='*',
           label=f'July Prediction: Rs.{july_pred:.0f}')
ax.set_title('Monthly Expense Prediction')
ax.set_xlabel('Month')
ax.set_ylabel('Amount (Rs.)')
ax.set_xticks(range(1, 8))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul(pred)'])
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('expense_prediction.png', dpi=150, bbox_inches='tight')
plt.show()
print("prediction chart saved")


# final summary
print("\n--- Summary ---")
print(f"Total records: {len(df)}")
print(f"Total spent: Rs.{total}")
print(f"Monthly average: Rs.{total//6}")
print(f"Highest category: {category_total.idxmax()}")
print(f"Most used payment: {payment_split.idxmax()}")
print(f"July prediction: Rs.{july_pred:.0f}")
