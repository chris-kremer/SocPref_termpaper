import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data from Excel file
file_path = 'Soc Pref Literature Overview1.xlsx'  # Update with the actual file name
sheet_name = 'Sheet1'  # Update with the actual sheet name, if necessary

replication_df = pd.read_excel(file_path, sheet_name=sheet_name)

# Debug: Check for missing and infinite values
print("Initial Data Overview:")
print(replication_df.info())
print("Missing values per column:")
print(replication_df.isna().sum())

# Add a constant to the model for the intercept term
replication_df['const'] = 1

# Replace infinite values and drop rows with missing data
replication_df.replace([np.inf, -np.inf], np.nan, inplace=True)
replication_df = replication_df.dropna(subset=['Replicated', 'Citations'])

print("Data Overview After Cleaning:")
print(replication_df.info())

# Perform the regression of citations on replicated
model = sm.OLS(replication_df['Citations'], replication_df[['const', 'Replicated']])
results = model.fit()

# Print the regression summary
print(results.summary())

# Calculate mean, min, and max citations for replicated=0 and replicated=1
stats = replication_df.groupby('Replicated')['Citations'].agg(
    mean='mean', min_val='min', max_val='max'
).reset_index()

# Plotting the bar chart with min and max range, adding a different color and grid for style
plt.figure(figsize=(8, 6))
sns.barplot(x='Replicated', y='mean', data=stats, color="cornflowerblue", edgecolor="black", capsize=0.1)

# Adding lines with hooks for the min and max values
hook_length = 0.15  # length of the horizontal hooks for better visibility

for index, row in stats.iterrows():
    # Vertical line between min and max values
    plt.plot([index, index], [row['min_val'], row['max_val']], color="darkred", linewidth=2, linestyle="--")
    # Horizontal hooks at the ends of the lines
    plt.plot([index - hook_length, index + hook_length], [row['min_val'], row['min_val']], color="darkred", linewidth=2)
    plt.plot([index - hook_length, index + hook_length], [row['max_val'], row['max_val']], color="darkred", linewidth=2)

# Customizing the plot aesthetics
plt.xlabel('Replication Status')
plt.ylabel('Average Citations')
plt.title('Average Citations by Replication Status with Min/Max Ranges')
plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7)  # adding gridlines for the y-axis
plt.show()