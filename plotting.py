import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('scenario_6.csv')

# Extract data for the first 60 steps
df_subset = df.head(60)

# Plot the data
plt.figure(figsize=(10, 6))

plt.plot(df_subset['Step'], df_subset['Susceptible'], label='Susceptible', color='green')
plt.plot(df_subset['Step'], df_subset['Exposed'], label='Exposed', color='yellow')
plt.plot(df_subset['Step'], df_subset['Infected'], label='Infected', color='red')
plt.plot(df_subset['Step'], df_subset['Recovered'], label='Recovered', color='grey')

plt.xlabel('Step')
plt.ylabel('Number of Individuals')
# plt.title('The model runs without any interventional measures')
plt.legend()
plt.show()
