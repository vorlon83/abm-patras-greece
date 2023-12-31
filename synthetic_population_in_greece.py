# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# https://elstat-outsourcers.statistics.gr/Census2022_GR.pdf
# POPULATION_SIZE = 211593

# https://www.ypes.gr/eklogika-tmimata/
# https://www.statistics.gr/documents/20181/17286366/APOF_APOT_MON_DHM_KOIN.pdf/41ae8e6c-5860-b58e-84f7-b64f9bc53ec4
# απογραφή Πάτρας 2021: 187486
# εγγεγραμμένοι ψηφοφόροι στην Πάτρα: 132102
# εγγεγραμμένοι ψηφοφόροι στο 4ο ΔΔ Κέντρο Πάτρας: 62143
# ==> πληθυσμός 4ου ΔΔ Πάτρας: 88225

# TOTAL_SCHOOLS = 150
# NUM_WORKPLACES = 12500
# POPULATION_SIZE = 10000
POPULATION_SIZE = 88225

TOTAL_SCHOOLS = 8 * 8
NUM_WORKPLACES = 600 * 8

# Create a dataframe with POPULATION_SIZE rows and 5 columns
df = pd.DataFrame(np.random.randint(0,100,size=(POPULATION_SIZE, 5)), columns=['Gender', 'Age', 'Family_ID', 'Work_ID', 'School_ID'])

# Display the dataframe
print(df)

"""read an external csv file "PatrasPopulation.csv" in a new dataframe 'rdf' and keep the columns ['age_group', 'population', 'Mpop', 'Fpop']"""

# Read the CSV file into a dataframe, keeping only the desired columns
rdf = pd.read_csv("PatrasPopulation.csv", usecols=['age_group', 'population', 'Mpop', 'Fpop'])

# Display the dataframe
print(rdf.head())

"""calculate female ratio from rdf dataframe, if 'Fpop' is the total number of females and 'population' is the total population"""

# Calculate the female ratio
rdf['female_ratio'] = rdf['Fpop'] / rdf['population']

# Display the dataframe
print(rdf)

"""in the df dataframe, replace the elements of the 'Gender' column, according to their df['Age'] and rdf['female_ratio'], as follows: if df['Gender'] is below rdf['female_ratio'], than make it 0 (thus female), else make it 1 (thus male)

"""

# import numpy as np

# Define a function that takes a row of the dataframe and returns a new value for the 'Gender' column
def update_gender(row):
    # Get the female ratio for the corresponding age group
    p = rdf[rdf['age_group'] == row['Age']]['female_ratio'].iloc[0]

    # Sample from a binomial distribution with probability p
    return np.random.binomial(1, p)

# Apply the function to each row of the dataframe
df['Gender'] = df.apply(update_gender, axis=1)

# Display the dataframe
# df

# df.describe()

# df

import random

# Calculate the total population and the population weights
age_groups = rdf['age_group']
population = rdf['population']
total_population = sum(population)
population_weights = population / total_population

population_weights.plot()

# Sample from the categorical distribution defined by the age groups and population weights
samples = random.choices(age_groups, population_weights, k=POPULATION_SIZE)

# Round the values in the 'Age' column to the nearest integer
df['Age'] = round(df['Age'])

# Map the values in the 'Age' column to the corresponding age groups
df['Age'] = df['Age'].map(age_groups)

# df

"""now, read an external scv file "familySizeGreece.csv" and specifically the columns 'family_size' and 'family_sum'. Calculate the percentage of being a member of a family size from 1 to 10

"""

# import pandas as pd

# Read the CSV file into a dataframe
psdf = pd.read_csv('familySizeGreece.csv')

# Select the 'family_size' and 'family_sum' columns
psdf = psdf[['family_size', 'family_sum']]

# Calculate the percentage of being a member of each family size
psdf['percentage'] = psdf['family_sum'] / psdf['family_sum'].sum()

# Filter the dataframe to include only family sizes from 1 to 10
psdf = psdf[psdf['family_size'].between(1, 10)]

# Save the resulting dataframe as a variable named 'population_weights'
population_weights = psdf

# Display the 'population_weights' dataframe
population_weights

# import random

# Extract the family sizes and percentages from the 'population_weights' dataframe
family_sizes = population_weights['family_size'].tolist()
family_weights = population_weights['percentage'].tolist()

# Iterate over the 'df' dataframe
for index, row in df.iterrows():
    # Sample a family size from the categorical distribution defined by the family sizes and percentages
    family_size = random.choices(family_sizes, family_weights)[0]

    # Assign the same 'Family_ID' value to the members of the same family
    df.at[index, 'Family_ID'] = index // family_size + 1

df = df.sort_values('Family_ID')

# Display the resulting dataframe
df

# import pandas as pd

# Find the maximum 'Family_ID' value
max_family_id = df['Family_ID'].max()

# Replace the missing values in the 'Family_ID' column with a sequence of integers from 1 to the maximum 'Family_ID' value
df['Family_ID'] = df['Family_ID'].fillna(pd.Series(range(1, max_family_id+1)))

# Display the resulting dataframe
df

# Count the number of unique 'Family_ID' values in the 'df' dataframe
num_families = df['Family_ID'].nunique()

# Print the result
print(f"Number of families: {num_families}")

# import numpy as np
# import random


# https://www.kathimerini.gr/economy/562220620/eurostat-sto-11-4-i-anergia-ton-noemvrio/
UNEMPLOYMENT_RATE = 0.114

# Create a new column in the 'df' dataframe called 'Work_Status'
df['Work_Status'] = np.nan

# Assign a 'Work_Status' value of -1 to members of the population under 18 years old
df['Work_Status'] = np.where(df['Age'] < 18, -1, df['Work_Status'])

# Assign a 'Work_Status' value of -2 to members of the population over 65 years old
df['Work_Status'] = np.where(df['Age'] > 65, -2, df['Work_Status'])

# Iterate over the 'df' dataframe
for index, row in df.iterrows():
    # Generate a random number between 0 and 1
    rnd = random.random()

    # If the random number is less than the unemployment rate and the 'Work_Status' value is 'NaN', assign a 'Work_Status' value of 0 (unemployed)
    if (rnd < UNEMPLOYMENT_RATE) & np.isnan(df.at[index, 'Work_Status']):
        df.at[index, 'Work_Status'] = 0
    # Otherwise, if the 'Work_Status' value is 'NaN', assign a 'Work_Status' value of 1 (employed)
    elif np.isnan(df.at[index, 'Work_Status']):
        df.at[index, 'Work_Status'] = 1

# Cast the 'Work_Status' column to the 'int' type
df['Work_Status'] = df['Work_Status'].astype(int)


# Display the resulting dataframe
df

"""assign each member a school status based on the following rules:
if 'Age' >= 18 then 'School_Status' is 0
else if 'Age' >= 15 then 'School_Status' is 1
else if 'Age' >= 12 then 'School_Status' is 2
else if 'Age' >= 6 then 'School_Status' is 3
else if 'Age' >= 4 then 'School_Status' is 4
else if 'Age' >= 2 then 'School_Status' is 5
else 'School_Status' is 6

"""

# import numpy as np

# Create a new column in the 'df' dataframe called 'School_Status'
df['School_Status'] = np.nan

# Assign a 'School_Status' value of 0 to members of the population 18 years old or older
df['School_Status'] = np.where(df['Age'] >= 18, 0, df['School_Status'])

# Assign a 'School_Status' value of 1 to members of the population 15 to 17 years old
df['School_Status'] = np.where((df['Age'] >= 15) & (df['Age'] < 18), 1, df['School_Status'])

# Assign a 'School_Status' value of 2 to members of the population 12 to 14 years old
df['School_Status'] = np.where((df['Age'] >= 12) & (df['Age'] < 15), 2, df['School_Status'])

# Assign a 'School_Status' value of 3 to members of the population 6 to 11 years old
df['School_Status'] = np.where((df['Age'] >= 6) & (df['Age'] < 12), 3, df['School_Status'])

# Assign a 'School_Status' value of 4 to members of the population 4 to 5 years old
df['School_Status'] = np.where((df['Age'] >= 4) & (df['Age'] < 6), 4, df['School_Status'])

# Assign a 'School_Status' value of 5 to members of the population 2 to 3 years old
df['School_Status'] = np.where((df['Age'] >= 2) & (df['Age'] < 4), 5, df['School_Status'])

# Assign a 'School_Status' value of 6 to members of the population under 2 years old
df['School_Status'] = np.where(df['Age'] < 2, 6, df['School_Status'])

df['School_Status'] = df['School_Status'].astype(int)

# Display the resulting dataframe
df

# Create a new column in the 'df' dataframe called 'Work_ID'
# df['Work_ID'] = np.nan

# Assign a 'Work_ID' value of NaN to members of the population with a 'Work_Status' value of -2, -1, or 0
df['Work_ID'] = np.where((df['Work_Status'] == -2) | (df['Work_Status'] == -1) | (df['Work_Status'] == 0), np.nan, df['Work_ID'])

df

# Calculate the total number of employees in the 'df' dataframe
total_num_employees = df[df['Work_Status'] == 1]['Work_Status'].sum()

# Print the total number of employees
print(f'Total number of employees: {total_num_employees}')

"""create a new dataframe wdf
Poisson distribution of workers in workplaces based on number / workplace
we make these assumptions:
i. average employee 4.1 arbitrary
ii. minimum number of employees: 1 and maximum: 15 arbitrary
iii. the total number of employees is equal to total_num_employees
"""

# import numpy as np
# import pandas as pd

# Set the mean number of employees per workplace
mean_employees_per_workplace = 4.1

# Set the minimum and maximum number of employees per workplace
min_employees_per_workplace = 1
max_employees_per_workplace = 15

# Calculate the number of workplaces based on the total number of employees and the mean number of employees per workplace
num_workplaces = int(total_num_employees / mean_employees_per_workplace)

# Generate random numbers that follow a Poisson distribution with the mean number of employees per workplace
employees_per_workplace = np.random.poisson(mean_employees_per_workplace, size=num_workplaces)

# Ensure that the number of employees per workplace is within the specified range
employees_per_workplace = np.clip(employees_per_workplace, min_employees_per_workplace, max_employees_per_workplace)

# Create the 'wdf' dataframe with the number of employees per workplace
wdf = pd.DataFrame({'Workplace_ID': np.arange(num_workplaces), 'Num_Employees': employees_per_workplace})

# Display the 'wdf' dataframe
wdf

# Calculate the total number of employees in the 'wdf' dataframe
total_num_employees = wdf['Num_Employees'].sum()

# Print the total number of employees
print(f'Total number of employees: {total_num_employees}')

# import pandas as pd
# Create a list of the possible school IDs
school_ids = [i for i in range(1, TOTAL_SCHOOLS+1)]

# Create a list of the possible work IDs
work_ids = [i for i in range(1, NUM_WORKPLACES+1)]

# Initialize the school and work IDs for each person to be 'None'
df['School_ID'] = None
df['Work_ID'] = None

# Set the school and work IDs for each person based on their age and work status
for i, row in df.iterrows():
    if row['Work_Status'] == 1:
    # For working people, assign a random work ID
        df.at[i, 'Work_ID'] = random.choice(work_ids)
        df.at[i, 'School_ID'] = None

    elif row['Age'] < 18:
    # For people under 18, assign a random school ID
        df.at[i, 'School_ID'] = random.choice(school_ids)
        df.at[i, 'Work_ID'] = None

    else:
    # For people over 65, assign a 'pension' work ID
        df.at[i, 'Work_ID'] = None
        df.at[i, 'School_ID'] = None
df

df[df['School_ID'] == 77].head(10)

# Check that all people have a Work_ID or School_ID
print((df['Work_ID'].isnull() & df['School_ID'].isnull()).any())

# Count the number of people with each Work_ID
work_id_counts = df['Work_ID'].value_counts()

# Print the count of people with each Work_ID
print(work_id_counts)

# Count the number of people with each School_ID
school_id_counts = df['School_ID'].value_counts()

# Print the count of people with each School_ID
print(school_id_counts)



df.to_csv('df.csv', index=False)
df

df.describe()
