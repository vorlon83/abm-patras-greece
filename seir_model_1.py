import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random

import timeit
start = timeit.default_timer()

BETA_FAMILY = 0.8
BETA_WORK = 0.1
BETA_SCHOOL = 0.04
BETA_RANDOM = 0.01
GAMMA = 0.1
SIGMA = 0.2

BETA_SAME_AGE = 0.0005

# ELDERDLY_FACTOR = 0.1

AGE_GROUPS = {
    '0-4': {
        'contacts_per_day': 10.21,
        'interaction_probability': 0.093
    },
    '5-9': {
        'contacts_per_day': 14.81,
        'interaction_probability': 0.135
    },
    '10-14': {
        'contacts_per_day': 18.22,
        'interaction_probability': 0.166
    },
    '15-19': {
        'contacts_per_day': 17.58,
        'interaction_probability': 0.160
    },
    '20-29': {
        'contacts_per_day': 13.57,
        'interaction_probability': 0.124
    },
    '30-39': {
        'contacts_per_day': 14.14,
        'interaction_probability': 0.129
    },
    '40-49': {
        'contacts_per_day': 13.83,
        'interaction_probability': 0.126
    },
    '50-59': {
        'contacts_per_day': 12.30,
        'interaction_probability': 0.112
    },
    '60-69': {
        'contacts_per_day': 9.21,
        'interaction_probability': 0.084 #* ELDERDLY_FACTOR
    },
    '70+': {
        'contacts_per_day': 6.89,
        'interaction_probability': 0.063 #* ELDERDLY_FACTOR
    }
}

total_steps = 180

class SEIRAgent(Agent):
    def __init__(self, unique_id, age, gender, family_id, work_id, school_id, infection_status, model):
        super().__init__(unique_id, model)
        self.age = age
        self.gender = gender
        self.family_id = family_id
        self.work_id = work_id
        self.school_id = school_id
        self.infection_status = infection_status
        self.next_status = infection_status

    def step(self):
        if self.infection_status == 'E':
            if random.random() < SIGMA:
                self.next_status = 'I'

        if self.infection_status == 'I':
            family_members = [a for a in self.model.schedule.agents if a.family_id == self.family_id]
            for family_member in family_members:
                if family_member.infection_status == 'S' and family_member.next_status != 'E' and random.random() < BETA_FAMILY:
                    family_member.next_status = 'E'

            work_members = [a for a in self.model.schedule.agents if a.work_id == self.work_id and a.work_id != -999]
            for work_member in work_members:
                if work_member.infection_status == 'S' and work_member.next_status != 'E' and random.random() < BETA_WORK:
                    work_member.next_status = 'E'

            school_members = [a for a in self.model.schedule.agents if a.school_id == self.school_id and a.school_id != -999]
            for school_member in school_members:
                if school_member.infection_status == 'S' and school_member.next_status != 'E' and random.random() < BETA_SCHOOL:
                    school_member.next_status = 'E'

            if random.random() < BETA_RANDOM:
                random_agent = random.choice(self.model.schedule.agents)
                if random_agent.infection_status == 'S':
                    if random.random() < BETA_RANDOM:
                        random_agent.next_status = 'E'
                else:  # i.e. 'R' or 'I' or 'E'
                    pass

            # New layer of random interactions based on age
            age_group = self._get_age_group(self.age)
            contacts_per_day = AGE_GROUPS[age_group]['contacts_per_day']
            interaction_probability = AGE_GROUPS[age_group]['interaction_probability']

            if random.random() < (BETA_SAME_AGE * contacts_per_day):
                same_age_agents = [a for a in self.model.schedule.agents if a.age == self.age and a.unique_id != self.unique_id]
                for agent in same_age_agents:
                    if agent.infection_status == 'S' and agent.next_status != 'E' and random.random() < interaction_probability:
                        agent.next_status = 'E'

            if random.random() < GAMMA:
                self.next_status = 'R'

        else:  # i.e. 'S' or 'R'
            pass

        return None

    def _get_age_group(self, age):
        for age_group, age_range in AGE_GROUPS.items():
            if age_group == '70+':
                return '70+'  # Return the special age group label directly
            age_min, age_max = map(int, age_group.split('-'))
            if age_min <= age <= age_max:
                return age_group
        return '70+'  # If age is greater than 70, assign to the last age group


class SEIRModel(Model):
    def __init__(self):
        self.schedule = RandomActivation(self)
        self.running = True

    def step(self):
        agents = self.schedule.agents
        for agent in agents:
            agent.next_status = agent.infection_status

        # Call the 'step' method of all agents
        self.schedule.step()

        # Copy 'next_status' to 'infection_status' for all agents
        for agent in agents:
            agent.infection_status = agent.next_status

        # Collect data after step
        # data_collector.collect(self)


def compute_S(model):
    agents = model.schedule.agents
    return np.sum([agent.infection_status == 'S' for agent in agents])


def compute_E(model):
    agents = model.schedule.agents
    return np.sum([agent.infection_status == 'E' for agent in agents])


def compute_I(model):
    agents = model.schedule.agents
    return np.sum([agent.infection_status == 'I' for agent in agents])


def compute_R(model):
    agents = model.schedule.agents
    return np.sum([agent.infection_status == 'R' for agent in agents])


# Create a data collector
data_collector = DataCollector(
    model_reporters={
        'Susceptible': compute_S,
        'Exposed': compute_E,
        'Infected': compute_I,
        'Recovered': compute_R
    }
)

# Create an instance of the model
df = pd.read_csv('df_88225.csv')
model = SEIRModel()
for i, row in df.iterrows():
    a = SEIRAgent(i, row['Age'], row['Gender'], row['Family_ID'], row['Work_ID'], row['School_ID'], row['Infection_Status'], model)
    model.schedule.add(a)

# Add data collector to the model
model.datacollector = data_collector

# Create an empty list to store intermediate states at each step
intermediate_data = []

# Run the model for 'total_steps' time steps
for i in range(total_steps):
    # Collect data before step
    data_collector.collect(model)

    # Run a single step of the model
    model.step()

    # Store intermediate data in the list
    intermediate_data.append({
        'Step': i,
        'Susceptible': compute_S(model),
        'Exposed': compute_E(model),
        'Infected': compute_I(model),
        'Recovered': compute_R(model)
    })

    print("Step:", i)
    print("S:", compute_S(model))
    print("E:", compute_E(model))
    print("I:", compute_I(model))
    print("R:", compute_R(model))

    print('Time elapsed:', timeit.default_timer() - start)

# Get the data as a Pandas DataFrame
df1 = data_collector.get_model_vars_dataframe()

# Convert the list to a DataFrame
intermediate_data_df = pd.DataFrame(intermediate_data)

# Save the intermediate data to a CSV file
intermediate_data_df.to_csv('scenario_1.csv', index=False)

stop = timeit.default_timer()
print('Total time:', stop - start)

# Plot the number of susceptible, exposed, infected, and recovered individuals over time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df1['Susceptible'], label='Susceptible')
ax.plot(df1['Exposed'], label='Exposed')
ax.plot(df1['Infected'], label='Infected')
ax.plot(df1['Recovered'], label='Recovered')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number of individuals')
ax.set_title('SEIR Model')
ax.legend()
plt.show()