#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd
from hmmlearn import hmm
import yfinance as yf
import matplotlib.pyplot as plt

# Gathering historical Bitcoin prices via 
BTC_Ticker = yf.Ticker('BTC-USD')
BTC_Data = BTC_Ticker.history(period='max')

bitcoin_prices = BTC_Data['Close'].values

# Converting each value in the bitcoin_prices list to a 0 or 1 depending if it is lower or higher than the next value
for i in range(len(bitcoin_prices)-1):
    bitcoin_prices[i] = 0 if bitcoin_prices[i] <= bitcoin_prices[i+1] else 1

# Removing the last element in the bitcoin_prices since it cannot be compared with the next value, which is nonexistent; converting all floats to ints
bitcoin_prices = bitcoin_prices[:-1].astype(int)

# Define the state space; 0 = Falling and 1 = Rising
states = ["Falling", "Rising"]
n_states = len(states)
print('Number of hidden states :',n_states)

# Define the observation space; falling price action is bearish while rising price action is bullish
observations = ["Bear", "Bull"]
n_observations = len(observations)
print('Number of observations :',n_observations)

# Define the initial state distribution; since Bitcoin has risen drastically over time, I'm using a 20% and 80% split here as an example
state_probability = np.array([0.2, 0.8]) 
print("State probability:\n", state_probability)

# Define the state transition probabilities; again, these are just made up accounting for Bitcoin's historically rising price
transition_probability = np.array([[0.6, 0.4], # Falling to Rising, Falling to Falling
                                   [0.4, 0.6]])   # Rising to Falling, Rising to Rising
print("\nTransition probability:\n", transition_probability)

# Define the observation likelihoods
emission_probability= np.array([[0.9, 0.1],
                                [0.2, 0.8]])
print("\nEmission probability:\n", emission_probability)

# Creating the Hidden Markov Model object using the CategoricalHMM class, passing the number of states (which is 2 in this instance)
model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = state_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# Reshaping the sequence of operations (bitcoin price action via bitcoin_prices) so that it can be passed to the model
bitcoin_prices = bitcoin_prices.reshape(-1,1)

# Predict the most likely sequence of hidden states (falling or rising), which is the same length as the bitcoin_prices list 
hidden_states = model.predict(bitcoin_prices)
print("Most likely hidden states:", hidden_states)

# Decode the prediction using the "Viterbi" algorithm to get the log probability; the hidden_states value returned here is the same as above
log_probability, hidden_states = model.decode(bitcoin_prices,
                                              lengths = len(bitcoin_prices),
                                              algorithm ='viterbi' )

# Typically, a higher log probability means the quality of a model fit is better
print('Log Probability :',log_probability)


# In[75]:


# Pulling a 10-day prediction from the model, which uses probabilstic learning to determine if the price action will be falling or rising each day
# Traders at a financial firm, for example, could use this information to help guide decisions
future_observations, _ = model.sample(n_samples=10, random_state=42)
print(future_observations)


# In[79]:


# Plotting the price action every 30 days (roughly monthly); there are too many datapoints to plot each day individually
# This simulates the price action chart; it is not purely accurate since it is just comparing each month start versus month end
bitcoin_prices_compressed = bitcoin_prices[::30]

# Create a time axis (x-axis) from 0 to the length of the binary data
time_steps = np.arange(len(bitcoin_prices_compressed))

# Create a line plot to visualize the binary data over time
plt.figure(figsize=(10, 4))
plt.plot(time_steps, bitcoin_prices_compressed, marker='o', linestyle='-')
plt.xlabel("Months")
plt.ylabel("Likelihood of Bitcoin Price Falling or Rising")
plt.title("Bitcoin Price Action Likelihood Time Series")
plt.grid(True)
plt.show()

