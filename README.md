# CS 229 project

## Introduction

This is a joint-work by Caitlin Stanton (stanton1 'at' stanford.edu) and Beite Zhu (jupiterz 'at' stanford.edu) for the CS229 2018 Fall final project. This work implements a neural network to run the deep Q learning algorithm on the Lunar Lander arcade game and then adapts this model to instead run on stock data.  Our agent learns---on stock data from tech companies such as Google, Apple, and Facebook---when it should buy or sell a stock, given features related to the recent stock price history.  Furthermore, our model allows the agent to opt to buy and sell smaller amounts of the stock instead of larger amounts, which increases the nuance and complexity of our model's decision-making.

## Motivation

Reinforcement learning has been at the core of many of the most exciting recent developments in AI.  For example, while computers have been relatively successful at playing chess for many years--notably, the computer Deep Blue was able to defeat the reigning world chess champion Garry Kasparov in 1996--the game of Go was considered much harder; it wasn't until reinforcement learning techniques were used in 2015 that the program AlphaGo was finally able to beat a professional human Go player. 

Here we use deep Q learning to train an agent to learn the arcade game Lunar Lander, a game where the goal is to steer a landing module and successfully land it on the surface of the moon.  After understanding how to apply this model to Lunar Lander, we then use this same technique to a less conventional application of reinforcement learning techniques: investing in the stock market.  We rephrase stock market investment as a game where states involve data such as recent change and volatility of a stock, and discretize the possible investment amounts in order to create a finite set of actions at each state.  In this way, we apply our deep Q learning algorithm from our Lunar Lander model to the problem of stock market prediction.

## StockAgent Model

Our stock prediction agent deploys the following structure:

![Alt text](image/model_graph.PNG?raw=true "layers SADQN")

with less than 100 epochs of training, on the stock of AAPL and GOOG we have the following results:

![Alt text](image/merged_baseline.png?raw=true "SADQN result")

Our model can return 500-2000 dollars of profit in the course of 5 years on a strong stock. The majority of its actions consist of soft buys, and the agent will go into negative cash budget, but with a sizable stock value thus resulting in a positive portfolio.


## Conclusion

By drawing connections between the game of Lunar Lander and stock investment, we have established a baseline structure of a stock predicting agent using the model of deep Q learning. The model is demonstrated to be rather risk averse but can master long term investment strategy with reasonably volatile stocks. 

##More details

Please refer to our poster and report in the course_related folder.

## LunarLander Model

We have a base line policy gradient model that has the following structure:

![Alt text](image/layers.PNG?raw=true "layers")

Here's its result from episode 200, 500 and 1000.

![Alt text](image/episode-200.gif?raw=true "episode 200") 
![Alt text](image/episode-500.gif?raw=true "episode 500") 



