
import os
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import RMSprop
import pylab
import tkinter


def A2CModels(input_shape, action_space, lr):

    X_input = Input(shape=input_shape)
    X = Flatten(input_shape=input_shape)(X_input)

    X_hid = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)
    X_hid2 = Dense(32, activation="elu", kernel_initializer='he_uniform')(X_hid)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X_hid2)
    value = Dense(1, kernel_initializer='he_uniform')(X_hid2)

    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=lr))

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(learning_rate=lr))

    return Actor, Critic



class A2C_agent:
    def __init__(self):
        # -- RL AGENT --
        #Input: TC variables + outdoor humidity and temperature + time of day (3) 
        #Output: 10 possible actions for the fan mass flow rate
        self.state_size = (9,1)
        self.action_size = 10
        self.lr = 0.0001

        self.Actor, self.Critic = A2CModels(input_shape = self.state_size, action_space = self.action_size, lr=self.lr)

        # Instantiate games and plot memory
        self.states, self.actions, self.rewards = [], [], []
        self.scores, self.episodes, self.average = [], [], []
        self.EPISODES = 1000
        self.max_average = -99999999999 #To save the best model
        self.Save_Path = 'Models'

        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A2C_{}'.format("SDU_Building", self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)
        self.state = None
    
        self.time_of_day = None

    def remember(self, queue):
        try:
            mem = queue.get(block = True, timeout = 2)
            self.states.append(mem[0])
            action_onehot = np.zeros([self.action_size])
            action_onehot[mem[1]] = 1
            self.actions.append(action_onehot)
            self.rewards.append(mem[2])
        except:
            #After the queue is emptied and no info comes after two seconds
            pass

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        state = np.expand_dims(state, axis=0) #Need to add a dimension to the state to make it a 2D array                 
        prediction = self.Actor(state)
        action = np.random.choice(self.action_size, p=np.squeeze(prediction))#Squeeze to remove the extra dimension
        return action
        
    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    def replay(self):
        # reshape memory to appropriate shape for training
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        # Compute discounted rewards
        discounted_r = self.discount_rewards(self.rewards)
        # Get Critic network predictions
        values = self.Critic.predict(states)[:, 0]
        # Compute advantages
        advantages = discounted_r - values
        # training Actor and Critic networks
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
        # reset training memory
        self.states, self.actions, self.rewards = [], [], []

    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        #self.Critic.save(self.Model_name + '_Critic.h5')
    

    def evaluate_model(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))


        #if str(episode)[-1:] == "0":# much faster than episode % 100
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.plot(self.episodes, self.average, 'r')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        try:
            pylab.savefig(self.path+".png")
        except OSError:
            pass
        
        # saving best models
        if self.average[-1] >= self.max_average:
            self.max_average = self.average[-1]
            self.save()
            SAVING = "SAVING"
        else:
            SAVING = ""
        print("episode: {}/{}, score: {}, average: {:.2f} {}".format(episode, self.EPISODES, score, self.average[-1], SAVING))

        return self.average[-1]
