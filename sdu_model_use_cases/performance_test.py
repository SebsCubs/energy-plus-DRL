import time
start = time.time()
import numpy as np
from keras.models import load_model



model_path = r'C:\Projects\SDU\Thesis\pyenergyplus\sdu_model_use_cases\Models\SDU_Building_A2C_0.001_Actor.h5'

class Actor:
    def __init__(self):
        self.Actor = self.load(model_path)
        self.action_size = 10

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        state = np.expand_dims(state, axis=0) #Need to add a dimension to the state to make it a 2D array                 
        prediction = self.Actor(state)
        action = np.random.choice(self.action_size, p=np.squeeze(prediction))#Squeeze to remove the extra dimension
        return action
        

    def load(self, Actor_name):
        return load_model(Actor_name, compile=False)


if __name__=="__main__":
    
    actor = Actor()
    print("Actor loaded...")
    end = time.time()
    print("Time to load actor: ", end - start)
    #Perform 1000 predictions and measure the time
    start = time.time()
    
    for i in range(1000):
        action = actor.act(np.random.rand(9))
        #print(action)
    end = time.time()
    print("Time to perform 1000 predictions: ", end - start)

