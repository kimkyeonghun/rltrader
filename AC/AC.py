import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization, Flatten, Dropout
from keras.optimizers import sgd, Adam
from keras import backend as K

K.set_learning_phase(1) 

class ACagent:
    def __init__(self,input_dim=0,output_dim=0,lr=0.001):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.action_size = output_dim
        
        self.DF = 0.98
        self.actor_lr = lr
        self.critic_lr = lr

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()
        
    def reset(self):
        self.prob = None

    def build_actor(self):
        
        actor = Sequential()

        actor.add(Dense(128,input_dim=self.input_dim))
        actor.add(Dropout(0.5))
        actor.add(BatchNormalization())
        actor.add(Dense(128))
        actor.add(Dropout(0.5))
        actor.add(BatchNormalization())
        actor.add(Dense(128))
        actor.add(Dropout(0.5))
        actor.add(BatchNormalization())
        actor.add(Dense(self.output_dim))
        actor.add(Activation('softmax'))

        return actor

    def build_critic(self):

        critic = Sequential()

        critic.add(Dense(128,input_dim=self.input_dim))
        critic.add(Dropout(0.5))
        critic.add(BatchNormalization())
        critic.add(Dense(128))
        critic.add(Dropout(0.5))
        critic.add(BatchNormalization())
        critic.add(Dense(128))
        critic.add(Dropout(0.5))
        critic.add(BatchNormalization())
        critic.add(Dense(1,activation='linear'))

        return critic

    def predict(self,sample):
        self.prob = self.actor.predict(np.array(sample).reshape(1,self.input_dim))[0]
        return self.prob

    def actor_optimizer(self):
        action = K.placeholder(shape=[None,self.action_size])
        advantage = K.placeholder(shape=[None,])

        action_prob = K.sum(action * self.actor.output,axis=1)
        cross_entropy = K.log(action_prob)*advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights,[],loss)
        train = K.function([self.actor.input,action,advantage],[],updates=updates)

        return train

    def critic_optimizer(self):
        target = K.placeholder(shape=[None,])

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights,[],loss)
        train = K.function([self.critic.input,target],[],updates=updates)

        return train

    def train_model(self,state,action,reward,next_state):
        value = self.critic.predict(np.array(state).reshape(1,self.input_dim))[0]
        next_value = self.critic.predict(np.array(next_state).reshape(1,self.input_dim))[0]

        act = np.zeros([1,self.action_size])
        act[0][action] = 1

        advantage = (reward+self.DF*next_value)-value
        target = reward + self.DF*next_value

        state=np.array(state).reshape(1,self.input_dim)

        self.actor_updater([state,np.array(act),np.array(advantage)])
        self.critic_updater([state,target])

    def save_model(self, model_path):
        if model_path is not None and self.actor is not None:
            self.actor.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.actor.load_weights(model_path)
