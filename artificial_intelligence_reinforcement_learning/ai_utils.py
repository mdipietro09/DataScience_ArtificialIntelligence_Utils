
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import gym
from tensorflow.keras import models, layers, optimizers, BatchNormalization
import collections
from PIL import Image



###############################################################################
#                 ENVIRONMENT                                                 #
###############################################################################
'''
'''
def get_gym_envs(lst_search=[]):
    lst_envs = [env for env in gym.envs.registry.all()]
    dtf_envs = pd.DataFrame( [{"id":env.id, "from":env._entry_point} for env in lst_envs] )
    dtf_envs["from"] = dtf_envs["from"].apply(lambda x: x[:x.find(":")].replace("gym.envs.", ""))
    dtf_envs["from"] = dtf_envs["from"].apply(lambda x: x[:x.find(".")] if "." in x else x )
    if len(lst_search) >0 :
       dtf_envs = dtf_envs[ dtf_envs["id"].str.lower().str.contains('|'.join(lst_search)) ]
    return dtf_envs



'''
'''
def get_env_space(env):
    ## states
    states_space = env.observation_space.shape
    if states_space is None:
        states_space = 0
    else:
        states_space = states_space[0]
    ## actions
    actions_space = env.action_space.n
    return {"states_space":states_space, "actions_space":actions_space}



'''
'''
def env_random_play(env, episodes=10):
    lst_logs = []
    for episode in range(episodes):
        print(episode+1)
        env.reset()
        done = False
        step = 0
        while not done:
            step += 1
            state, reward, done, info = env.step( env.action_space.sample() )
            goal = utils_env_goal_achieved(env, state)
            lst_logs.append( (episode+1, step, state, reward, done, info, goal) )
            env.render()
    env.close()
    dtf_logs = pd.DataFrame(lst_logs, columns=['episode','steps','state','reward','done','info','goal'])
    return dtf_logs



'''
'''
def utils_env_get_img(env, state):
    if "MountainCar" in str(env.env):
        array = np.reshape(state, [1, len(state)])
        img = Image.fromarray(array, 'RGB')
        return img
    
    

'''
Objective of the agent in the env
'''
def utils_env_goal_achieved(env, state):
    if "MountainCar" in str(env.env):
        return True if state[0] >= env.goal_position else False
    


###############################################################################
#                 TRAIN / TEST                                                #
###############################################################################
'''
'''
def utils_plot_stats(dic_rewards, figsize=(20,13)):
    fig, ax = plt.subplots(figsize=figsize, nrows=2, ncols=1, sharex=False, sharey=False)
    fig.suptitle("Stats", fontsize=20)
    
    ax[0].plot(dic_rewards['episode'], dic_rewards['mean'], label="mean")
    ax[0].plot(dic_rewards['episode'], dic_rewards['max'], label="max")
    ax[0].plot(dic_rewards['episode'], dic_rewards['min'], label="min")
    ax[0].legend(loc=4)
    ax[0].set_title('Rewards', fontsize=15)
    
    ax[1].plot(dic_rewards["completion"])
    ax[1].set_title('Completions', fontsize=15)
    plt.show()
   
    
    
'''
''' 
def trainAI(env, ai, episodes=1000, min_eploration=0.001, min_replay_memory_size=1000, plot_rate=1):
    ## set stuff
    render = True
    dic_rewards = {'episode':[], 'mean':[], 'max':[], 'min':[], "completion":[]}
    lst_rewards, lst_completion, lst_logs = [], [], []
    
    ## episodes loop
    for episode in range(episodes):
        ### reset
        current_state = ai.preprocess_states(state=env.reset(), env=env)
        episode_reward, episode_complete = 0, 0
        
        ### steps loop
        done = False
        step = 0
        while not done:
            step += 1
            
            #### select next action and get reward
            action = ai.get_next_action(current_state)
            new_state, reward, done, info = env.step(action)          
            goal = utils_env_goal_achieved(env, new_state)
            new_state = ai.preprocess_states(state=new_state, env=env)
            episode_reward += reward
            lst_logs.append( (episode+1, step, current_state, reward, done, info, goal) )
            if render == True:
                env.render()
            
            #### update Q
            if not done:
                if ai.name == "Q":
                    ai.update_Q(current_state, action, reward, new_state, goal)
                elif ai.name == "DQN":
                    ai.update_Q()
            
            #### goal achieved ?
            elif goal:
                episode_complete = episode_complete + 1
                print("!!! Completed at episode", episode, ", reward:", reward, ", cum:", episode_reward)
            
            #### next ep
            current_state = new_state
          
        ### append reward
        lst_rewards.append(episode_reward)
        lst_completion.append(episode_complete)
        
        ### decay epsilon
        ai.decay_exploration(min_eploration)
        
        ### plot and stats settings
        render = True if episode % plot_rate == 0 else False
        if episode % plot_rate == 0:
            dic_rewards["episode"].append(episode)
            dic_rewards["mean"].append( np.mean(lst_rewards[-episode:]) )
            dic_rewards["max"].append( max(lst_rewards[-episode:]) )
            dic_rewards["min"].append( min(lst_rewards[-episode:]) )
            dic_rewards["completion"].append( sum(lst_completion[-episode:]) )
            print(f'episode: {episode}, n completions: {sum(lst_completion[-episode:])}, mean reward: {np.mean(lst_rewards[-episode:])}, explore rate: {round(ai.explore_rate,4)}')
        
    ## plot stats
    utils_plot_stats(dic_rewards)
    env.close()
    
    ## return logs dtf
    dtf_logs = pd.DataFrame(lst_logs, columns=['episode','steps','state','reward','done','info','goal'])
    return dtf_logs



'''
'''
def testAI(env, ai, start_state):
    current_state = ai.preprocess_states(start_state, env=env)
    done = False
    while not done:
        action = ai.get_next_action(current_state)
        new_state, reward, done, info = env.step(action)
        new_state = ai.preprocess_states(state=new_state, env=env)
        env.render()
        current_state = new_state
    env.close()
     
    
    
###############################################################################
#                 Q-LEARNING                                                  #
###############################################################################
'''
'''
class agentQ():
    
    def __init__(self, states_space, actions_space, explore_rate=1, explore_decay=0.99, 
                 discount_rate=0.8, learning_rate=0.1, batch_size=20):
        self.name = "Q"
        ## env params
        self.states_space = states_space
        self.actions_space = actions_space  
        ## learning params
        self.explore_rate = explore_rate
        self.explore_decay = explore_decay
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        ## create Q
        self.Q = np.array( np.zeros(shape=( [self.batch_size]*self.states_space + [self.actions_space] )) )
        print(self.name)
        print("--- env params ---")
        print("states_space:", self.states_space, ",actions_space:", self.actions_space)
        print("--- learning params ---")
        print("explore_rate:", self.explore_rate, ",explore_decay:", self.explore_decay, ",discount_rate:", self.discount_rate, 
              ",learning_rate:", self.learning_rate, ",batch_size:", self.batch_size)


    def preprocess_states(self, state, env):
        high, low = env.observation_space.high, env.observation_space.low
        states_space_size = [self.batch_size]*self.states_space
        dim_space_window = (high - low) / states_space_size
        state_preprocessed = (state - low) / dim_space_window
        state_preprocessed = tuple(state_preprocessed.astype(np.int))
        return state_preprocessed
    
    
    def get_next_action(self, current_state):            
        ## Explore: select a random action
        if np.random.uniform(0,1) < self.explore_rate:
            action = np.random.randint(0, self.actions_space)
        ## Exploit: select the action with max q_value
        else:
            current_qs = self.Q[current_state]
            action = np.argmax( current_qs )
        return action
    
    
    def decay_exploration(self, min_eploration):
        if self.explore_rate > min_eploration:
            epsilon_decayed = self.explore_rate * self.explore_decay
            self.explore_rate = max(min_eploration, epsilon_decayed)
    
    
    def update_Q(self, current_state, action, reward, new_state, goal):
        ## comupte new_q_value
        if not goal:
            new_qs = self.Q[new_state]
            max_q_value = np.max( new_qs )
            current_q_value = self.Q[current_state + (action, )]
            new_q_value = ((1-self.learning_rate) * current_q_value) + (self.learning_rate * (reward + (self.discount_rate * max_q_value)))
        else:
            new_q_value = reward
        ## update
        self.Q[current_state + (action, )] = new_q_value
        
    

###############################################################################
#                      DEEP Q-LEARNING                                        #
###############################################################################
'''
Replace the Q-table with a neural network that approximates q_values.
ML compares y_pred to y_true (which is constant throughout the entire training).
In DQN both y_pred and y_true are predicted by the network, therefore might vary at every iteration.
The agent has a memory, samples randomly from this memory to train regression ANN and predicts q_values.
'''
class agentDQN():
    
    def __init__(self, states_space, actions_space, explore_rate=1, explore_decay=0.99, 
                 discount_rate=0.8, learning_rate=0.01, memory_size=50000, batch_size=64, min_replay_memory_size=1000):
        self.name = "DQN"
        ## env params
        self.states_space = states_space
        self.actions_space = actions_space  
        ## learning params
        self.explore_rate = explore_rate
        self.explore_decay = explore_decay
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.min_replay_memory_size = min_replay_memory_size
        ## create models
        self.memory = collections.deque(maxlen=memory_size)
        self.model = self.dqn()
        print(self.name)
        print("--- env params ---")
        print("states_space:", self.states_space, ",actions_space:", self.actions_space)
        print("--- learning params ---")
        print("explore_rate:", self.explore_rate, ",explore_decay:", self.explore_decay, ",discount_rate:", self.discount_rate, 
              ",learning_rate:", self.learning_rate, ",memory_size:", memory_size, ",batch_size:", self.batch_size)

    
    def preprocess_states(self, state, env):
        state_preprocessed = np.reshape(state, [1, self.states_space])
        return state_preprocessed
    
    
    def dqn(self):
        try:
            ## layer 1 - fully connected
            model = models.Sequential()
            model.add( layers.Dense(input_shape=(self.states_space,), units=64, activation="relu") )
            model.add( layers.BatchNormalization() )
            model.add( layers.Dropout(0.2) )
            ## layer 2 - fully connected
            model.add( layers.Dense(units=24, activation="relu") )
            model.add( layers.BatchNormalization() )
            model.add( layers.Dropout(0.2) )
            ## layer output - linear: calculate qs for each ation
            model.add( layers.Dense(units=self.actions_space, activation='linear') )            
            model.compile(loss="mse", optimizer=optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
            return model
        except Exception as e:
            print("--- got error in dqn model---")
            print(e)        
            
    
    def store_in_memory(self, tupla_experience):
        self.memory.append( tupla_experience )
        
    
    def get_next_action(self, current_state):            
        ## Explore: select a random action
        if np.random.uniform(0,1) < self.explore_rate:
            action = np.random.randint(0, self.actions_space)
        ## Exploit: select the action with max q_value
        else:
            current_qs = self.model.predict(current_state)[0]
            action = np.argmax( current_qs )
        return action
    
    
    def decay_exploration(self, min_eploration):
        if self.explore_rate > min_eploration:
            epsilon_decayed = self.explore_rate * self.explore_decay
            self.explore_rate = max(min_eploration, epsilon_decayed)
    
    
    def update_Q(self): 
        ## start only if there are enough experiences
        if len(self.memory) > self.min_replay_memory_size: 
            ## batch of random samples from the memory
            batch = random.sample(self.memory, self.batch_size)    
            ## replay memory loop
            lst_X = []
            lst_y = []
            for current_state, action, reward, new_state, goal in batch:
                ### comupte new_qs
                if not goal:
                    new_qs = self.model.predict(new_state)[0]
                    max_q_value = np.max( new_qs )
                    new_q_value = reward + self.discount_rate * max_q_value
                else:
                    new_q_value = reward
                ### update current_qs
                current_qs = self.model.predict(current_state)
                current_qs[0][action] = new_q_value
                ### dataset
                lst_X.append(current_state[0])
                lst_y.append(current_qs[0])
            ## fit
            X = np.array(lst_X)  #array: batch_size x states_space
            y = np.array(lst_y)  #array: batch_size x actions_space
            self.model.fit(x=X, y=y, batch_size=self.batch_size, epochs=1, verbose=0, shuffle=False)
              
          

###############################################################################
#                DOUBLE DEEP Q-LEARNING                                       #
###############################################################################
'''
Keep two copies of the Q Network, but only one is being updated, the other one remains still.
'''
class agentDDQN():
    
    def __init__(self, states_space, actions_space, img_size=20, img_rgb=3, img_scaler=255, conv_size=256, 
                 maxpool_size=(2,2), explore_rate=0.5, explore_decay=0.99, discount_rate=0.99, 
                 learning_rate=0.001, memory_size=50000, batch_size=64):
        self.name = "DDQN"
        ## env params
        self.states_space = tuple( [img_size]*states_space + [img_rgb] )
        self.actions_space = actions_space
        self.img_scaler = img_scaler
        ## model params
        self.conv_size = conv_size
        self.maxpool_size = maxpool_size
        ## learning params
        self.explore_rate = explore_rate
        self.explore_decay = explore_decay
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        ## create models
        self.memory = collections.deque(maxlen=memory_size)
        self.simple_model = self.conv_dqn()
        self.target_model = self.conv_dqn()
        self.target_model.set_weights( self.simple_model.get_weights() )
        print(self.name)
        print("--- env params ---")
        print("states_space:", self.states_space, ", actions_space:", self.actions_space, ", img_scaler:", self.img_scaler)
        print("--- model params ---")
        print("conv_size:", self.conv_size, ", maxpool_size:", self.maxpool_size)
        print("--- learning params ---")
        print("explore_rate:", self.explore_rate, ", explore_decay:", self.explore_decay, ", discount_rate:", self.discount_rate, ", learning_rate:", self.learning_rate, 
              ", memory_size:", memory_size, ", batch_size:", self.batch_size)   
              
    
    def conv_dqn(self):
        try:
            ## layer 1 - first convolution
            model = models.Sequential()
            model.add( layers.Conv2D(input_shape=self.states_space, filters=self.conv_size, kernel_size=(3,3), activation="relu") )
            model.add( layers.MaxPooling2D(pool_size=self.maxpool_size) )
            model.add( layers.Dropout(0.2) )
            ## layer 2 - second convolution
            model.add( layers.Conv2D(filters=self.conv_size, kernel_size=(3,3), activation="relu") )
            model.add( layers.MaxPooling2D(pool_size=self.maxpool_size) )
            model.add( layers.Dropout(0.2) )
            ## layer 3 - flattening and fully connected
            model.add( layers.Flatten() )
            model.add( layers.Dense(64) )
            ## layer output - linear: calculate qs for each ation
            model.add( layers.Dense(units=self.actions_space, activation='linear') )
            ## compile
            model.compile(loss="mse", optimizer=optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
            return model
        except Exception as e:
            print("--- got error in dqn model---")
            print(e)
    
    
    def store_in_memory(self, tupla_experience):
        self.memory.append( tupla_experience )
        
    
    def preprocess_imgs(self, lst_imgs):
        X = np.array(lst_imgs)
        X = X / self.img_scaler
        return X
    
    
    def get_next_action(self, current_state):            
        ## Explore: select a random action
        if np.random.uniform(0,1) < self.explore_rate:
            action = np.random.randint(0, self.actions_space)
        ## Exploit: select the action with max q_value
        else:
            X = np.array(current_state).reshape(-1, *current_state.shape) / self.img_scaler   #####<<<<<
            lst_current_qs = self.simple_model.predict(X)[0]  #####<<<<<
            action = np.argmax( lst_current_qs )
        return action
    
    
    def update_Q(self, done=True, update_target_model_rate=5): 
        ## batch of random samples from the memory
        batch = np.random.sample(self.memory, self.batch_size)
        
        ## process current_states and predict current q_values con il simple_model
        lst_current_states = [tupla_experience[0] for tupla_experience in batch]
        X = self.preprocess_imgs( lst_current_states )
        lst_current_qs = self.simple_model.predict(X)    #####<<<<<

        ## process new_states and predict new q_values con il target_model
        lst_new_states = [tupla_experience[3] for tupla_experience in batch]
        X = self.preprocess_imgs( lst_new_states )
        lst_new_qs = self.target_model.predict(X)    #####<<<<<

        ## create dataset for training
        lst_X = []  #imgs dataset, cioè gli states
        lst_y = []  #actions dataset, cioè i q_values
        for index, (current_state, action, reward, new_state, done) in enumerate(batch):
            lst_X.append(current_state) #<--- metto gli states nella X
            if not done:
                max_q = np.max( lst_new_qs[index] )
                new_q = reward + self.discount_rate * max_q
            else:
                new_q = reward
            current_qs = lst_current_qs[index]
            current_qs[action] = new_q    
            lst_y.append(current_qs) #<--- metto i q_values nella y

        ## if done train il simple_model
        if done:
            X = self.preprocess_imgs( lst_X )
            y = np.array( lst_y )
            self.simple_model.fit(x=X, y=y, batch_size=self.batch_size, epochs=1, verbose=0, shuffle=False)
            self.counter += 1

        # if counter update il target_model
        if self.counter > update_target_model_rate:
            self.target_model.set_weights( self.simple_model.get_weights() )
            self.counter = 0



###############################################################################
#                DUELING DEEP Q-LEARNING                                      #
###############################################################################
'''
'''
class agentA2C():
    def __init__(self):
        return 0