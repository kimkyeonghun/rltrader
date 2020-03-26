import os
import logging
import numpy as np
import settings
from environment import Environment
from agent import Agent
from AC_LSTM import ACagent
from visualizer import Visualizer


class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05, lr=0.01,tax=False):
        self.stock_code = stock_code  # Stock coder
        self.chart_data = chart_data
        self.environment = Environment(chart_data)  # Environment object
        self.tax = tax
        # Agent object
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold,
                           tax=tax)
        self.training_data = training_data  # Training data
        self.sample = None
        self.training_data_idx = -1

        # Policy neural network; Input size = size of training data + agent state size
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        self.AC = ACagent(
            input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        self.visualizer = Visualizer()  # Visualization module

    def reset(self):
        self.sample = None
        self.training_data_idx = -1

    def fit(
        self, num_epoches=1000, max_memory=60, balance=10000000,
        discount_factor=0, start_epsilon=.5, learning=True):
        logging.info("\n\nAcotr LR: {Alr}, Critic LR: {Clr}, DF: {discount_factor}, "
                    "TU: [{min_trading_unit}, {max_trading_unit}], "
                    "DRT: {delayed_reward_threshold}, Tax: {tax}".format(
            Alr=self.AC.actor_lr,
            Clr=self.AC.critic_lr,
            discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold,
            tax = self.tax
        ))

        # Visualization Preparation
        # Pre-visualization the chart data as it does not change
        self.visualizer.prepare(self.environment.chart_data)

        # Prepare the folders to store visualization results
        epoch_summary_dir = os.path.join(
            settings.BASE_DIR, 'epoch_summary/%s/epoch_summary_%s' % (
                self.stock_code, settings.timestr))
        if not os.path.isdir(epoch_summary_dir):
            os.makedirs(epoch_summary_dir)

        # Set agent's initial balance
        self.agent.set_balance(balance)

        # Initialize the information about training
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # Training repetition
        for epoch in range(num_epoches):
            # Initialize the information about epoch
            #loss = 0.
            itr_cnt = 0
            win_cnt = 0
            exploration_cnt = 0
            batch_size = 0
            pos_learning_cnt = 0
            neg_learning_cnt = 0

            # Initialize the memory
            memory_sample = []
            memory_action = []
            memory_reward = []
            memory_prob = []
            memory_pv = []
            memory_num_stocks = []
            memory_exp_idx = []
            memory_learning_idx = []
            
            # Initialize the environment, agent and policy nerual network
            self.environment.reset()
            self.agent.reset()
            self.AC.reset()
            self.reset()

            # Initialize the visualizer
            self.visualizer.clear([0, len(self.chart_data)])

            # Exploration rate decreases as you progress
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = 0

            while True:
                # Sample generation
                next_sample = self._build_sample()
                if next_sample is None:
                    break

                # Actions decided by policy neural network or exploration


                action, confidence, exploration = self.agent.decide_action(
                    self.AC, self.sample, epsilon)

                # Perform the action you decided and earn immediate and delayed rewards
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                # Store the actions and the consequences for the actions
                memory_sample.append(next_sample)
                memory_action.append(action)
                memory_reward.append(immediate_reward)
                memory_pv.append(self.agent.portfolio_value)
                memory_num_stocks.append(self.agent.num_stocks)
                memory = [(
                    memory_sample[i],
                    memory_action[i],
                    memory_reward[i])
                    for i in list(range(len(memory_action)))[-max_memory:]
                ]
                if exploration:
                    memory_exp_idx.append(itr_cnt)
                    memory_prob.append([np.nan] * Agent.NUM_ACTIONS)
                else:
                    memory_prob.append(self.AC.prob)

                # Update the information about iterations
                batch_size += 1
                itr_cnt += 1
                exploration_cnt += 1 if exploration else 0
                win_cnt += 1 if delayed_reward > 0 else 0

                # Update policy neural network when in training mode and delay rewards exist
                if delayed_reward == 0 and batch_size >= max_memory:
                    delayed_reward = immediate_reward
                    self.agent.base_portfolio_value = self.agent.portfolio_value
                if learning and delayed_reward != 0:
                    # Size of batch traning data
                    batch_size = min(batch_size, max_memory)
                    # Generate batch training data
                    x, _ = self._get_batch(
                        memory, batch_size, discount_factor, delayed_reward)
                    if len(x) > 0:
                        if delayed_reward > 0:
                            pos_learning_cnt += 1
                        else:
                            neg_learning_cnt += 1
                        # Update Policy neural network
                        self.AC.train_model(self.sample,action,delayed_reward,next_sample)
                        memory_learning_idx.append([itr_cnt, delayed_reward])
                    batch_size = 0

            # Visualize the information about epoches
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')

            self.visualizer.plot(
                epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
                action_list=Agent.ACTIONS, actions=memory_action,
                num_stocks=memory_num_stocks, outvals=memory_prob,
                exps=memory_exp_idx, learning=memory_learning_idx,
                initial_balance=self.agent.initial_balance, pvs=memory_pv
            )
            self.visualizer.save(os.path.join(
                epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                    settings.timestr, epoch_str)))

            logging.info("[Epoch {}/{}]\tEpsilon:{}\t#Expl.:{}/{}\t"
                        "#Buy:{}\t#Sell:{}\t#Hold:{}\t"
                        "#Stocks:{}\tPV:{:,}원\t"
                        "POS:{}\tNEG:{}".format(
                            epoch_str, num_epoches, round(epsilon,4), exploration_cnt, itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_stocks,
                            int(self.agent.portfolio_value),
                            pos_learning_cnt, neg_learning_cnt))

            # Update the information about training
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # Record the information about training in log
        logging.info("Max PV: {:,}원, \t # Win: {}".format(
            int(max_portfolio_value), epoch_win_cnt))

    def _get_batch(self, memory, batch_size, discount_factor, delayed_reward):
        x = np.zeros((batch_size, 1, self.num_features))
        y = np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)

        for i, (sample, action, _) in enumerate(
                reversed(memory[-batch_size:])):
            x[i] = np.array(sample).reshape((-1, 1, self.num_features))
            y[i, action] = (delayed_reward + 1) / 2
            if discount_factor > 0:
                y[i, action] *= discount_factor ** i
        return x, y

    def _build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.AC.load_model(model_path=model_path)
        self.fit(balance=balance, num_epoches=1, learning=False)
