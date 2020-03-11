import numpy as np


class Agent:
    # The number of values that agent status configures
    STATE_DIM = 2  # stock holding ratio, portfolio value ratio

    # Action
    ACTION_BUY = 0  # Buying
    ACTION_SELL = 1  # Selling
    ACTION_HOLD = 2  # Holding
    ACTIONS = [ACTION_BUY, ACTION_SELL]  # Actions to find probabilities in artificial neural networks (holding?)
    NUM_ACTIONS = len(ACTIONS)  # Number of outputs to consider in artificial neural network

    def __init__(
        self, environment, min_trading_unit=1, max_trading_unit=2, 
        delayed_reward_threshold=.05,tax=False):
        # Environment object
        self.environment = environment  # Referring environment to get current stock price

        # Minimum trading unit, maximum trading unit, delay reward threshold
        self.min_trading_unit = min_trading_unit  # Minimum trading unit
        self.max_trading_unit = max_trading_unit  # Maximum trading unit
        self.delayed_reward_threshold = delayed_reward_threshold  # Delay reward threshold

        # Properties of the agent class
        self.initial_balance = 0  # initial balance
        self.balance = 0  # Current cash balance
        self.num_stocks = 0  # Current number of holding stocks
        self.portfolio_value = 0  # balance + num_stocks * {Current stock price}
        self.base_portfolio_value = 0  # PV at the last learning point
        self.num_buy = 0  # Number of buying
        self.num_sell = 0  # Number of selling
        self.num_hold = 0  # Number of holding
        self.immediate_reward = 0  # Immediate reward

        # State of the agent class
        self.ratio_hold = 0  # Stock holding ratio
        self.ratio_portfolio_value = 0  # Portfolio Value Ratio

        if not tax:
            self.TRADING_CHARGE = 0  # Not considering trading charge (generally 0.015%)
            self.TRADING_TAX = 0  # Not considering trading tax (actual 0.3%)
        else:
            # Trading charge and trading tax
            self.TRADING_CHARGE = 0.00015  # Not considering trading charge (generally 0.015%)
            self.TRADING_TAX = 0.003  # Not considering trading tax (actual 0.3%)



    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.
        # Exploration decision
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)  # Randomly decide the action
        else:
            exploration = False
            probs = policy_network.predict(sample)  # Probability for each action
            action = np.argmax(probs)
            confidence = probs[action]
        return action, confidence, exploration

    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_BUY:
            # Check if you can buy at least one stock
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:
            # Check if you have stock balance
            if self.num_stocks <= 0:
                validity = False
        return validity

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # Get current price in environment
        curr_price = self.environment.get_price()

        # Initiate the immediate reward
        self.immediate_reward = 0

        # Buying
        if action == Agent.ACTION_BUY:
            # Determine the unit to buy
            trading_unit = self.decide_trading_unit(confidence)
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # If you do not have enough cash, buy as much as you can with the cash you have.
            if balance < 0:
                trading_unit = max(min(
                    int(self.balance / (
                        curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
                    self.min_trading_unit
                )
            # Calculate total purchase amount by applying fee
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount  # Update the holding cash
            self.num_stocks += trading_unit  # Update the number of holding stock
            self.num_buy += 1  # Increase the number of buying

        # Selling
        elif action == Agent.ACTION_SELL:
            # Determine the unit to sell
            trading_unit = self.decide_trading_unit(confidence)
            # If you do not have enough stock, then sell as much as you can
            trading_unit = min(trading_unit, self.num_stocks)
            # Selling
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit  # Update the number of holding stock
            self.balance += invest_amount  # Update the holding cash
            self.num_sell += 1  # Increase the number of selling

        # Holding
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # Increase the holding number

        # Update the portfolio value
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        # Determine the immediate reward
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # Determine the delayed reward
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
            # Update base portfolio value by achieving target benefit
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1
            # Update base portfolio value by exceeding the loss trheshold
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward
