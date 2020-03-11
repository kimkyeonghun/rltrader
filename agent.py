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
        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)  # 무작위로 행동 결정
        else:
            exploration = False
            probs = policy_network.predict(sample)  # 각 행동에 대한 확률
            action = np.argmax(probs)
            confidence = probs[action]
        return action, confidence, exploration

    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인 
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

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(min(
                    int(self.balance / (
                        curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount  # 보유 현금을 갱신
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신
            self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
            self.balance += invest_amount  # 보유 현금을 갱신
            self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        # 즉시 보상 판단
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # 지연 보상 판단
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward
