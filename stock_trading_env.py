
import gym
import numpy as np
import pandas as pd
from gym import spaces

class StockTradingEnv(gym.Env):
    """
    A single-stock trading environment for RL.
    Action space:
        0 = Hold
        1 = Buy (use all cash to buy at next day's open price)
        2 = Sell (sell all shares at next day's open price)
    Observation:
        window of normalized OHLCV features + [cash_ratio, holdings_ratio]
    Reward:
        change in total portfolio value (t+1 vs t) minus transaction cost
    """

    def __init__(self,df, window_size=10, initial_cash=10000.0,
                 transaction_cost_pct=0.001, reward_scaling=1.0,N_stocks=5):
        super().__init__()

        assert isinstance(df, pd.DataFrame)
        self.df = df.reset_index(drop=True).copy()
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.reward_scaling = reward_scaling
        self.N_stocks = N_stocks

        self.action_space = spaces.Box(low=0, high=1, shape=(N_stocks,), dtype=np.float32)
        # observation: window_size * n_features + 2 (cash ratio, holdings ratio)
        self.price_col = "close"
        self.features = ["close", "volume", "open", "high", "low"]
        obs_len = window_size * len(self.features) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        self._start_index = window_size  # earliest index we can start stepping from
        self._end_index = len(self.df) - 1
        self.reset()

    def reset(self):
        self.current_step = self._start_index
        self.cash = self.initial_cash
        self.shares = np.zeros(self.N_stocks)
        self.total_shares_bought = np.zeros(self.N_stocks)
        self.total_shares_sold = np.zeros(self.N_stocks)
        self.done = False

        self._initial_portfolio_value = self.initial_cash
        return self._get_obs()

    def _get_obs(self):
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.df.loc[start:end - 1, self.features].values  

        mean = window.mean(axis=0)
        std = window.std(axis=0)
        std[std == 0] = 1.0
        norm_window = (window - mean) / std

        obs = norm_window.flatten()

        self.current_price = np.array(self.df.loc[self.current_step, self.price_col], dtype=np.int16)
        portfolio_value = self.cash + self.shares * self.current_price
        cash_ratio = self.cash / (portfolio_value + 1e-9)
        holdings_ratio = (self.shares * self.current_price) / (portfolio_value + 1e-9)

        obs = np.concatenate([obs, np.array([cash_ratio, holdings_ratio])]).astype(np.float32)
        return obs

    def step(self, action):
        """
        action: np.array of shape (N_stocks,)
            Fraction of portfolio to allocate to each stock.
            Must be >=0. Will be normalized to sum<=1.
            Remaining weight goes to cash.
        """
        # normalize action
        action = np.clip(action, 0, 1)
        if action.sum() > 1:
            action = action / action.sum()

        weights = action
        cash_weight = 1 - weights.sum()

        # portfolio value before step
        prev_val = self.cash + np.sum(self.shares * self.current_prices)

        # move one step forward
        self.current_step += 1
        if self.current_step >= self._end_index:
            self.done = True

        # new prices
        self.current_prices = self.df.loc[self.current_step, self.price_cols].values.astype(float)

        # target dollar allocation
        target_val = (prev_val) * weights
        target_cash = (prev_val) * cash_weight

        # current dollar allocation
        current_val = self.shares * self.current_prices
        current_cash = self.cash

        # difference = how much to trade
        delta = target_val - current_val

        # transaction costs
        costs = np.sum(np.abs(delta) * self.transaction_cost_pct)

        # update holdings
        self.shares = target_val / (self.current_prices + 1e-9)
        self.cash = target_cash

        # portfolio value after rebalancing
        new_val = self.cash + np.sum(self.shares * self.current_prices) - costs

        # reward = change in portfolio value
        reward = (new_val - prev_val) / prev_val

        obs = self._get_obs() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"portfolio_value": new_val, "weights": weights}

        return obs, reward, self.done, info

    def render(self, mode="human"):
        price = float(self.df.loc[self.current_step, self.price_col])
        print(f"Step: {self.current_step}, Price: {price:.2f}, Cash: {self.cash:.2f}, Shares: {self.shares:.4f}")
