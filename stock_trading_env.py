
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
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, window_size=10, initial_cash=1_0000.0,
                 transaction_cost_pct=0.001, reward_scaling=1.0):
        super().__init__()

        assert isinstance(df, pd.DataFrame)
        self.df = df.reset_index(drop=True).copy()
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.reward_scaling = reward_scaling

        # actions: 0 hold, 1 buy, 2 sell
        self.action_space = spaces.Discrete(3)

        # observation: window_size * n_features + 2 (cash ratio, holdings ratio)
        self.price_col = "close"
        self.features = ["close", "volume", "open", "high", "low"]
        # we'll normalize by recent window mean/std
        obs_len = window_size * len(self.features) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        # internal state
        self._start_index = window_size  # earliest index we can start stepping from
        self._end_index = len(self.df) - 1
        self.reset()

    def reset(self):
        self.current_step = self._start_index
        self.cash = self.initial_cash
        self.shares = 0.0
        self.total_shares_bought = 0.0
        self.total_shares_sold = 0.0
        self.trades = 0
        self.done = False

        self._initial_portfolio_value = self.initial_cash
        return self._get_obs()

    def _get_obs(self):
        # get window of features ending at current_step-1 (so agent only sees past data)
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.df.loc[start:end - 1, self.features].values  # shape (window_size, n_features)

        # normalize window by window mean/std (avoid division by zero)
        mean = window.mean(axis=0)
        std = window.std(axis=0)
        std[std == 0] = 1.0
        norm_window = (window - mean) / std

        # flatten
        obs = norm_window.flatten()

        # add portfolio info (cash ratio, holdings ratio)
        current_price = float(self.df.loc[self.current_step, self.price_col])
        portfolio_value = self.cash + self.shares * current_price
        cash_ratio = self.cash / (portfolio_value + 1e-9)
        holdings_ratio = (self.shares * current_price) / (portfolio_value + 1e-9)

        obs = np.concatenate([obs, np.array([cash_ratio, holdings_ratio])]).astype(np.float32)
        return obs

    def step(self, action):
        assert self.action_space.contains(action)
        if self.done:
            return self._get_obs(), 0.0, True, {}

        prev_portfolio_value = self.cash + self.shares * float(self.df.loc[self.current_step, self.price_col])

        # execute action at today's open price of next step (simulate next-day execution)
        # to keep it simple, we use current step price (could use 'open' of next step)
        price = float(self.df.loc[self.current_step, self.price_col])

        if action == 1:  # BUY: spend all cash to buy
            if self.cash > 0:
                shares_to_buy = (self.cash * (1.0 - self.transaction_cost_pct)) / price
                self.shares += shares_to_buy
                spent = shares_to_buy * price
                fee = spent * self.transaction_cost_pct
                self.cash -= spent + fee  # approximate double-counting fee; keep simple
                self.total_shares_bought += shares_to_buy
                self.trades += 1
        elif action == 2:  # SELL: sell all shares
            if self.shares > 0:
                proceeds = self.shares * price * (1.0 - self.transaction_cost_pct)
                fee = self.shares * price * self.transaction_cost_pct
                self.cash += proceeds - fee
                self.total_shares_sold += self.shares
                self.shares = 0.0
                self.trades += 1
        # else: hold

        # move forward one step
        self.current_step += 1

        # compute reward: portfolio value change
        current_price = float(self.df.loc[self.current_step, self.price_col]) if self.current_step <= self._end_index else price
        portfolio_value = self.cash + self.shares * current_price
        reward = (portfolio_value - prev_portfolio_value) * self.reward_scaling

        # optional: penalize too many trades (uncomment if desired)
        # reward -= 0.001 * self.trades

        # done if out of data
        if self.current_step >= self._end_index:
            self.done = True

        obs = self._get_obs() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
        }
        return obs, float(reward), bool(self.done), info

    def render(self, mode="human"):
        price = float(self.df.loc[self.current_step, self.price_col])
        print(f"Step: {self.current_step}, Price: {price:.2f}, Cash: {self.cash:.2f}, Shares: {self.shares:.4f}")
