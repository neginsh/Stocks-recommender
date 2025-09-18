"""
train_rl_trader.py
Simple RL trading 
"""

import gym
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gather_windowed_data
from stock_trading_env import StockTradingEnv


# ---------------------------
# Helper functions: metrics and data
# ---------------------------
def compute_drawdown(returns):
    arr = np.array(returns)
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / (peak + 1e-9)
    max_dd = drawdown.max()
    return max_dd

def simple_backtest(env, model=None, baseline="equal"):
    """
    Run the environment with a trained model OR a baseline strategy.
    
    Args:
        env: StockTradingEnv (multi-stock)
        model: RL model (stable-baselines3). If None, run baseline.
        baseline: "equal" -> equal weights, "buy_and_hold" -> buy once at start, hold.
    
    Returns:
        portfolio_values: list of portfolio values over time
    """
    obs = env.reset()
    portfolio_values = []
    
    done = False
    step = 0
    
    while not done:
        if model is None:
            N = env.n_stocks
            if baseline == "equal":
                action = np.ones(N) / N   # rebalance equally each step
            elif baseline == "buy_and_hold":
                # only buy at first step, then hold weights
                if step == 0:
                    action = np.ones(N) / N
                    hold_action = action.copy()
                else:
                    action = hold_action
            else:
                raise ValueError("Unknown baseline")
        else:
            action, _states = model.predict(obs, deterministic=True)
            action = np.clip(action, 0, 1)  # ensure valid
        
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        step += 1
    
    return portfolio_values

# ---------------------------
# Example main: load data, create env, train PPO
# ---------------------------
def main():


    # ---------- load data and split to train and test ----------
    df = gather_windowed_data.get_data()
    train_df,test_df = gather_windowed_data.split_train_test(df)



    # ---------- Create envs ----------
    window_size = 10
    train_env = DummyVecEnv([lambda: StockTradingEnv(train_df, window_size=window_size, initial_cash=10000,N_stocks=gather_windowed_data.get_num_stocks())])
    test_env = StockTradingEnv(test_df, window_size=window_size, initial_cash=10000,N_stocks=gather_windowed_data.get_num_stocks())

    # ---------- Create and train model ---------- MlpLstmPolicy,MlpLnLstmPolicy
    model = PPO("MlpPolicy", train_env, verbose=1,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                batch_size=64,
                n_steps=2048,
                learning_rate=3e-4)
    # callbacks: checkpoint and simple eval callback (here eval uses deterministic = True)
    checkpoint_cb = CheckpointCallback(save_freq=5_000, save_path="./models/", name_prefix="ppo_trader")
    eval_cb = EvalCallback(train_env, best_model_save_path="./models/best/", log_path="./logs/",
                           eval_freq=10000, deterministic=True, render=False)

    print("Starting training...")
    model.learn(total_timesteps=50000, callback=[checkpoint_cb, eval_cb])
    model.save("ppo_trader_final")

    # ---------- Evaluate on test ----------
    print("Backtesting trained agent on test set...")
    portvals_agent = simple_backtest(test_env, model)
    # buy-and-hold baseline
    # create a copy of test env for buy-and-hold baseline
    test_env_bh = StockTradingEnv(test_df, window_size=window_size, initial_cash=10000)
    portvals_bh = simple_backtest(test_env_bh, model=None)

    # ---------- Metrics ----------
    agent_return = (portvals_agent[-1] / portvals_agent[0] - 1.0) * 100
    bh_return = (portvals_bh[-1] / portvals_bh[0] - 1.0) * 100
    agent_drawdown = compute_drawdown(portvals_agent)
    bh_drawdown = compute_drawdown(portvals_bh)

    print(f"Agent total return: {agent_return:.2f}% | Buy&Hold: {bh_return:.2f}%")
    print(f"Agent max drawdown: {agent_drawdown:.2%} | Buy&Hold max drawdown: {bh_drawdown:.2%}")

    # ---------- Plot ----------
    plt.figure(figsize=(12, 6))
    plt.plot(portvals_agent, label="Agent")
    plt.plot(portvals_bh, label="Buy & Hold")
    plt.title("Portfolio value (test period)")
    plt.xlabel("Step")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.grid(True)
    plt.savefig("backtest.png")
    plt.show()


if __name__ == "__main__":
    main()
