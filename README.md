# Kalshi Probability Arbitrage

Backtesting system for trading WTI crude oil weekly binary options on [Kalshi](https://kalshi.com) prediction markets. The strategy compares model-derived theoretical probabilities against market prices to find mispriced contracts, then simulates SELL-side trading with Kelly Criterion position sizing and active exit management.

## Economic Rationale

There are three main factors that cause the edge found in this strategy to exist: longshot bias, lower quality market participants, and illiquidity. Longshot bias is betting pattern where betters overvalue longshots and undervalue favorite outcomes. Due to the lower quality of market participants who choose to trade in a betting market rather than trading WTI futures (if they had a legitimate strategy worth funding at scale, they would be trading the significantly more liquid asset), I believe this bias exists within these markets. 

If this bias exists, why hasn't it already been arbitraged away by a well-capitalized fund or individual? These betting markets for liquid financial assets, which this strategy focuses on, act as extremely illiquid derivatives (the entire market value of a weekly WTI prediction market is around $500k), where the profits of exploiting this edge would not be scalable to the level required for a larger fund to justify the expenditure of time and computing power it takes to create and deploy a trading strategy.

Several pricing strategies have been developed to take advantage of longshot bias, such as Shin's model and its variants and adaptations, which I would recommend looking into using for general betting market trading strategies that trade actual bets. The WTI price prediction markets act as markets of binary option contracts, so we can apply models that generate probabilities of the asset price hitting a certain target based on information provided by the market of the underlying asset, which causes our method to be more accurate than the application of a pure mathematical correction such as Shin's model given the constraint that we only apply this strategy within derivative betting markets. 

## How It Works

The strategy sells overpriced binary option contracts. Each contract pays $1 if oil settles in a specific price range at weekly expiry, and $0 otherwise. When the model estimates a lower probability than what the market implies, it sells the contract and collects a premium when the price corrects.

**Pipeline:**

1. **Load Kalshi market data** from CSV (hourly timestamps, strike ranges as columns, prices in cents)
2. **Fetch hourly WTI futures prices** (CL=F) from Yahoo Finance with extended lookback for model calibration
3. **Fit GARCH(1,1)** to hourly returns and forecast annualized volatility
4. **Outcome probability estimates** 50/50 weight between a distribution of Montecarlo price paths simulated on GARCH forecasted vol and an empirical historical return distribution, with mean-reversion z-score adjustment
5. **Generate SELL signals** where model probability < market probability (negative edge)
6. **Size positions** using Kelly criterion, capped by synthetic orderbook liquidity
7. **Manage exits** actively: edge reversal, edge decay below 3%, profit target, or stop loss
8. **Settle** all open positions at market conclusion with binary payout

## Setup

```bash
pip install pandas numpy scipy yfinance arch
```

## Usage

```bash
# Walk-forward testing across out of sample markets
python main.py walk-forward
```

## Walk-Forward Validation

Markets are split chronologically into three phases:

| Phase | Markets | Purpose |
|---|---|---|
| Training (4) | Nov 14 - Dec 5 | Grid search over base_edge_threshold |
| Validation (3) | Dec 19 - Jan 2 | Select best params by Sharpe ratio |
| Test (7) | Jan 9 - Feb 20 | Out-of-sample evaluation |

Execution constraints applied during walk-forward:
- **Slippage:** 0.25 cents per contract
- **Partial fills:** Per-trade fill probability from orderbook simulation (trades that round to 0 contracts are dropped)
- **Fees:** 7% on gross profits


## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| Bankroll | 10,000 cents ($100) | Starting capital per market |
| Kelly fraction | Optimized to 0.3 during validation | Kelly criterion for conservative sizing |
| Min edge threshold | 10% | Minimum model vs. market disagreement to trade |
| Max positions | 3 | Simultaneous open positions, limited to decrease impact on market price when liquidity low |
| Stop loss | 500 cents ($5) per strike | Cumulative loss limit before blocking a strike |
| Fee rate | 7% | Applied to gross profits only |
| Profit target | 50% | Unrealized P&L trigger for early exit |
| Exit edge threshold | 3% | Close if remaining edge decays below this |

## Results

Total P&L: +62024.21 cents ($+620.24)<br>
Sharpe: 2.04<br>
Win Rate: 78.9<br>
Profitable markets: 7/7<br>
<br>
Re-running this backtest may shift results by a few trades due to variance in the Montecarlo simulations, but every result I have gotten re-testing has been close to those listed above in every metric.

## Weekly Results

| Week | P&L | Trade Count | Winrate |
|---|---|---|---|
| Jan 5 - 9 | +2617.23c | 16 | 93.8% |
| Jan 13 - 16 | +3832.74c | 32 | 78.1% |
| Jan 19 - 23 | +34053.64c | 9 | 88.9% |
| Jan 26 - 30 | +407.59c | 5 | 60.0% |
| Feb 2 - 6 | +9902.17c | 13 | 69.2% |
| Feb 9 - 13 | +6289.62c | 11 | 90.9% |
| Feb 16 - 20 | +4921.22c | 14 | 71.4% |

## Discussion
As this strategy usually ends up trading lower-probability markets that have inflated prices due to factors including but not limited to psychological biases and lack of information, this good performance is within the realm of possibility of continuing if live traded. This happens because we are trading very small amounts due to the fact that these markets have incredibly thin liquidity, which is accounted for in the simulated order book used. In practice, the maximum liquidity available to still have a significant edge with this strategy varies, but should be around the $500-1000 dollar mark, meaning there is little money to be made with this strategy for the effort and computing power you would put in. However, conceptually, this demonstrates the alpha available in prediction markets through creating a prediction model that uses information not considered by the average gambler, albeit this needs to be tested on a greater sample size than the 3.5 months of data I had available.

