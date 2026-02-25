# Kalshi Probability Arbitrage

Backtesting system for trading WTI crude oil weekly binary options on [Kalshi](https://kalshi.com) prediction markets. The strategy compares model-derived theoretical probabilities against market prices to find mispriced contracts, then simulates SELL-side trading with Kelly Criterion position sizing and active exit management.

## How It Works

The strategy sells overpriced binary option contracts. Each contract pays $1 if oil settles in a specific price range at weekly expiry, and $0 otherwise. When the model estimates a lower probability than what the market implies, it sells the contract and collects the premium.

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
| Test (6) | Jan 9 - Feb 13 | Out-of-sample evaluation |

Execution constraints applied during walk-forward:
- **Slippage:** 0.25 cents per contract
- **Partial fills:** Per-trade fill probability from orderbook simulation (trades that round to 0 contracts are dropped)
- **Fees:** 7% on gross profits


## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| Bankroll | 10,000 cents ($100) | Starting capital per market |
| Kelly fraction | Variable, depends on forecasted probability | Kelly criterion for conservative sizing |
| Min edge threshold | 10% | Minimum model vs. market disagreement to trade |
| Max positions | 3 | Simultaneous open positions, limited to decrease impact on market price when liquidity low |
| Stop loss | 500 cents ($5) per strike | Cumulative loss limit before blocking a strike |
| Fee rate | 7% | Applied to gross profits only |
| Profit target | 50% | Unrealized P&L trigger for early exit |
| Exit edge threshold | 3% | Close if remaining edge decays below this |

## Results

Total P&L: +54594.79 cents ($+545.95)<br>
Sharpe: 1.92<br>
Win Rate: 78.0<br>
Profitable markets: 6/6<br>
<br>
Re-running this backtest may shift results by a few trades due to variance in the Montecarlo simulations, but every result I have gotten re-testing has been close to those listed above in every metric.

## Weekly Results

| Week | P&L | Trade Count | Winrate |
|---|---|---|---|
| Jan 5 - 9 | +2239.33c | 17 | 82.4% |
| Jan 13 - 16 | +3557.61c | 31 | 77.4% |
| Jan 19 - 23 | +31917.04c | 8 | 87.5% |
| Jan 26 - 30 | +501.69c | 5 | 60.0% |
| Feb 2 - 6 | +9411.42c | 13 | 69.2% |
| Feb 9 - 13 | +6967.70c | 12 | 91.7% |

## Discussion
As this strategy usually ends up trading lower-probability markets that have inflated prices due to factors including but not limited to psychological biases and lack of information, this good performance is within the realm of possibility of continuing if live traded. This is due to the fact that these markets have incredibly thin liquidity, which is reflected in the simulated order book used, and is also the reason that this edge hasn't been taken by other traders. In practice, the maximum liquidity available to still have an edge with this strategy varies, but should be around the $500-1000 dollar mark, meaning there is little money to be made with this strategy for the effort and computing power you would put in. However, conceptually, this demonstrates the alpha available in prediction markets through creating a prediction model that uses information not considered by the average gambler, albeit this needs to be tested on a greater sample size than the 3.5 months of data I had available.

