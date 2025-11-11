# Premarket Agent

Premarket Agent is a modular, multi-agent research and trading system that continuously monitors premarket equity activity, aggregates fundamental and alternative data, generates risk-aware trade decisions, and executes both paper and live trades. The architecture is designed for extensibility, rigorous evaluation, and compliance with broker APIs.

## Key Features
- **Market Intelligence Agent**: Streams premarket movers using Interactive Brokers, Yahoo Finance, and other data sources, filtering for sustained upside momentum.
- **Research Agent**: Collects news, SEC filings, earnings transcripts, alternative data, and analyst sentiment; produces normalized research artifacts.
- **Scoring & Allocation Agent**: Fuses classical risk heuristics with a trainable neural policy to rank opportunities, perform portfolio allocation, and define trade plans.
- **Execution Agent**: Places simulated or real orders via broker adapters, handles order state, and enforces risk tolerances.
- **Post-Trade Agent**: Monitors fills, price action, and news to optimize exits and update performance analytics.
- **Neural Policy Training Loop**: Collects virtual-trade experiences, updates a PyTorch policy network, and harmonizes it with analyst news and trend signals.
- **Live Portfolio Dashboard**: Streams paper-trade performance with cash, PnL, and open positions in a color-coded terminal UI.
- **Resilient Data Layer**: Shared rate-limiting and a fallback symbol universe keep the agents running when third-party APIs throttle requests.
- **Simulation & Reinforcement Learning**: Supports paper trading with historical and live market data to train policies before going live.

## Project Structure
```
src/
  premarket_agent/
    agents/        # Agent abstractions and orchestrators
    config/        # Settings and configuration schemas
    data/          # Data access layer, caching, and feature stores
    services/      # External integrations (brokers, news, analytics)
    utils/         # Shared utilities and helpers
tests/             # Unit and integration tests
```

## Getting Started
1. Create and activate a virtual environment.
2. Install the project dependencies:
   - `python -m pip install pip==24.3.1`
   - `python -m pip install --upgrade setuptools wheel backports.tarfile`
   - `python -m pip install --upgrade --user numexpr==2.7.3 gevent==24.2.1 multitasking==0.0.11`
   - `python -m pip install -e .[dev]`
3. Copy `.env.example` to `.env` and set broker/API credentials.
4. Run `python -m premarket_agent.cli` to start the orchestrator and open the live dashboard (paper mode by default).
5. Train the neural policy with `python -m premarket_agent.cli simulate --episodes 200`.

## Simulation to Live Workflow
1. **Data Ingestion**: Collect historical premarket data, news, and fundamentals.
2. **Training**: Use the simulation runner to evaluate strategies with virtual capital.
3. **Evaluation**: Review risk metrics, win rates, and drawdown constraints.
4. **Promotion**: Switch the execution adapter from `paper` to `live` mode after passing guardrail checks.
5. **Monitoring**: Continuously monitor allocations, PnL, and compliance events.

## License
@ Copyright property of the developers Ofir Elmakias and Sergey Liflandsky.
This software is propriatary and cant be used withought a written permission from both developers.


