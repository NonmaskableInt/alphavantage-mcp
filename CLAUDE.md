# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a standalone **AlphaVantage MCP Server** that provides comprehensive market data and fundamental analysis through the AlphaVantage API.

**Critical Context**: This repository was extracted from the OmniT MCP servers collection. It now uses **uv** for package management (instead of Poetry) and includes a local copy of the `shared` types module for independence.

## Architecture

### MCP Server Structure
- Built using FastMCP framework (`mcp.server.fastmcp`)
- Implements the Model Context Protocol for LLM integration
- Runs on port 8002 with debug logging enabled
- Designed for integration with Claude Desktop and other MCP clients

### Package Management
- Uses **uv** for fast, reliable Python package management
- Removed unused dependencies (alpaca-py, python-dateutil)
- Core dependencies: mcp, alpha-vantage, pydantic, httpx, pandas, numpy

### Shared Type Definitions
The server imports shared type definitions from the local `shared.types` module:
- Response models: `MCPResponse`, `CompanyResponse`, `NewsResponse`, `TechnicalIndicatorResponse`, `BarsResponse`
- Data models: `CompanyOverview`, `NewsArticle`, `TechnicalIndicatorResult`, `BarData`
- Request models: `GetTechnicalIndicatorRequest`, `GetNewsRequest`
- Enums: `TechnicalIndicator`, `TimeFrame`

All response models follow a consistent pattern:
- `success: bool` - indicates if operation succeeded
- `data: Optional[...]` - contains the requested data
- `error: Optional[str]` - error message if operation failed
- `timestamp: datetime` - UTC timestamp of response

### AlphaVantage API Integration
The server initializes three AlphaVantage clients:
- `FundamentalData` - company financials and fundamentals
- `TechIndicators` - technical analysis indicators
- `TimeSeries` - historical price data

All clients use pandas output format for efficient data manipulation.

### Data Truncation Strategy
To prevent response size issues with LLMs:
- **Financial statements**: Limited to last 20 quarters/years
- **Technical indicators**: Limited to last 100 data points with summary statistics
- **Price data**: Limited to last 100 data points
- Summary metadata is added when data is truncated to provide context

## Available Tools

### Fundamental Data
- `get_company_overview(symbol)` - Company fundamentals with extensive metrics (PE ratio, market cap, EPS, margins, etc.)
- `get_income_statement(symbol)` - Annual income statement data
- `get_balance_sheet(symbol)` - Annual balance sheet data
- `get_cash_flow(symbol)` - Annual cash flow statements
- `get_earnings(symbol, period)` - Quarterly or annual earnings

### Market Data
- `get_daily_prices(symbol, outputsize)` - Daily OHLCV data (compact=100 days, full=20+ years)
- `get_intraday_prices(symbol, interval, outputsize)` - Intraday prices (1min, 5min, 15min, 30min, 60min)

### News & Sentiment
- `get_market_news(tickers, topics, time_from, time_to, limit)` - Market news with sentiment analysis

### Technical Analysis
- `get_technical_indicators(symbol, indicator, interval, time_period)` - Supports:
  - RSI, MACD, BBANDS (Bollinger Bands)
  - SMA, EMA (Moving Averages)
  - STOCH (Stochastic), ADX, WILLR (Williams %R)

## Development Commands

### Environment Setup
```bash
# Install dependencies (uv will automatically create venv and install)
uv sync

# Install with dev dependencies
uv sync --extra dev

# Set required API key
export ALPHA_VANTAGE_API_KEY=your_key_here
```

### Running the Server
```bash
# Direct execution (uv automatically manages the virtual environment)
uv run alphavantage-mcp-server

# Using launch script
./launch-alphavantage.sh
```

### Testing
```bash
# Run tests
uv run pytest

# Type checking
uv run mypy .

# Format code
uv run black .
```

## Integration with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "alphavantage-data": {
      "command": "uv",
      "args": ["run", "alphavantage-mcp-server"],
      "cwd": "/Users/chris/dev/onyx/alphavantage_mcp",
      "env": {
        "ALPHA_VANTAGE_API_KEY": "your_key"
      }
    }
  }
}
```

## Important Implementation Details

### Error Handling
- All tools return typed response objects with `success` flag
- Errors are caught and returned in `error` field rather than raising exceptions
- Empty data checks prevent returning incomplete responses

### Data Type Conversion
- Pandas DataFrames are converted to lists of dictionaries for JSON serialization
- String values of "None" or "-" are filtered out and converted to Python `None`
- All numeric fields are explicitly cast to `float` or `int` to ensure type safety

### AlphaVantage API Quirks
- Uses direct HTTP calls for news API (via httpx) as the official SDK doesn't support it
- Different indicators require different parameter combinations (MACD doesn't use time_period, BBANDS does)
- Timestamp parsing for news requires special handling (removing 'T' and 'Z' characters)

## Python Version
Requires Python >=3.10, <3.13 (due to numpy<2.0.0 constraint)
The repository includes a `.python-version` file set to 3.11 for consistency.

## Dependencies Summary

### Production Dependencies
- **mcp** - Model Context Protocol framework
- **alpha-vantage** - AlphaVantage API client
- **pydantic** - Data validation and settings management
- **httpx** - HTTP client for news API calls
- **pandas** - Data manipulation (used by alpha-vantage)
- **numpy** - Numerical computing (dependency of pandas)

### Development Dependencies (optional)
- **pytest** - Testing framework
- **pytest-asyncio** - Async testing support
- **mypy** - Static type checking
- **black** - Code formatting
