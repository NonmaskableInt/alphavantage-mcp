# AlphaVantage MCP Server

A Model Context Protocol (MCP) server that provides comprehensive market data and fundamental analysis through the AlphaVantage API. Designed for integration with Claude Desktop and other MCP-compatible clients.

## Requirements

- Python 3.10 - 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- AlphaVantage API key ([get one free](https://www.alphavantage.co/support/#api-key))

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd alphavantage_mcp

# Install dependencies
uv sync
```

## Configuration

Set your AlphaVantage API key as an environment variable:

```bash
# macOS/Linux
export ALPHA_VANTAGE_API_KEY=your_key_here

# Windows (Command Prompt)
set ALPHA_VANTAGE_API_KEY=your_key_here

# Windows (PowerShell)
$env:ALPHA_VANTAGE_API_KEY="your_key_here"
```

Or create a `.env` file in the project directory:

```
ALPHA_VANTAGE_API_KEY=your_alphavantage_api_key
```

### Server Configuration (Optional)

| Variable | Description | Default |
|----------|-------------|---------|
| `ALPHAVANTAGE_DEBUG` | Enable debug mode | `false` |
| `ALPHAVANTAGE_PORT` | Server port | `8002` |
| `ALPHAVANTAGE_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

## Running the Server

```bash
# Direct execution
uv run alphavantage-mcp-server

# Using launch script (macOS/Linux)
./launch-alphavantage.sh

# Using launch script (Windows)
launch-alphavantage.bat
```

## Usage with Claude Desktop

Add to your Claude Desktop configuration file:

| Platform | Config Location |
|----------|-----------------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

**Option 1: Using launch script (recommended)**
```json
{
  "mcpServers": {
    "alphavantage-data": {
      "command": "/path/to/alphavantage_mcp/launch-alphavantage.sh",
      "env": {
        "ALPHA_VANTAGE_API_KEY": "your_key"
      }
    }
  }
}
```

**Option 2: Using absolute path to uv**

First, find your `uv` path: `which uv`

```json
{
  "mcpServers": {
    "alphavantage-data": {
      "command": "/Users/yourname/.local/bin/uv",
      "args": ["run", "alphavantage-mcp-server"],
      "cwd": "/path/to/alphavantage_mcp",
      "env": {
        "ALPHA_VANTAGE_API_KEY": "your_key"
      }
    }
  }
}
```

**Note:** GUI applications like Claude Desktop don't inherit your shell's PATH, so using absolute paths is more reliable than relying on `uv` being in PATH.

On Windows, use backslashes: `"cwd": "C:\\path\\to\\alphavantage_mcp"`

## Available Tools

### Fundamental Data

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_company_overview` | Company fundamentals including PE ratio, market cap, EPS, margins, and more | `symbol` |
| `get_income_statement` | Annual income statement data (last 20 periods) | `symbol` |
| `get_balance_sheet` | Annual balance sheet data (last 20 periods) | `symbol` |
| `get_cash_flow` | Annual cash flow statements (last 20 periods) | `symbol` |
| `get_earnings` | Quarterly or annual earnings data | `symbol`, `period` (quarterly/annual) |

### Market Data

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_daily_prices` | Daily OHLCV data | `symbol`, `outputsize` (compact=100 days, full=20+ years) |
| `get_intraday_prices` | Intraday price data | `symbol`, `timeframe` (TimeFrame enum), `outputsize` |

### News & Sentiment

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_market_news` | Market news with sentiment analysis | `tickers` (list), `topics` (list), `time_from`, `time_to`, `limit` |

### Technical Analysis

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_technical_indicators` | Calculate technical indicators | `symbol`, `indicator` (TechnicalIndicator enum), `timeframe` (TimeFrame enum), `time_period` |

**Supported indicators (TechnicalIndicator enum):**
- `RSI` - Relative Strength Index
- `MACD` - Moving Average Convergence Divergence
- `BOLLINGER_BANDS` - Bollinger Bands
- `SMA` - Simple Moving Average
- `EMA` - Exponential Moving Average
- `STOCHASTIC` - Stochastic Oscillator
- `ADX` - Average Directional Index
- `WILLIAMS_R` - Williams %R

**Supported timeframes (TimeFrame enum):**
- `MINUTE` - 1 minute
- `FIVE_MINUTES` - 5 minutes
- `FIFTEEN_MINUTES` - 15 minutes
- `THIRTY_MINUTES` - 30 minutes
- `HOUR` - 1 hour
- `DAILY` - Daily
- `WEEKLY` - Weekly
- `MONTHLY` - Monthly

## Examples

**Get company fundamentals:**
```
"Get the company overview for AAPL"
```

**Analyze price trends:**
```
"Show me the daily prices for TSLA over the last 100 days"
```

**Technical analysis:**
```
"Calculate the RSI for MSFT with a 14-day period"
```

**Market news:**
```
"Get recent news about NVDA with sentiment analysis"
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run the server directly
uv run alphavantage-mcp-server

# Run tests
uv run pytest

# Type checking
uv run mypy .

# Format code
uv run black .
```

## API Rate Limits

The free AlphaVantage API tier allows 25 requests per day. Consider upgrading to a premium plan for higher limits.

## Security Notes

- Never commit API keys to version control
- Use environment variables or secure secret management for API keys
- The server validates all inputs to prevent injection attacks
