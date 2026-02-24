"""AlphaVantage MCP Server implementation."""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import httpx
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries

from mcp.server.fastmcp import FastMCP

from shared.types import (
    CompanyOverview,
    NewsArticle,
    TechnicalIndicatorResult,
    BarData,
    MCPResponse,
    CompanyResponse,
    NewsResponse,
    TechnicalIndicatorResponse,
    BarsResponse,
    TechnicalIndicator,
    TimeFrame,
)


logger = logging.getLogger(__name__)

# Valid parameter values
VALID_OUTPUT_SIZES = {"compact", "full"}
SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,10}$")

# Mapping from TimeFrame enum to AlphaVantage interval strings
TIMEFRAME_TO_AV_INTERVAL: Dict[TimeFrame, str] = {
    TimeFrame.MINUTE: "1min",
    TimeFrame.FIVE_MINUTES: "5min",
    TimeFrame.FIFTEEN_MINUTES: "15min",
    TimeFrame.THIRTY_MINUTES: "30min",
    TimeFrame.HOUR: "60min",
    TimeFrame.DAILY: "daily",
    TimeFrame.WEEKLY: "weekly",
    TimeFrame.MONTHLY: "monthly",
}

# Reverse mapping for string inputs
AV_INTERVAL_TO_TIMEFRAME: Dict[str, TimeFrame] = {v: k for k, v in TIMEFRAME_TO_AV_INTERVAL.items()}

# Valid AlphaVantage intervals (for backward compatibility with string inputs)
VALID_INTERVALS = set(TIMEFRAME_TO_AV_INTERVAL.values())

# Valid intraday intervals
VALID_INTRADAY_INTERVALS = {"1min", "5min", "15min", "30min", "60min"}
VALID_INTRADAY_TIMEFRAMES = {
    TimeFrame.MINUTE, TimeFrame.FIVE_MINUTES, TimeFrame.FIFTEEN_MINUTES,
    TimeFrame.THIRTY_MINUTES, TimeFrame.HOUR
}

# Data limits
MAX_FINANCIAL_RECORDS = 20
MAX_PRICE_POINTS = 100
MAX_INDICATOR_POINTS = 100
MAX_NEWS_LIMIT = 50


def _parse_float(value: Any, invalid_values: tuple = ("None", "-", "")) -> Optional[float]:
    """Parse a float value, returning None for invalid values.

    Args:
        value: The value to parse
        invalid_values: Tuple of string values to treat as None

    Returns:
        Parsed float or None if invalid
    """
    if value is None or value in invalid_values:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_int(value: Any, invalid_values: tuple = ("None", "-", "")) -> Optional[int]:
    """Parse an int value, returning None for invalid values."""
    if value is None or value in invalid_values:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _validate_symbol(symbol: str) -> Optional[str]:
    """Validate a stock symbol.

    Returns:
        Error message if invalid, None if valid
    """
    if not symbol:
        return "Symbol cannot be empty"
    if not SYMBOL_PATTERN.match(symbol):
        return f"Invalid symbol format: {symbol}"
    return None


def _check_rate_limit_error(data: Dict[str, Any]) -> Optional[str]:
    """Check if the API response indicates a rate limit error.

    Returns:
        Error message if rate limited, None otherwise
    """
    # AlphaVantage returns a "Note" field when rate limited
    if "Note" in data:
        note = data["Note"]
        if "rate limit" in note.lower() or "api call frequency" in note.lower():
            return "AlphaVantage API rate limit exceeded. Free tier allows 25 requests/day. Please wait or upgrade your plan."
        return note

    # Also check for "Information" field
    if "Information" in data:
        return data["Information"]

    return None


def _parse_news_timestamp(time_str: str) -> datetime:
    """Parse AlphaVantage news timestamp format.

    Args:
        time_str: Timestamp string in format "20231101T120000" or similar

    Returns:
        Parsed datetime object
    """
    if not time_str:
        return datetime.now(timezone.utc)

    try:
        # AlphaVantage format: "20231101T120000"
        # Clean up the string
        clean_str = time_str.replace("T", " ").replace("Z", "")

        # Try parsing with different formats
        for fmt in ["%Y%m%d %H%M%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d"]:
            try:
                return datetime.strptime(clean_str, fmt)
            except ValueError:
                continue

        # Last resort: try fromisoformat
        return datetime.fromisoformat(clean_str)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


class AlphaVantageMCPServer:
    """MCP Server for AlphaVantage market data and analysis."""

    def __init__(self):
        """Initialize the AlphaVantage MCP server."""
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY must be set")

        # Initialize AlphaVantage clients
        self.fundamental_data = FundamentalData(
            key=self.api_key, output_format="pandas"
        )
        self.tech_indicators = TechIndicators(key=self.api_key, output_format="pandas")
        self.time_series = TimeSeries(key=self.api_key, output_format="pandas")

        # Reusable HTTP client for news API
        self._http_client: Optional[httpx.AsyncClient] = None

        # Server configuration from environment
        debug = os.getenv("ALPHAVANTAGE_DEBUG", "false").lower() == "true"
        port = int(os.getenv("ALPHAVANTAGE_PORT", "8002"))
        log_level = os.getenv("ALPHAVANTAGE_LOG_LEVEL", "INFO")
        host = os.getenv("MCP_HOST", "127.0.0.1")

        # Initialize MCP server with host
        self.app = FastMCP(
            "alphavantage-data",
            debug=debug,
            port=port,
            log_level=log_level,
            host=host,
        )
        self._register_tools()

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    def _build_company_overview(self, symbol: str, info: Dict[str, Any]) -> CompanyOverview:
        """Build a CompanyOverview from API response data."""
        return CompanyOverview(
            symbol=symbol,
            name=info.get("Name", ""),
            description=info.get("Description"),
            sector=info.get("Sector"),
            industry=info.get("Industry"),
            market_cap=_parse_float(info.get("MarketCapitalization")),
            pe_ratio=_parse_float(info.get("PERatio")),
            peg_ratio=_parse_float(info.get("PEGRatio")),
            book_value=_parse_float(info.get("BookValue")),
            dividend_per_share=_parse_float(info.get("DividendPerShare")),
            dividend_yield=_parse_float(info.get("DividendYield")),
            eps=_parse_float(info.get("EPS")),
            revenue_per_share_ttm=_parse_float(info.get("RevenuePerShareTTM")),
            profit_margin=_parse_float(info.get("ProfitMargin")),
            operating_margin_ttm=_parse_float(info.get("OperatingMarginTTM")),
            return_on_assets_ttm=_parse_float(info.get("ReturnOnAssetsTTM")),
            return_on_equity_ttm=_parse_float(info.get("ReturnOnEquityTTM")),
            revenue_ttm=_parse_float(info.get("RevenueTTM")),
            gross_profit_ttm=_parse_float(info.get("GrossProfitTTM")),
            diluted_eps_ttm=_parse_float(info.get("DilutedEPSTTM")),
            quarterly_earnings_growth_yoy=_parse_float(info.get("QuarterlyEarningsGrowthYOY")),
            quarterly_revenue_growth_yoy=_parse_float(info.get("QuarterlyRevenueGrowthYOY")),
            analyst_target_price=_parse_float(info.get("AnalystTargetPrice")),
            trailing_pe=_parse_float(info.get("TrailingPE")),
            forward_pe=_parse_float(info.get("ForwardPE")),
            price_to_sales_ratio_ttm=_parse_float(info.get("PriceToSalesRatioTTM")),
            price_to_book_ratio=_parse_float(info.get("PriceToBookRatio")),
            ev_to_revenue=_parse_float(info.get("EVToRevenue")),
            ev_to_ebitda=_parse_float(info.get("EVToEBITDA")),
            beta=_parse_float(info.get("Beta")),
            week_52_high=_parse_float(info.get("52WeekHigh")),
            week_52_low=_parse_float(info.get("52WeekLow")),
            day_50_moving_average=_parse_float(info.get("50DayMovingAverage")),
            day_200_moving_average=_parse_float(info.get("200DayMovingAverage")),
        )

    def _register_tools(self):
        """Register all MCP tools."""

        @self.app.tool()
        async def get_company_overview(symbol: str) -> CompanyResponse:
            """Get company fundamentals and overview.

            Args:
                symbol: Stock symbol to get overview for (e.g., AAPL, MSFT)
            """
            # Validate input
            if error := _validate_symbol(symbol):
                return CompanyResponse(success=False, error=error)

            try:
                symbol = symbol.upper()
                data, _ = self.fundamental_data.get_company_overview(symbol)

                if data.empty:
                    return CompanyResponse(
                        success=False, error=f"No data found for {symbol}"
                    )

                # Convert pandas series to dict
                info = data.iloc[0].to_dict()

                # Check for rate limit
                if rate_error := _check_rate_limit_error(info):
                    return CompanyResponse(success=False, error=rate_error)

                company = self._build_company_overview(symbol, info)
                return CompanyResponse(success=True, data=company)
            except Exception as e:
                return CompanyResponse(success=False, error=str(e))

        @self.app.tool()
        async def get_income_statement(symbol: str) -> MCPResponse:
            """Get company income statement data.

            Args:
                symbol: Stock symbol to get income statement for (e.g., AAPL, MSFT)
            """
            if error := _validate_symbol(symbol):
                return MCPResponse(success=False, error=error)

            try:
                symbol = symbol.upper()
                data, _ = self.fundamental_data.get_income_statement_annual(symbol)

                if data.empty:
                    return MCPResponse(
                        success=False,
                        error=f"No income statement data found for {symbol}",
                    )

                # Limit data to prevent response size issues
                if len(data) > MAX_FINANCIAL_RECORDS:
                    data = data.tail(MAX_FINANCIAL_RECORDS)

                # Convert to list of dictionaries for JSON serialization
                income_data = []
                for index, row in data.iterrows():
                    income_data.append(
                        {
                            "fiscal_date_ending": index,
                            **{k: v for k, v in row.to_dict().items() if v != "None"},
                        }
                    )

                return MCPResponse(success=True, data=income_data)
            except Exception as e:
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def get_balance_sheet(symbol: str) -> MCPResponse:
            """Get company balance sheet data.

            Args:
                symbol: Stock symbol to get balance sheet for (e.g., AAPL, MSFT)
            """
            if error := _validate_symbol(symbol):
                return MCPResponse(success=False, error=error)

            try:
                symbol = symbol.upper()
                data, _ = self.fundamental_data.get_balance_sheet_annual(symbol)

                if data.empty:
                    return MCPResponse(
                        success=False, error=f"No balance sheet data found for {symbol}"
                    )

                # Limit data to prevent response size issues
                if len(data) > MAX_FINANCIAL_RECORDS:
                    data = data.tail(MAX_FINANCIAL_RECORDS)

                # Convert to list of dictionaries for JSON serialization
                balance_data = []
                for index, row in data.iterrows():
                    balance_data.append(
                        {
                            "fiscal_date_ending": index,
                            **{k: v for k, v in row.to_dict().items() if v != "None"},
                        }
                    )

                return MCPResponse(success=True, data=balance_data)
            except Exception as e:
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def get_cash_flow(symbol: str) -> MCPResponse:
            """Get company cash flow statement data.

            Args:
                symbol: Stock symbol to get cash flow for (e.g., AAPL, MSFT)
            """
            if error := _validate_symbol(symbol):
                return MCPResponse(success=False, error=error)

            try:
                symbol = symbol.upper()
                data, _ = self.fundamental_data.get_cash_flow_annual(symbol)

                if data.empty:
                    return MCPResponse(
                        success=False, error=f"No cash flow data found for {symbol}"
                    )

                # Limit data to prevent response size issues
                if len(data) > MAX_FINANCIAL_RECORDS:
                    data = data.tail(MAX_FINANCIAL_RECORDS)

                # Convert to list of dictionaries for JSON serialization
                cash_flow_data = []
                for index, row in data.iterrows():
                    cash_flow_data.append(
                        {
                            "fiscal_date_ending": index,
                            **{k: v for k, v in row.to_dict().items() if v != "None"},
                        }
                    )

                return MCPResponse(success=True, data=cash_flow_data)
            except Exception as e:
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def get_earnings(
            symbol: str, period: Literal["quarterly", "annual"] = "quarterly"
        ) -> MCPResponse:
            """Get quarterly or annual earnings data.

            Args:
                symbol: Stock symbol to get earnings for (e.g., AAPL, MSFT)
                period: 'quarterly' or 'annual' (default: quarterly)
            """
            if error := _validate_symbol(symbol):
                return MCPResponse(success=False, error=error)

            try:
                symbol = symbol.upper()
                if period == "annual":
                    data, _ = self.fundamental_data.get_earnings_annual(symbol)
                else:
                    data, _ = self.fundamental_data.get_earnings_quarterly(symbol)

                if data.empty:
                    return MCPResponse(
                        success=False, error=f"No earnings data found for {symbol}"
                    )

                # Limit data to prevent response size issues
                if len(data) > MAX_FINANCIAL_RECORDS:
                    data = data.tail(MAX_FINANCIAL_RECORDS)

                # Convert to list of dictionaries for JSON serialization
                earnings_data = []
                for index, row in data.iterrows():
                    earnings_data.append(
                        {
                            "fiscal_date_ending": index,
                            **{k: v for k, v in row.to_dict().items() if v != "None"},
                        }
                    )

                return MCPResponse(success=True, data=earnings_data)
            except Exception as e:
                return MCPResponse(success=False, error=str(e))

        @self.app.tool()
        async def get_market_news(
            tickers: Optional[List[str]] = None,
            topics: Optional[List[str]] = None,
            time_from: Optional[str] = None,
            time_to: Optional[str] = None,
            limit: int = 10,
        ) -> NewsResponse:
            """Get latest market news and sentiment.

            Args:
                tickers: List of stock symbols (e.g., ["AAPL", "MSFT"]). Provide at least one ticker for best results.
                topics: List of topics (e.g., ["technology", "earnings", "ipo", "mergers_and_acquisitions"])
                time_from: Start time in YYYYMMDDTHHMM format (e.g., "20240101T0000")
                time_to: End time in YYYYMMDDTHHMM format (e.g., "20240131T2359")
                limit: Maximum number of articles to return (1-50, default: 10)
            """
            # Validate limit
            if limit < 1:
                limit = 1
            elif limit > MAX_NEWS_LIMIT:
                limit = MAX_NEWS_LIMIT

            try:
                # Use httpx to call AlphaVantage news API directly
                url = "https://www.alphavantage.co/query"
                params: Dict[str, Any] = {
                    "function": "NEWS_SENTIMENT",
                    "apikey": self.api_key,
                    "limit": limit,
                }

                if tickers:
                    # Join list into comma-separated string, uppercase
                    params["tickers"] = ",".join(t.upper() for t in tickers)
                if topics:
                    # Join list into comma-separated string
                    params["topics"] = ",".join(topics)
                if time_from:
                    params["time_from"] = time_from
                if time_to:
                    params["time_to"] = time_to

                client = await self._get_http_client()
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                # Check for rate limit
                if rate_error := _check_rate_limit_error(data):
                    return NewsResponse(success=False, error=rate_error)

                if "Error Message" in data:
                    return NewsResponse(success=False, error=data["Error Message"])

                if "feed" not in data:
                    return NewsResponse(success=False, error="No news data found")

                feed = data["feed"]
                if not feed:
                    return NewsResponse(
                        success=False,
                        error="No news articles found. Try specifying tickers (e.g., ['AAPL']) or topics to get relevant news.",
                    )

                articles = []
                for article in feed:
                    news_article = NewsArticle(
                        title=article.get("title", ""),
                        url=article.get("url", ""),
                        time_published=_parse_news_timestamp(article.get("time_published", "")),
                        summary=article.get("summary"),
                        source=article.get("source"),
                        sentiment_score=_parse_float(article.get("overall_sentiment_score")) or 0.0,
                        sentiment_label=article.get("overall_sentiment_label"),
                        topics=[
                            t for t in (
                                topic.get("topic") for topic in article.get("topics", [])
                            ) if t is not None
                        ],
                        tickers=[
                            t for t in (
                                ticker.get("ticker")
                                for ticker in article.get("ticker_sentiment", [])
                            ) if t is not None
                        ],
                    )
                    articles.append(news_article)

                return NewsResponse(success=True, data=articles)
            except httpx.HTTPStatusError as e:
                return NewsResponse(success=False, error=f"HTTP error: {e.response.status_code}")
            except httpx.RequestError as e:
                return NewsResponse(success=False, error=f"Request failed: {str(e)}")
            except Exception as e:
                return NewsResponse(success=False, error=str(e))

        @self.app.tool()
        async def get_technical_indicators(
            symbol: str,
            indicator: TechnicalIndicator,
            timeframe: TimeFrame = TimeFrame.DAILY,
            time_period: int = 14,
        ) -> TechnicalIndicatorResponse:
            """Get technical indicators (RSI, MACD, Bollinger Bands, etc.).

            Args:
                symbol: Stock symbol (e.g., AAPL, MSFT)
                indicator: Technical indicator type. Valid values: RSI, MACD, BBANDS, SMA, EMA, STOCH, ADX, WILLR
                timeframe: Time frame interval. Valid values: 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day, 1Week, 1Month (default: 1Day)
                time_period: Time period for the indicator calculation (default: 14)
            """
            # Validate inputs
            if error := _validate_symbol(symbol):
                return TechnicalIndicatorResponse(success=False, error=error)

            if time_period < 1:
                return TechnicalIndicatorResponse(
                    success=False, error="time_period must be at least 1"
                )

            # Convert TimeFrame enum to AlphaVantage interval string
            av_interval = TIMEFRAME_TO_AV_INTERVAL.get(timeframe)
            if not av_interval:
                return TechnicalIndicatorResponse(
                    success=False,
                    error=f"Invalid timeframe: {timeframe}. Valid options: {', '.join(t.value for t in TimeFrame)}",
                )

            # Get the indicator value (handles both enum and string)
            indicator_value = indicator.value if isinstance(indicator, TechnicalIndicator) else indicator.upper()

            try:
                symbol = symbol.upper()
                indicator_map = {
                    TechnicalIndicator.RSI.value: self.tech_indicators.get_rsi,
                    TechnicalIndicator.MACD.value: self.tech_indicators.get_macd,
                    TechnicalIndicator.BOLLINGER_BANDS.value: self.tech_indicators.get_bbands,
                    TechnicalIndicator.SMA.value: self.tech_indicators.get_sma,
                    TechnicalIndicator.EMA.value: self.tech_indicators.get_ema,
                    TechnicalIndicator.STOCHASTIC.value: self.tech_indicators.get_stoch,
                    TechnicalIndicator.ADX.value: self.tech_indicators.get_adx,
                    TechnicalIndicator.WILLIAMS_R.value: self.tech_indicators.get_willr,
                }

                # Get the indicator function
                indicator_func = indicator_map.get(indicator_value)
                if not indicator_func:
                    valid_indicators = ", ".join(i.name for i in TechnicalIndicator)
                    return TechnicalIndicatorResponse(
                        success=False,
                        error=f"Unsupported indicator: {indicator_value}. Valid options: {valid_indicators}",
                    )

                # Call the indicator function with appropriate parameters
                if indicator_value in (TechnicalIndicator.MACD.value, TechnicalIndicator.STOCHASTIC.value):
                    # These indicators don't use time_period
                    data, _ = indicator_func(symbol, interval=av_interval)
                else:
                    data, _ = indicator_func(
                        symbol, interval=av_interval, time_period=time_period
                    )

                if data.empty:
                    return TechnicalIndicatorResponse(
                        success=False, error=f"No {indicator_value} data found for {symbol}"
                    )

                # Sort data by date to ensure most recent data is at the end
                data = data.sort_index()

                # Limit data to prevent response size issues
                truncated = len(data) > MAX_INDICATOR_POINTS
                if truncated:
                    data = data.tail(MAX_INDICATOR_POINTS)

                # Convert to list of TechnicalIndicatorResult objects
                results = []
                for timestamp, row in data.iterrows():
                    values = {}
                    for col, val in row.to_dict().items():
                        parsed = _parse_float(val)
                        if parsed is not None:
                            values[col] = parsed

                    result = TechnicalIndicatorResult(
                        indicator=indicator_value,
                        symbol=symbol,
                        timestamp=timestamp.to_pydatetime(),
                        values=values,
                    )
                    results.append(result)

                # Add summary statistics if data was truncated
                if truncated and results:
                    summary_values: Dict[str, float] = {
                        "data_points_returned": float(len(results)),
                    }

                    # Add current value and basic statistics
                    latest_values = results[-1].values
                    main_indicator_keys = [
                        k for k in latest_values.keys() if indicator_value.lower() in k.lower()
                    ]
                    if main_indicator_keys:
                        main_key = main_indicator_keys[0]
                        all_values = [
                            r.values.get(main_key)
                            for r in results
                            if r.values.get(main_key) is not None
                        ]
                        if all_values:
                            summary_values.update({
                                f"current_{indicator_value.lower()}": float(latest_values[main_key]),
                                f"avg_{indicator_value.lower()}": float(sum(all_values) / len(all_values)),
                                f"min_{indicator_value.lower()}": float(min(all_values)),
                                f"max_{indicator_value.lower()}": float(max(all_values)),
                            })

                    summary_result = TechnicalIndicatorResult(
                        indicator=f"{indicator_value}_SUMMARY",
                        symbol=symbol,
                        timestamp=results[-1].timestamp,
                        values=summary_values,
                        interpretation=f"Data truncated: showing last {MAX_INDICATOR_POINTS} data points.",
                    )
                    results.append(summary_result)

                return TechnicalIndicatorResponse(success=True, data=results)
            except Exception as e:
                return TechnicalIndicatorResponse(success=False, error=str(e))

        @self.app.tool()
        async def get_daily_prices(
            symbol: str, outputsize: str = "compact"
        ) -> BarsResponse:
            """Get daily OHLCV data.

            Args:
                symbol: Stock symbol (e.g., AAPL, MSFT)
                outputsize: 'compact' (last 100 days) or 'full' (20+ years)
            """
            if error := _validate_symbol(symbol):
                return BarsResponse(success=False, error=error)

            if outputsize not in VALID_OUTPUT_SIZES:
                return BarsResponse(
                    success=False,
                    error=f"Invalid outputsize: {outputsize}. Valid options: compact, full",
                )

            try:
                symbol = symbol.upper()
                data, _ = self.time_series.get_daily(symbol, outputsize=outputsize)

                if data.empty:
                    return BarsResponse(
                        success=False, error=f"No daily data found for {symbol}"
                    )

                # Sort data by date to ensure most recent data is at the end
                data = data.sort_index()

                # Limit data to prevent response size issues
                if len(data) > MAX_PRICE_POINTS:
                    data = data.tail(MAX_PRICE_POINTS)

                # Convert to list of BarData objects
                bars = []
                for timestamp, row in data.iterrows():
                    row_dict = row.to_dict()
                    # Use flexible key lookup to handle possible column name variations
                    open_val = _parse_float(row_dict.get("1. open"))
                    high_val = _parse_float(row_dict.get("2. high"))
                    low_val = _parse_float(row_dict.get("3. low"))
                    close_val = _parse_float(row_dict.get("4. close"))
                    volume_raw = row_dict.get("5. volume") if "5. volume" in row_dict else row_dict.get("6. volume")
                    volume_val = _parse_int(volume_raw)

                    if open_val is None or high_val is None or low_val is None or close_val is None or volume_val is None:
                        continue  # Skip bars with missing required fields

                    bar = BarData(
                        symbol=symbol,
                        timestamp=timestamp.to_pydatetime(),
                        open=open_val,
                        high=high_val,
                        low=low_val,
                        close=close_val,
                        volume=volume_val,
                    )
                    bars.append(bar)

                if not bars:
                    return BarsResponse(
                        success=False, error=f"No valid price data found for {symbol}"
                    )

                return BarsResponse(success=True, data=bars)
            except Exception as e:
                return BarsResponse(success=False, error=str(e))

        @self.app.tool()
        async def get_intraday_prices(
            symbol: str,
            timeframe: TimeFrame = TimeFrame.FIVE_MINUTES,
            outputsize: str = "compact",
        ) -> BarsResponse:
            """Get intraday price data (requires premium API key).

            Args:
                symbol: Stock symbol (e.g., AAPL, MSFT)
                timeframe: Time frame interval. Valid values: 1Min, 5Min, 15Min, 30Min, 1Hour (default: 5Min)
                outputsize: 'compact' (last 100 data points) or 'full' (30+ days)
            """
            if error := _validate_symbol(symbol):
                return BarsResponse(success=False, error=error)

            if timeframe not in VALID_INTRADAY_TIMEFRAMES:
                valid_values = ", ".join(t.value for t in VALID_INTRADAY_TIMEFRAMES)
                return BarsResponse(
                    success=False,
                    error=f"Invalid timeframe for intraday: {timeframe.value}. Valid options: {valid_values}",
                )

            if outputsize not in VALID_OUTPUT_SIZES:
                return BarsResponse(
                    success=False,
                    error=f"Invalid outputsize: {outputsize}. Valid options: compact, full",
                )

            # Convert TimeFrame to AlphaVantage interval string
            av_interval = TIMEFRAME_TO_AV_INTERVAL[timeframe]

            try:
                symbol = symbol.upper()
                data, _ = self.time_series.get_intraday(
                    symbol, interval=av_interval, outputsize=outputsize
                )

                if data.empty:
                    return BarsResponse(
                        success=False, error=f"No intraday data found for {symbol}"
                    )

                # Sort data by date to ensure most recent data is at the end
                data = data.sort_index()

                # Limit data to prevent response size issues
                if len(data) > MAX_PRICE_POINTS:
                    data = data.tail(MAX_PRICE_POINTS)

                # Convert to list of BarData objects
                bars = []
                for timestamp, row in data.iterrows():
                    row_dict = row.to_dict()
                    open_val = _parse_float(row_dict.get("1. open"))
                    high_val = _parse_float(row_dict.get("2. high"))
                    low_val = _parse_float(row_dict.get("3. low"))
                    close_val = _parse_float(row_dict.get("4. close"))
                    volume_raw = row_dict.get("5. volume") if "5. volume" in row_dict else row_dict.get("6. volume")
                    volume_val = _parse_int(volume_raw)

                    if open_val is None or high_val is None or low_val is None or close_val is None or volume_val is None:
                        continue

                    bar = BarData(
                        symbol=symbol,
                        timestamp=timestamp.to_pydatetime(),
                        open=open_val,
                        high=high_val,
                        low=low_val,
                        close=close_val,
                        volume=volume_val,
                    )
                    bars.append(bar)

                if not bars:
                    return BarsResponse(
                        success=False, error=f"No valid intraday data found for {symbol}"
                    )

                return BarsResponse(success=True, data=bars)
            except Exception as e:
                return BarsResponse(success=False, error=str(e))


def main():
    """Main entry point for the Alpha Vantage MCP server."""
    server = AlphaVantageMCPServer()
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    server.app.run(transport=transport)


if __name__ == "__main__":
    main()
