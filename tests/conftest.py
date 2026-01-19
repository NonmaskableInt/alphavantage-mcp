"""Pytest configuration and fixtures for AlphaVantage MCP Server tests."""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    """Set a dummy API key for all tests."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_api_key")


@pytest.fixture
def mock_company_overview_data():
    """Sample company overview data as returned by AlphaVantage."""
    return pd.DataFrame([{
        "Name": "Apple Inc",
        "Description": "Apple Inc. designs, manufactures, and markets smartphones.",
        "Sector": "Technology",
        "Industry": "Consumer Electronics",
        "MarketCapitalization": "3000000000000",
        "PERatio": "28.5",
        "PEGRatio": "2.1",
        "BookValue": "4.25",
        "DividendPerShare": "0.96",
        "DividendYield": "0.005",
        "EPS": "6.15",
        "RevenuePerShareTTM": "24.32",
        "ProfitMargin": "0.25",
        "OperatingMarginTTM": "0.30",
        "ReturnOnAssetsTTM": "0.20",
        "ReturnOnEquityTTM": "1.47",
        "RevenueTTM": "394000000000",
        "GrossProfitTTM": "170000000000",
        "DilutedEPSTTM": "6.13",
        "QuarterlyEarningsGrowthYOY": "0.05",
        "QuarterlyRevenueGrowthYOY": "0.02",
        "AnalystTargetPrice": "200.00",
        "TrailingPE": "28.5",
        "ForwardPE": "25.0",
        "PriceToSalesRatioTTM": "7.5",
        "PriceToBookRatio": "45.0",
        "EVToRevenue": "7.8",
        "EVToEBITDA": "22.5",
        "Beta": "1.25",
        "52WeekHigh": "199.62",
        "52WeekLow": "124.17",
        "50DayMovingAverage": "178.50",
        "200DayMovingAverage": "165.20",
    }])


@pytest.fixture
def mock_income_statement_data():
    """Sample income statement data."""
    return pd.DataFrame({
        "totalRevenue": ["394000000000", "383000000000"],
        "grossProfit": ["170000000000", "165000000000"],
        "operatingIncome": ["115000000000", "110000000000"],
        "netIncome": ["97000000000", "95000000000"],
    }, index=["2023-09-30", "2022-09-30"])


@pytest.fixture
def mock_balance_sheet_data():
    """Sample balance sheet data."""
    return pd.DataFrame({
        "totalAssets": ["352000000000", "340000000000"],
        "totalLiabilities": ["290000000000", "280000000000"],
        "totalShareholderEquity": ["62000000000", "60000000000"],
    }, index=["2023-09-30", "2022-09-30"])


@pytest.fixture
def mock_cash_flow_data():
    """Sample cash flow data."""
    return pd.DataFrame({
        "operatingCashflow": ["110000000000", "105000000000"],
        "capitalExpenditures": ["-10000000000", "-9500000000"],
        "freeCashFlow": ["100000000000", "95500000000"],
    }, index=["2023-09-30", "2022-09-30"])


@pytest.fixture
def mock_earnings_data():
    """Sample earnings data."""
    return pd.DataFrame({
        "reportedEPS": ["1.53", "1.46", "1.40", "1.52"],
        "estimatedEPS": ["1.50", "1.45", "1.38", "1.50"],
        "surprise": ["0.03", "0.01", "0.02", "0.02"],
    }, index=["2023-09-30", "2023-06-30", "2023-03-31", "2022-12-31"])


@pytest.fixture
def mock_daily_prices_data():
    """Sample daily price data."""
    dates = pd.date_range(end=datetime.now(), periods=5, freq="D")
    return pd.DataFrame({
        "1. open": [175.0, 176.5, 177.0, 178.5, 179.0],
        "2. high": [178.0, 179.0, 180.5, 181.0, 182.0],
        "3. low": [174.0, 175.5, 176.0, 177.5, 178.0],
        "4. close": [177.5, 178.0, 179.5, 180.0, 181.5],
        "5. volume": [50000000, 48000000, 52000000, 49000000, 51000000],
    }, index=dates)


@pytest.fixture
def mock_intraday_prices_data():
    """Sample intraday price data."""
    dates = pd.date_range(end=datetime.now(), periods=5, freq="5min")
    return pd.DataFrame({
        "1. open": [180.0, 180.5, 181.0, 180.75, 181.25],
        "2. high": [180.5, 181.0, 181.5, 181.25, 181.75],
        "3. low": [179.75, 180.25, 180.75, 180.5, 181.0],
        "4. close": [180.25, 180.75, 181.25, 181.0, 181.5],
        "5. volume": [100000, 95000, 110000, 98000, 105000],
    }, index=dates)


@pytest.fixture
def mock_rsi_data():
    """Sample RSI indicator data."""
    dates = pd.date_range(end=datetime.now(), periods=5, freq="D")
    return pd.DataFrame({
        "RSI": [65.5, 68.2, 72.1, 70.5, 67.8],
    }, index=dates)


@pytest.fixture
def mock_macd_data():
    """Sample MACD indicator data."""
    dates = pd.date_range(end=datetime.now(), periods=5, freq="D")
    return pd.DataFrame({
        "MACD": [1.25, 1.35, 1.45, 1.40, 1.50],
        "MACD_Signal": [1.15, 1.22, 1.30, 1.35, 1.40],
        "MACD_Hist": [0.10, 0.13, 0.15, 0.05, 0.10],
    }, index=dates)


@pytest.fixture
def mock_bbands_data():
    """Sample Bollinger Bands indicator data."""
    dates = pd.date_range(end=datetime.now(), periods=5, freq="D")
    return pd.DataFrame({
        "Real Upper Band": [185.0, 186.0, 187.0, 186.5, 187.5],
        "Real Middle Band": [180.0, 181.0, 182.0, 181.5, 182.5],
        "Real Lower Band": [175.0, 176.0, 177.0, 176.5, 177.5],
    }, index=dates)


@pytest.fixture
def mock_news_response():
    """Sample news API response."""
    return {
        "feed": [
            {
                "title": "Apple Reports Strong Q4 Earnings",
                "url": "https://example.com/apple-q4",
                "time_published": "20231101T120000",
                "summary": "Apple Inc. reported better than expected earnings.",
                "source": "Financial Times",
                "overall_sentiment_score": 0.75,
                "overall_sentiment_label": "Bullish",
                "topics": [{"topic": "Earnings"}, {"topic": "Technology"}],
                "ticker_sentiment": [{"ticker": "AAPL"}],
            },
            {
                "title": "Tech Sector Rally Continues",
                "url": "https://example.com/tech-rally",
                "time_published": "20231101T100000",
                "summary": "Technology stocks continue their upward momentum.",
                "source": "Bloomberg",
                "overall_sentiment_score": 0.60,
                "overall_sentiment_label": "Somewhat-Bullish",
                "topics": [{"topic": "Technology"}],
                "ticker_sentiment": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
            },
        ]
    }


@pytest.fixture
def mock_fundamental_data(
    mock_company_overview_data,
    mock_income_statement_data,
    mock_balance_sheet_data,
    mock_cash_flow_data,
    mock_earnings_data,
):
    """Create a mock FundamentalData client."""
    mock = MagicMock()
    mock.get_company_overview.return_value = (mock_company_overview_data, None)
    mock.get_income_statement_annual.return_value = (mock_income_statement_data, None)
    mock.get_balance_sheet_annual.return_value = (mock_balance_sheet_data, None)
    mock.get_cash_flow_annual.return_value = (mock_cash_flow_data, None)
    mock.get_earnings_quarterly.return_value = (mock_earnings_data, None)
    mock.get_earnings_annual.return_value = (mock_earnings_data, None)
    return mock


@pytest.fixture
def mock_time_series(mock_daily_prices_data, mock_intraday_prices_data):
    """Create a mock TimeSeries client."""
    mock = MagicMock()
    mock.get_daily.return_value = (mock_daily_prices_data, None)
    mock.get_intraday.return_value = (mock_intraday_prices_data, None)
    return mock


@pytest.fixture
def mock_tech_indicators(mock_rsi_data, mock_macd_data, mock_bbands_data):
    """Create a mock TechIndicators client."""
    mock = MagicMock()
    mock.get_rsi.return_value = (mock_rsi_data, None)
    mock.get_macd.return_value = (mock_macd_data, None)
    mock.get_bbands.return_value = (mock_bbands_data, None)
    mock.get_sma.return_value = (mock_rsi_data, None)  # Reuse RSI structure
    mock.get_ema.return_value = (mock_rsi_data, None)
    mock.get_stoch.return_value = (mock_macd_data, None)  # Reuse MACD structure
    mock.get_adx.return_value = (mock_rsi_data, None)
    mock.get_willr.return_value = (mock_rsi_data, None)
    return mock


@pytest.fixture
def server(mock_fundamental_data, mock_time_series, mock_tech_indicators):
    """Create a server instance with mocked clients."""
    with patch("alphavantage_mcp.server.FundamentalData", return_value=mock_fundamental_data), \
         patch("alphavantage_mcp.server.TechIndicators", return_value=mock_tech_indicators), \
         patch("alphavantage_mcp.server.TimeSeries", return_value=mock_time_series):
        from alphavantage_mcp.server import AlphaVantageMCPServer
        return AlphaVantageMCPServer()
