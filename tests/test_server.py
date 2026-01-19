"""Tests for AlphaVantage MCP Server."""

from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

import pandas as pd
import pytest

from alphavantage_mcp.server import (
    _parse_float,
    _parse_int,
    _validate_symbol,
    _check_rate_limit_error,
    _parse_news_timestamp,
    TIMEFRAME_TO_AV_INTERVAL,
)
from shared.types import (
    CompanyResponse,
    MCPResponse,
    NewsResponse,
    TechnicalIndicatorResponse,
    BarsResponse,
    TechnicalIndicator,
    TimeFrame,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_float_valid(self):
        """Test parsing valid float values."""
        assert _parse_float("123.45") == 123.45
        assert _parse_float(123.45) == 123.45
        assert _parse_float("0") == 0.0
        assert _parse_float(0) == 0.0

    def test_parse_float_invalid(self):
        """Test parsing invalid float values returns None."""
        assert _parse_float("None") is None
        assert _parse_float("-") is None
        assert _parse_float("") is None
        assert _parse_float(None) is None
        assert _parse_float("invalid") is None

    def test_parse_int_valid(self):
        """Test parsing valid int values."""
        assert _parse_int("123") == 123
        assert _parse_int(123) == 123
        assert _parse_int("123.7") == 123  # Truncates

    def test_parse_int_invalid(self):
        """Test parsing invalid int values returns None."""
        assert _parse_int("None") is None
        assert _parse_int("-") is None
        assert _parse_int(None) is None

    def test_validate_symbol_valid(self):
        """Test validation of valid symbols."""
        assert _validate_symbol("AAPL") is None
        assert _validate_symbol("MSFT") is None
        assert _validate_symbol("BRK.A") is None
        assert _validate_symbol("BRK-B") is None

    def test_validate_symbol_invalid(self):
        """Test validation of invalid symbols."""
        assert _validate_symbol("") is not None
        assert _validate_symbol("TOOLONGSYMBOL") is not None
        assert _validate_symbol("INVALID@SYMBOL") is not None
        assert _validate_symbol("HAS SPACE") is not None

    def test_check_rate_limit_error_rate_limited(self):
        """Test rate limit detection."""
        data = {"Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute."}
        error = _check_rate_limit_error(data)
        assert error is not None
        assert "rate limit" in error.lower()

    def test_check_rate_limit_error_information(self):
        """Test Information field detection."""
        data = {"Information": "Invalid API call"}
        error = _check_rate_limit_error(data)
        assert error == "Invalid API call"

    def test_check_rate_limit_error_none(self):
        """Test no rate limit error."""
        data = {"feed": []}
        assert _check_rate_limit_error(data) is None

    def test_parse_news_timestamp_valid(self):
        """Test parsing valid news timestamps."""
        result = _parse_news_timestamp("20231101T120000")
        assert result.year == 2023
        assert result.month == 11
        assert result.day == 1

    def test_parse_news_timestamp_empty(self):
        """Test parsing empty timestamp returns current time."""
        result = _parse_news_timestamp("")
        assert result is not None
        # Should be close to now
        assert (datetime.now() - result.replace(tzinfo=None)).total_seconds() < 5

    def test_parse_news_timestamp_invalid(self):
        """Test parsing invalid timestamp returns current time."""
        result = _parse_news_timestamp("invalid")
        assert result is not None


class TestAlphaVantageMCPServerInit:
    """Tests for server initialization."""

    def test_server_init_success(self, server):
        """Test successful server initialization."""
        assert server.api_key == "test_api_key"
        assert server.app is not None

    def test_server_init_missing_api_key(self, monkeypatch):
        """Test server initialization fails without API key."""
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ALPHA_VANTAGE_API_KEY must be set"):
            from alphavantage_mcp.server import AlphaVantageMCPServer
            AlphaVantageMCPServer()


class TestGetCompanyOverview:
    """Tests for get_company_overview tool."""

    @pytest.mark.asyncio
    async def test_get_company_overview_success(self, server):
        """Test successful company overview retrieval."""
        tools = server.app._tool_manager._tools
        get_company_overview = tools["get_company_overview"].fn

        result = await get_company_overview(symbol="AAPL")

        assert isinstance(result, CompanyResponse)
        assert result.success is True
        assert result.data is not None
        assert result.data.symbol == "AAPL"
        assert result.data.name == "Apple Inc"
        assert result.data.sector == "Technology"
        assert result.data.market_cap == 3000000000000.0
        assert result.data.pe_ratio == 28.5

    @pytest.mark.asyncio
    async def test_get_company_overview_empty_data(self, server):
        """Test company overview with no data."""
        server.fundamental_data.get_company_overview.return_value = (
            pd.DataFrame(),
            None,
        )

        tools = server.app._tool_manager._tools
        get_company_overview = tools["get_company_overview"].fn

        result = await get_company_overview(symbol="INVALID")

        assert isinstance(result, CompanyResponse)
        assert result.success is False
        assert "No data found" in result.error

    @pytest.mark.asyncio
    async def test_get_company_overview_handles_none_values(self, server):
        """Test that None string values are converted properly."""
        server.fundamental_data.get_company_overview.return_value = (
            pd.DataFrame([{
                "Name": "Test Corp",
                "PERatio": "None",
                "MarketCapitalization": "None",
                "EPS": "-",
                "Beta": "None",
            }]),
            None,
        )

        tools = server.app._tool_manager._tools
        get_company_overview = tools["get_company_overview"].fn

        result = await get_company_overview(symbol="TEST")

        assert result.success is True
        assert result.data.pe_ratio is None
        assert result.data.market_cap is None
        assert result.data.eps is None
        assert result.data.beta is None

    @pytest.mark.asyncio
    async def test_get_company_overview_invalid_symbol(self, server):
        """Test company overview with invalid symbol."""
        tools = server.app._tool_manager._tools
        get_company_overview = tools["get_company_overview"].fn

        result = await get_company_overview(symbol="")

        assert result.success is False
        assert "Symbol cannot be empty" in result.error

    @pytest.mark.asyncio
    async def test_get_company_overview_symbol_normalized(self, server):
        """Test that symbol is normalized to uppercase."""
        tools = server.app._tool_manager._tools
        get_company_overview = tools["get_company_overview"].fn

        await get_company_overview(symbol="aapl")

        # Verify the call was made with uppercase symbol
        server.fundamental_data.get_company_overview.assert_called_with("AAPL")


class TestGetIncomeStatement:
    """Tests for get_income_statement tool."""

    @pytest.mark.asyncio
    async def test_get_income_statement_success(self, server):
        """Test successful income statement retrieval."""
        tools = server.app._tool_manager._tools
        get_income_statement = tools["get_income_statement"].fn

        result = await get_income_statement(symbol="AAPL")

        assert isinstance(result, MCPResponse)
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 2
        assert "totalRevenue" in result.data[0]

    @pytest.mark.asyncio
    async def test_get_income_statement_empty_data(self, server):
        """Test income statement with no data."""
        server.fundamental_data.get_income_statement_annual.return_value = (
            pd.DataFrame(),
            None,
        )

        tools = server.app._tool_manager._tools
        get_income_statement = tools["get_income_statement"].fn

        result = await get_income_statement(symbol="INVALID")

        assert result.success is False
        assert "No income statement data found" in result.error

    @pytest.mark.asyncio
    async def test_get_income_statement_invalid_symbol(self, server):
        """Test income statement with invalid symbol."""
        tools = server.app._tool_manager._tools
        get_income_statement = tools["get_income_statement"].fn

        result = await get_income_statement(symbol="INVALID@SYMBOL")

        assert result.success is False
        assert "Invalid symbol format" in result.error


class TestGetBalanceSheet:
    """Tests for get_balance_sheet tool."""

    @pytest.mark.asyncio
    async def test_get_balance_sheet_success(self, server):
        """Test successful balance sheet retrieval."""
        tools = server.app._tool_manager._tools
        get_balance_sheet = tools["get_balance_sheet"].fn

        result = await get_balance_sheet(symbol="AAPL")

        assert isinstance(result, MCPResponse)
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_get_balance_sheet_empty_data(self, server):
        """Test balance sheet with no data."""
        server.fundamental_data.get_balance_sheet_annual.return_value = (
            pd.DataFrame(),
            None,
        )

        tools = server.app._tool_manager._tools
        get_balance_sheet = tools["get_balance_sheet"].fn

        result = await get_balance_sheet(symbol="INVALID")

        assert result.success is False
        assert "No balance sheet data found" in result.error

    @pytest.mark.asyncio
    async def test_get_balance_sheet_truncation(self, server):
        """Test that balance sheet data is truncated."""
        # Create large dataset
        large_data = pd.DataFrame({
            "totalAssets": ["100"] * 30,
        }, index=[f"2020-{i:02d}-01" for i in range(1, 31)])

        server.fundamental_data.get_balance_sheet_annual.return_value = (large_data, None)

        tools = server.app._tool_manager._tools
        get_balance_sheet = tools["get_balance_sheet"].fn

        result = await get_balance_sheet(symbol="AAPL")

        assert result.success is True
        assert len(result.data) == 20  # MAX_FINANCIAL_RECORDS


class TestGetCashFlow:
    """Tests for get_cash_flow tool."""

    @pytest.mark.asyncio
    async def test_get_cash_flow_success(self, server):
        """Test successful cash flow retrieval."""
        tools = server.app._tool_manager._tools
        get_cash_flow = tools["get_cash_flow"].fn

        result = await get_cash_flow(symbol="AAPL")

        assert isinstance(result, MCPResponse)
        assert result.success is True
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_get_cash_flow_empty_data(self, server):
        """Test cash flow with no data."""
        server.fundamental_data.get_cash_flow_annual.return_value = (
            pd.DataFrame(),
            None,
        )

        tools = server.app._tool_manager._tools
        get_cash_flow = tools["get_cash_flow"].fn

        result = await get_cash_flow(symbol="INVALID")

        assert result.success is False
        assert "No cash flow data found" in result.error

    @pytest.mark.asyncio
    async def test_get_cash_flow_truncation(self, server):
        """Test that cash flow data is truncated."""
        large_data = pd.DataFrame({
            "operatingCashflow": ["100"] * 30,
        }, index=[f"2020-{i:02d}-01" for i in range(1, 31)])

        server.fundamental_data.get_cash_flow_annual.return_value = (large_data, None)

        tools = server.app._tool_manager._tools
        get_cash_flow = tools["get_cash_flow"].fn

        result = await get_cash_flow(symbol="AAPL")

        assert result.success is True
        assert len(result.data) == 20  # MAX_FINANCIAL_RECORDS


class TestGetEarnings:
    """Tests for get_earnings tool."""

    @pytest.mark.asyncio
    async def test_get_earnings_quarterly_success(self, server):
        """Test successful quarterly earnings retrieval."""
        tools = server.app._tool_manager._tools
        get_earnings = tools["get_earnings"].fn

        result = await get_earnings(symbol="AAPL", period="quarterly")

        assert isinstance(result, MCPResponse)
        assert result.success is True
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_get_earnings_annual_success(self, server):
        """Test successful annual earnings retrieval."""
        tools = server.app._tool_manager._tools
        get_earnings = tools["get_earnings"].fn

        result = await get_earnings(symbol="AAPL", period="annual")

        assert isinstance(result, MCPResponse)
        assert result.success is True
        server.fundamental_data.get_earnings_annual.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_earnings_empty_data(self, server):
        """Test earnings with no data."""
        server.fundamental_data.get_earnings_quarterly.return_value = (
            pd.DataFrame(),
            None,
        )

        tools = server.app._tool_manager._tools
        get_earnings = tools["get_earnings"].fn

        result = await get_earnings(symbol="INVALID")

        assert result.success is False
        assert "No earnings data found" in result.error


class TestGetDailyPrices:
    """Tests for get_daily_prices tool."""

    @pytest.mark.asyncio
    async def test_get_daily_prices_success(self, server):
        """Test successful daily prices retrieval."""
        tools = server.app._tool_manager._tools
        get_daily_prices = tools["get_daily_prices"].fn

        result = await get_daily_prices(symbol="AAPL")

        assert isinstance(result, BarsResponse)
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 5
        assert result.data[0].symbol == "AAPL"
        assert result.data[0].open > 0
        assert result.data[0].volume > 0

    @pytest.mark.asyncio
    async def test_get_daily_prices_empty_data(self, server):
        """Test daily prices with no data."""
        server.time_series.get_daily.return_value = (pd.DataFrame(), None)

        tools = server.app._tool_manager._tools
        get_daily_prices = tools["get_daily_prices"].fn

        result = await get_daily_prices(symbol="INVALID")

        assert result.success is False
        assert "No daily data found" in result.error

    @pytest.mark.asyncio
    async def test_get_daily_prices_outputsize_parameter(self, server):
        """Test daily prices with different output sizes."""
        tools = server.app._tool_manager._tools
        get_daily_prices = tools["get_daily_prices"].fn

        await get_daily_prices(symbol="AAPL", outputsize="full")

        server.time_series.get_daily.assert_called_with("AAPL", outputsize="full")

    @pytest.mark.asyncio
    async def test_get_daily_prices_invalid_outputsize(self, server):
        """Test daily prices with invalid outputsize."""
        tools = server.app._tool_manager._tools
        get_daily_prices = tools["get_daily_prices"].fn

        result = await get_daily_prices(symbol="AAPL", outputsize="invalid")

        assert result.success is False
        assert "Invalid outputsize" in result.error


class TestGetIntradayPrices:
    """Tests for get_intraday_prices tool."""

    @pytest.mark.asyncio
    async def test_get_intraday_prices_success(self, server):
        """Test successful intraday prices retrieval."""
        tools = server.app._tool_manager._tools
        get_intraday_prices = tools["get_intraday_prices"].fn

        result = await get_intraday_prices(symbol="AAPL")

        assert isinstance(result, BarsResponse)
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 5

    @pytest.mark.asyncio
    async def test_get_intraday_prices_with_timeframe(self, server):
        """Test intraday prices with specific timeframe."""
        tools = server.app._tool_manager._tools
        get_intraday_prices = tools["get_intraday_prices"].fn

        await get_intraday_prices(symbol="AAPL", timeframe=TimeFrame.FIFTEEN_MINUTES)

        server.time_series.get_intraday.assert_called_with(
            "AAPL", interval="15min", outputsize="compact"
        )

    @pytest.mark.asyncio
    async def test_get_intraday_prices_empty_data(self, server):
        """Test intraday prices with no data."""
        server.time_series.get_intraday.return_value = (pd.DataFrame(), None)

        tools = server.app._tool_manager._tools
        get_intraday_prices = tools["get_intraday_prices"].fn

        result = await get_intraday_prices(symbol="INVALID")

        assert result.success is False
        assert "No intraday data found" in result.error

    @pytest.mark.asyncio
    async def test_get_intraday_prices_invalid_timeframe(self, server):
        """Test intraday prices with non-intraday timeframe."""
        tools = server.app._tool_manager._tools
        get_intraday_prices = tools["get_intraday_prices"].fn

        result = await get_intraday_prices(symbol="AAPL", timeframe=TimeFrame.DAILY)

        assert result.success is False
        assert "Invalid timeframe for intraday" in result.error


class TestGetTechnicalIndicators:
    """Tests for get_technical_indicators tool."""

    @pytest.mark.asyncio
    async def test_get_rsi_success(self, server):
        """Test successful RSI retrieval."""
        tools = server.app._tool_manager._tools
        get_technical_indicators = tools["get_technical_indicators"].fn

        result = await get_technical_indicators(
            symbol="AAPL", indicator=TechnicalIndicator.RSI, timeframe=TimeFrame.DAILY, time_period=14
        )

        assert isinstance(result, TechnicalIndicatorResponse)
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 5
        assert result.data[0].indicator == "RSI"
        assert result.data[0].symbol == "AAPL"
        assert "RSI" in result.data[0].values

    @pytest.mark.asyncio
    async def test_get_macd_success(self, server):
        """Test successful MACD retrieval."""
        tools = server.app._tool_manager._tools
        get_technical_indicators = tools["get_technical_indicators"].fn

        result = await get_technical_indicators(
            symbol="AAPL", indicator=TechnicalIndicator.MACD, timeframe=TimeFrame.DAILY
        )

        assert result.success is True
        assert result.data is not None
        assert result.data[0].indicator == "MACD"
        # MACD doesn't use time_period parameter
        server.tech_indicators.get_macd.assert_called_with("AAPL", interval="daily")

    @pytest.mark.asyncio
    async def test_get_bbands_success(self, server):
        """Test successful Bollinger Bands retrieval."""
        tools = server.app._tool_manager._tools
        get_technical_indicators = tools["get_technical_indicators"].fn

        result = await get_technical_indicators(
            symbol="AAPL", indicator=TechnicalIndicator.BOLLINGER_BANDS, timeframe=TimeFrame.DAILY, time_period=20
        )

        assert result.success is True
        assert result.data is not None
        server.tech_indicators.get_bbands.assert_called_with(
            "AAPL", interval="daily", time_period=20
        )

    @pytest.mark.asyncio
    async def test_indicator_empty_data(self, server):
        """Test indicator with no data."""
        server.tech_indicators.get_rsi.return_value = (pd.DataFrame(), None)

        tools = server.app._tool_manager._tools
        get_technical_indicators = tools["get_technical_indicators"].fn

        result = await get_technical_indicators(symbol="INVALID", indicator=TechnicalIndicator.RSI)

        assert result.success is False
        assert "No RSI data found" in result.error

    @pytest.mark.asyncio
    async def test_indicator_with_different_timeframes(self, server):
        """Test indicator with different timeframes."""
        tools = server.app._tool_manager._tools
        get_technical_indicators = tools["get_technical_indicators"].fn

        # Test with weekly timeframe
        await get_technical_indicators(
            symbol="AAPL", indicator=TechnicalIndicator.RSI, timeframe=TimeFrame.WEEKLY
        )
        server.tech_indicators.get_rsi.assert_called_with("AAPL", interval="weekly", time_period=14)

        # Test with intraday timeframe
        await get_technical_indicators(
            symbol="AAPL", indicator=TechnicalIndicator.SMA, timeframe=TimeFrame.FIVE_MINUTES
        )
        server.tech_indicators.get_sma.assert_called_with("AAPL", interval="5min", time_period=14)

    @pytest.mark.asyncio
    async def test_indicator_invalid_time_period(self, server):
        """Test indicator with invalid time period."""
        tools = server.app._tool_manager._tools
        get_technical_indicators = tools["get_technical_indicators"].fn

        result = await get_technical_indicators(
            symbol="AAPL", indicator=TechnicalIndicator.RSI, time_period=0
        )

        assert result.success is False
        assert "time_period must be at least 1" in result.error

    @pytest.mark.asyncio
    async def test_all_indicators(self, server):
        """Test that all TechnicalIndicator enum values are supported."""
        tools = server.app._tool_manager._tools
        get_technical_indicators = tools["get_technical_indicators"].fn

        for indicator in TechnicalIndicator:
            result = await get_technical_indicators(symbol="AAPL", indicator=indicator)
            # All should succeed (mocked data is available)
            assert result.success is True, f"Failed for indicator: {indicator}"


class TestGetMarketNews:
    """Tests for get_market_news tool."""

    @pytest.mark.asyncio
    async def test_get_market_news_success(self, server, mock_news_response):
        """Test successful news retrieval."""
        tools = server.app._tool_manager._tools
        get_market_news = tools["get_market_news"].fn

        with patch.object(server, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_news_response
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await get_market_news(tickers=["AAPL"], limit=10)

        assert isinstance(result, NewsResponse)
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 2
        assert result.data[0].title == "Apple Reports Strong Q4 Earnings"
        assert result.data[0].sentiment_score == 0.75

    @pytest.mark.asyncio
    async def test_get_market_news_multiple_tickers(self, server, mock_news_response):
        """Test news retrieval with multiple tickers."""
        tools = server.app._tool_manager._tools
        get_market_news = tools["get_market_news"].fn

        with patch.object(server, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_news_response
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await get_market_news(tickers=["AAPL", "MSFT", "GOOGL"])

        assert result.success is True
        # Verify tickers are joined with commas
        call_args = mock_client.get.call_args
        assert call_args.kwargs["params"]["tickers"] == "AAPL,MSFT,GOOGL"

    @pytest.mark.asyncio
    async def test_get_market_news_with_topics(self, server, mock_news_response):
        """Test news retrieval with topic filter."""
        tools = server.app._tool_manager._tools
        get_market_news = tools["get_market_news"].fn

        with patch.object(server, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_news_response
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await get_market_news(topics=["technology", "earnings"])

        assert result.success is True
        # Verify topics are joined with commas
        call_args = mock_client.get.call_args
        assert call_args.kwargs["params"]["topics"] == "technology,earnings"

    @pytest.mark.asyncio
    async def test_get_market_news_api_error(self, server):
        """Test news retrieval with API error."""
        tools = server.app._tool_manager._tools
        get_market_news = tools["get_market_news"].fn

        with patch.object(server, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"Error Message": "API limit exceeded"}
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await get_market_news(tickers=["AAPL"])

        assert result.success is False
        assert "API limit exceeded" in result.error

    @pytest.mark.asyncio
    async def test_get_market_news_no_feed(self, server):
        """Test news retrieval with no feed in response."""
        tools = server.app._tool_manager._tools
        get_market_news = tools["get_market_news"].fn

        with patch.object(server, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {}
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await get_market_news(tickers=["AAPL"])

        assert result.success is False
        assert "No news data found" in result.error

    @pytest.mark.asyncio
    async def test_get_market_news_rate_limited(self, server):
        """Test news retrieval when rate limited."""
        tools = server.app._tool_manager._tools
        get_market_news = tools["get_market_news"].fn

        with patch.object(server, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "Note": "Thank you for using Alpha Vantage! Our API call frequency limit is reached."
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await get_market_news(tickers=["AAPL"])

        assert result.success is False
        assert "rate limit" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_market_news_limit_clamped(self, server, mock_news_response):
        """Test that news limit is clamped to valid range."""
        tools = server.app._tool_manager._tools
        get_market_news = tools["get_market_news"].fn

        with patch.object(server, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_news_response
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Test limit > 50 gets clamped
            await get_market_news(limit=100)
            call_args = mock_client.get.call_args
            assert call_args.kwargs["params"]["limit"] == 50

            # Test limit < 1 gets clamped
            await get_market_news(limit=-5)
            call_args = mock_client.get.call_args
            assert call_args.kwargs["params"]["limit"] == 1


class TestErrorHandling:
    """Tests for error handling across all tools."""

    @pytest.mark.asyncio
    async def test_company_overview_exception(self, server):
        """Test exception handling in company overview."""
        server.fundamental_data.get_company_overview.side_effect = Exception(
            "Network error"
        )

        tools = server.app._tool_manager._tools
        get_company_overview = tools["get_company_overview"].fn

        result = await get_company_overview(symbol="AAPL")

        assert result.success is False
        assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_daily_prices_exception(self, server):
        """Test exception handling in daily prices."""
        server.time_series.get_daily.side_effect = Exception("API timeout")

        tools = server.app._tool_manager._tools
        get_daily_prices = tools["get_daily_prices"].fn

        result = await get_daily_prices(symbol="AAPL")

        assert result.success is False
        assert "API timeout" in result.error

    @pytest.mark.asyncio
    async def test_technical_indicator_exception(self, server):
        """Test exception handling in technical indicators."""
        server.tech_indicators.get_rsi.side_effect = Exception("Invalid symbol")

        tools = server.app._tool_manager._tools
        get_technical_indicators = tools["get_technical_indicators"].fn

        result = await get_technical_indicators(symbol="INVALID", indicator=TechnicalIndicator.RSI)

        assert result.success is False
        assert "Invalid symbol" in result.error


class TestDataTruncation:
    """Tests for data truncation logic."""

    @pytest.mark.asyncio
    async def test_daily_prices_truncation(self, server):
        """Test that daily prices are truncated to 100 points."""
        # Create large dataset
        dates = pd.date_range(end=datetime.now(), periods=200, freq="D")
        large_data = pd.DataFrame({
            "1. open": [175.0] * 200,
            "2. high": [178.0] * 200,
            "3. low": [174.0] * 200,
            "4. close": [177.5] * 200,
            "5. volume": [50000000] * 200,
        }, index=dates)

        server.time_series.get_daily.return_value = (large_data, None)

        tools = server.app._tool_manager._tools
        get_daily_prices = tools["get_daily_prices"].fn

        result = await get_daily_prices(symbol="AAPL")

        assert result.success is True
        assert len(result.data) == 100  # Truncated to max_points

    @pytest.mark.asyncio
    async def test_technical_indicator_truncation_with_summary(self, server):
        """Test that technical indicators include summary when truncated."""
        # Create dataset with more than 100 points
        dates = pd.date_range(end=datetime.now(), periods=150, freq="D")
        large_data = pd.DataFrame({
            "RSI": [65.0] * 150,
        }, index=dates)

        server.tech_indicators.get_rsi.return_value = (large_data, None)

        tools = server.app._tool_manager._tools
        get_technical_indicators = tools["get_technical_indicators"].fn

        result = await get_technical_indicators(symbol="AAPL", indicator=TechnicalIndicator.RSI)

        assert result.success is True
        # Should have 100 data points plus summary
        assert len(result.data) == 101
        # Last item should be the summary
        assert result.data[-1].indicator == "RSI_SUMMARY"
        assert "data_points_returned" in result.data[-1].values
