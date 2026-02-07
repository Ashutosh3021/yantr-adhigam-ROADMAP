-- Microsoft Stock Price Analysis SQL Demo
-- Dataset: microsoft_stock_prices_2016_2026_refined.csv

-- 1. Table Creation for Stock Data
CREATE DATABASE IF NOT EXISTS stock_analysis;
USE stock_analysis;

CREATE TABLE microsoft_stock (
    date DATE PRIMARY KEY,
    close_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    open_price DECIMAL(10,4),
    volume BIGINT
);

-- 2. Basic Data Exploration

-- Get date range of data
SELECT 
    MIN(date) as first_date,
    MAX(date) as last_date,
    COUNT(*) as total_trading_days
FROM microsoft_stock;

-- Basic statistics
SELECT 
    ROUND(AVG(close_price), 2) as avg_closing_price,
    ROUND(MIN(close_price), 2) as min_closing_price,
    ROUND(MAX(close_price), 2) as max_closing_price,
    ROUND(AVG(volume), 0) as avg_trading_volume
FROM microsoft_stock;

-- 3. Time Series Analysis

-- Monthly average closing prices
SELECT 
    YEAR(date) as year,
    MONTH(date) as month,
    ROUND(AVG(close_price), 2) as avg_monthly_close,
    ROUND(AVG(volume), 0) as avg_monthly_volume
FROM microsoft_stock
GROUP BY YEAR(date), MONTH(date)
ORDER BY year, month;

-- Yearly performance summary
SELECT 
    YEAR(date) as year,
    ROUND(MIN(close_price), 2) as year_low,
    ROUND(MAX(close_price), 2) as year_high,
    ROUND(AVG(close_price), 2) as avg_price,
    ROUND(SUM(volume), 0) as total_volume,
    COUNT(*) as trading_days
FROM microsoft_stock
GROUP BY YEAR(date)
ORDER BY year;

-- 4. Price Movement Analysis

-- Daily price changes
SELECT 
    date,
    close_price,
    close_price - LAG(close_price) OVER (ORDER BY date) as daily_change,
    ROUND(((close_price - LAG(close_price) OVER (ORDER BY date)) / LAG(close_price) OVER (ORDER BY date)) * 100, 2) as daily_change_percent
FROM microsoft_stock
ORDER BY date DESC
LIMIT 30;

-- Volatility analysis (standard deviation of daily changes)
SELECT 
    YEAR(date) as year,
    ROUND(STDDEV(close_price - LAG(close_price) OVER (ORDER BY date)), 4) as daily_volatility,
    ROUND(AVG(volume), 0) as avg_volume
FROM microsoft_stock
WHERE close_price IS NOT NULL
GROUP BY YEAR(date)
ORDER BY year;

-- 5. Trading Volume Analysis

-- High volume trading days
SELECT 
    date,
    close_price,
    volume,
    ROUND(volume * close_price, 2) as dollar_volume
FROM microsoft_stock
ORDER BY volume DESC
LIMIT 20;

-- Volume trends by month
SELECT 
    YEAR(date) as year,
    MONTH(date) as month,
    ROUND(AVG(volume), 0) as avg_volume,
    ROUND(AVG(close_price), 2) as avg_price,
    COUNT(*) as trading_days
FROM microsoft_stock
GROUP BY YEAR(date), MONTH(date)
ORDER BY avg_volume DESC
LIMIT 20;

-- 6. Technical Indicators

-- Simple Moving Averages (20-day and 50-day)
SELECT 
    date,
    close_price,
    ROUND(AVG(close_price) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 2) as ma_20,
    ROUND(AVG(close_price) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW), 2) as ma_50
FROM microsoft_stock
ORDER BY date DESC
LIMIT 100;

-- Price range analysis
SELECT 
    date,
    close_price,
    high_price,
    low_price,
    ROUND(high_price - low_price, 2) as daily_range,
    ROUND(((high_price - low_price) / close_price) * 100, 2) as range_percentage
FROM microsoft_stock
ORDER BY range_percentage DESC
LIMIT 20;

-- 7. Market Performance Analysis

-- Best and worst performing days
SELECT 
    date,
    close_price,
    ROUND(((close_price - LAG(close_price) OVER (ORDER BY date)) / LAG(close_price) OVER (ORDER BY date)) * 100, 2) as daily_return
FROM microsoft_stock
ORDER BY daily_return DESC
LIMIT 10;

-- Worst performing days
SELECT 
    date,
    close_price,
    ROUND(((close_price - LAG(close_price) OVER (ORDER BY date)) / LAG(close_price) OVER (ORDER BY date)) * 100, 2) as daily_return
FROM microsoft_stock
ORDER BY daily_return ASC
LIMIT 10;

-- 8. Quarterly Analysis

-- Quarterly performance
SELECT 
    CONCAT(YEAR(date), '-Q', QUARTER(date)) as quarter,
    ROUND(MIN(close_price), 2) as quarter_low,
    ROUND(MAX(close_price), 2) as quarter_high,
    ROUND(AVG(close_price), 2) as avg_price,
    COUNT(*) as trading_days
FROM microsoft_stock
GROUP BY YEAR(date), QUARTER(date)
ORDER BY YEAR(date), QUARTER(date);

-- 9. Price Level Analysis

-- Price distribution analysis
SELECT 
    CASE 
        WHEN close_price < 50 THEN '< $50'
        WHEN close_price < 100 THEN '$50-99'
        WHEN close_price < 150 THEN '$100-149'
        WHEN close_price < 200 THEN '$150-199'
        WHEN close_price < 250 THEN '$200-249'
        ELSE '$250+'
    END as price_range,
    COUNT(*) as days_count,
    ROUND(AVG(volume), 0) as avg_volume
FROM microsoft_stock
GROUP BY 
    CASE 
        WHEN close_price < 50 THEN '< $50'
        WHEN close_price < 100 THEN '$50-99'
        WHEN close_price < 150 THEN '$100-149'
        WHEN close_price < 200 THEN '$150-199'
        WHEN close_price < 250 THEN '$200-249'
        ELSE '$250+'
    END
ORDER BY MIN(close_price);

-- 10. Correlation Analysis

-- Volume vs Price movement correlation
SELECT 
    CASE 
        WHEN volume > (SELECT AVG(volume) FROM microsoft_stock) THEN 'High Volume'
        ELSE 'Low Volume'
    END as volume_category,
    ROUND(AVG(close_price), 2) as avg_price,
    COUNT(*) as days_count,
    ROUND(AVG(ABS(close_price - LAG(close_price) OVER (ORDER BY date))), 2) as avg_daily_movement
FROM microsoft_stock
GROUP BY 
    CASE 
        WHEN volume > (SELECT AVG(volume) FROM microsoft_stock) THEN 'High Volume'
        ELSE 'Low Volume'
    END;

-- 11. Trend Analysis

-- Bullish vs Bearish days
SELECT 
    YEAR(date) as year,
    SUM(CASE WHEN close_price > LAG(close_price) OVER (ORDER BY date) THEN 1 ELSE 0 END) as bullish_days,
    SUM(CASE WHEN close_price < LAG(close_price) OVER (ORDER BY date) THEN 1 ELSE 0 END) as bearish_days,
    COUNT(*) as total_days,
    ROUND(SUM(CASE WHEN close_price > LAG(close_price) OVER (ORDER BY date) THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as bullish_percentage
FROM microsoft_stock
GROUP BY YEAR(date)
ORDER BY year;

-- 12. Performance Benchmarks

-- Year-over-year growth
SELECT 
    YEAR(date) as year,
    ROUND(FIRST_VALUE(close_price) OVER (PARTITION BY YEAR(date) ORDER BY date), 2) as year_open,
    ROUND(LAST_VALUE(close_price) OVER (PARTITION BY YEAR(date) ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING), 2) as year_close,
    ROUND(((LAST_VALUE(close_price) OVER (PARTITION BY YEAR(date) ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) - FIRST_VALUE(close_price) OVER (PARTITION BY YEAR(date) ORDER BY date)) / FIRST_VALUE(close_price) OVER (PARTITION BY YEAR(date) ORDER BY date)) * 100, 2) as year_return_percent
FROM microsoft_stock
WHERE DAYOFYEAR(date) = 1 OR date = (SELECT MAX(date) FROM microsoft_stock WHERE YEAR(date) = YEAR(microsoft_stock.date))
GROUP BY YEAR(date)
ORDER BY year;

-- 13. Risk Analysis

-- Maximum drawdown calculation (simplified)
WITH price_with_previous AS (
    SELECT 
        date,
        close_price,
        LAG(close_price) OVER (ORDER BY date) as prev_close,
        MAX(close_price) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING) as running_max
    FROM microsoft_stock
)
SELECT 
    date,
    close_price,
    running_max,
    ROUND(((close_price - running_max) / running_max) * 100, 2) as drawdown_percent
FROM price_with_previous
ORDER BY drawdown_percent ASC
LIMIT 10;

-- 14. Comprehensive Dashboard Query

-- Executive summary dashboard
SELECT 
    'Overall Period' as period,
    MIN(date) as start_date,
    MAX(date) as end_date,
    ROUND(MIN(close_price), 2) as lowest_price,
    ROUND(MAX(close_price), 2) as highest_price,
    ROUND(AVG(close_price), 2) as avg_price,
    ROUND(AVG(volume), 0) as avg_volume,
    COUNT(*) as trading_days
FROM microsoft_stock

UNION ALL

SELECT 
    'Last 30 Days' as period,
    MIN(date) as start_date,
    MAX(date) as end_date,
    ROUND(MIN(close_price), 2) as lowest_price,
    ROUND(MAX(close_price), 2) as highest_price,
    ROUND(AVG(close_price), 2) as avg_price,
    ROUND(AVG(volume), 0) as avg_volume,
    COUNT(*) as trading_days
FROM (SELECT * FROM microsoft_stock ORDER BY date DESC LIMIT 30) as recent_data

UNION ALL

SELECT 
    'Last 90 Days' as period,
    MIN(date) as start_date,
    MAX(date) as end_date,
    ROUND(MIN(close_price), 2) as lowest_price,
    ROUND(MAX(close_price), 2) as highest_price,
    ROUND(AVG(close_price), 2) as avg_price,
    ROUND(AVG(volume), 0) as avg_volume,
    COUNT(*) as trading_days
FROM (SELECT * FROM microsoft_stock ORDER BY date DESC LIMIT 90) as recent_data;

-- Sample data for verification
-- SELECT * FROM microsoft_stock ORDER BY date DESC LIMIT 5;