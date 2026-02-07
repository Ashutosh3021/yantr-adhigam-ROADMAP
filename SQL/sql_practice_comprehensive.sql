-- Comprehensive SQL Practice Set
-- Multiple Dataset Scenarios and Real-World Examples

-- 1. Cross-Dataset Analysis Setup
CREATE DATABASE IF NOT EXISTS sql_practice;
USE sql_practice;

-- Simulated table structures for practice
-- These represent the actual CSV structures from your Data directory

-- Campus Placement Table
CREATE TABLE campus_placement_practice (
    student_id VARCHAR(10),
    gender VARCHAR(10),
    age INT,
    ssc_percentage DECIMAL(5,2),
    hsc_percentage DECIMAL(5,2),
    degree_percentage DECIMAL(5,2),
    specialization VARCHAR(30),
    mba_percentage DECIMAL(5,2),
    internships_count INT,
    placed BOOLEAN,
    salary_lpa DECIMAL(6,2)
);

-- Microsoft Stock Table
CREATE TABLE stock_prices_practice (
    date DATE,
    close_price DECIMAL(10,2),
    volume BIGINT,
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2)
);

-- Student Placement Table
CREATE TABLE student_placement_practice (
    student_id INT,
    cgpa DECIMAL(3,2),
    iq INT,
    placed BOOLEAN
);

-- 2. Advanced JOIN Queries

-- Cross-analysis: Student academic performance vs placement outcomes
SELECT 
    cp.specialization,
    sp.cgpa,
    cp.degree_percentage,
    cp.placed as campus_placed,
    sp.placed as simple_placed,
    cp.salary_lpa
FROM campus_placement_practice cp
LEFT JOIN student_placement_practice sp 
    ON ROUND(cp.degree_percentage / 10, 2) = sp.cgpa  -- Approximate mapping
WHERE cp.specialization IN ('Marketing & Finance', 'Finance & HR')
LIMIT 20;

-- 3. Complex Aggregation Queries

-- Performance quartile analysis
SELECT 
    'Campus Placement' as dataset,
    specialization,
    COUNT(*) as student_count,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY degree_percentage) as q1_percentage,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY degree_percentage) as median_percentage,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY degree_percentage) as q3_percentage,
    ROUND(AVG(degree_percentage), 2) as mean_percentage
FROM campus_placement_practice
GROUP BY specialization
HAVING COUNT(*) > 50
ORDER BY mean_percentage DESC

UNION ALL

SELECT 
    'Student Placement' as dataset,
    'All Students' as specialization,
    COUNT(*) as student_count,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY cgpa) as q1_cgpa,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY cgpa) as median_cgpa,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY cgpa) as q3_cgpa,
    ROUND(AVG(cgpa), 2) as mean_cgpa
FROM student_placement_practice;

-- 4. Window Functions Practice

-- Ranking students by performance within specializations
SELECT 
    student_id,
    specialization,
    degree_percentage,
    salary_lpa,
    ROW_NUMBER() OVER (PARTITION BY specialization ORDER BY degree_percentage DESC) as rank_by_academics,
    ROW_NUMBER() OVER (PARTITION BY specialization ORDER BY salary_lpa DESC) as rank_by_salary,
    RANK() OVER (PARTITION BY specialization ORDER BY degree_percentage DESC) as dense_rank_by_academics,
    ROUND(PERCENT_RANK() OVER (PARTITION BY specialization ORDER BY degree_percentage), 4) as percentile_rank
FROM campus_placement_practice
WHERE placed = 1
ORDER BY specialization, rank_by_academics
LIMIT 30;

-- 5. Time Series and Trend Analysis

-- Stock price moving averages with lag analysis
WITH stock_moving_avg AS (
    SELECT 
        date,
        close_price,
        ROUND(AVG(close_price) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 2) as ma_20,
        ROUND(AVG(close_price) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW), 2) as ma_50,
        close_price - LAG(close_price, 1) OVER (ORDER BY date) as price_change_1day,
        close_price - LAG(close_price, 5) OVER (ORDER BY date) as price_change_5day,
        close_price - LAG(close_price, 20) OVER (ORDER BY date) as price_change_20day
    FROM stock_prices_practice
)
SELECT 
    date,
    close_price,
    ma_20,
    ma_50,
    CASE 
        WHEN ma_20 > ma_50 THEN 'Bullish Trend'
        WHEN ma_20 < ma_50 THEN 'Bearish Trend'
        ELSE 'Neutral'
    END as trend_signal,
    ROUND(price_change_1day, 2) as daily_change,
    ROUND((price_change_1day / LAG(close_price) OVER (ORDER BY date)) * 100, 2) as daily_change_percent
FROM stock_moving_avg
ORDER BY date DESC
LIMIT 25;

-- 6. Data Quality and Cleaning Practice

-- Identifying and handling NULL/inconsistent values
SELECT 
    'Campus Placement Dataset' as dataset,
    COUNT(*) as total_records,
    COUNT(*) - COUNT(student_id) as missing_student_ids,
    COUNT(*) - COUNT(gender) as missing_gender,
    COUNT(*) - COUNT(degree_percentage) as missing_degree_percentage,
    COUNT(*) - COUNT(placed) as missing_placement_status,
    SUM(CASE WHEN degree_percentage < 0 OR degree_percentage > 100 THEN 1 ELSE 0 END) as invalid_percentage,
    SUM(CASE WHEN salary_lpa < 0 THEN 1 ELSE 0 END) as invalid_salary
FROM campus_placement_practice

UNION ALL

SELECT 
    'Stock Prices Dataset' as dataset,
    COUNT(*) as total_records,
    COUNT(*) - COUNT(date) as missing_dates,
    COUNT(*) - COUNT(close_price) as missing_prices,
    COUNT(*) - COUNT(volume) as missing_volume,
    SUM(CASE WHEN close_price <= 0 THEN 1 ELSE 0 END) as invalid_prices,
    SUM(CASE WHEN volume < 0 THEN 1 ELSE 0 END) as invalid_volume
FROM stock_prices_practice

UNION ALL

SELECT 
    'Student Placement Dataset' as dataset,
    COUNT(*) as total_records,
    COUNT(*) - COUNT(student_id) as missing_ids,
    COUNT(*) - COUNT(cgpa) as missing_cgpa,
    COUNT(*) - COUNT(iq) as missing_iq,
    COUNT(*) - COUNT(placed) as missing_placement,
    SUM(CASE WHEN cgpa < 0 OR cgpa > 10 THEN 1 ELSE 0 END) as invalid_cgpa,
    SUM(CASE WHEN iq < 50 OR iq > 200 THEN 1 ELSE 0 END) as invalid_iq
FROM student_placement_practice;

-- 7. Advanced Filtering and CASE Logic

-- Complex student segmentation
SELECT 
    student_id,
    degree_percentage,
    internships_count,
    specialization,
    salary_lpa,
    CASE 
        WHEN degree_percentage >= 85 AND internships_count >= 3 AND salary_lpa >= 15 THEN 'Elite Performer'
        WHEN degree_percentage >= 75 AND internships_count >= 2 AND salary_lpa >= 10 THEN 'High Performer'
        WHEN degree_percentage >= 65 AND internships_count >= 1 THEN 'Average Performer'
        WHEN degree_percentage >= 55 THEN 'Below Average'
        ELSE 'Needs Improvement'
    END as performance_category,
    CASE 
        WHEN specialization LIKE '%Finance%' THEN 'Finance Domain'
        WHEN specialization LIKE '%HR%' THEN 'HR Domain'
        WHEN specialization LIKE '%Marketing%' THEN 'Marketing Domain'
        ELSE 'Other Domain'
    END as domain_category
FROM campus_placement_practice
WHERE placed = 1
ORDER BY performance_category DESC, salary_lpa DESC;

-- 8. Statistical Analysis Queries

-- Correlation and regression preparation
SELECT 
    ROUND(
        (COUNT(*) * SUM(degree_percentage * internships_count) - SUM(degree_percentage) * SUM(internships_count)) / 
        SQRT((COUNT(*) * SUM(degree_percentage * degree_percentage) - SUM(degree_percentage) * SUM(degree_percentage)) * 
             (COUNT(*) * SUM(internships_count * internships_count) - SUM(internships_count) * SUM(internships_count))), 
        4
    ) as correlation_degree_internships,
    
    ROUND(
        (COUNT(*) * SUM(degree_percentage * salary_lpa) - SUM(degree_percentage) * SUM(salary_lpa)) / 
        (COUNT(*) * SUM(degree_percentage * degree_percentage) - SUM(degree_percentage) * SUM(degree_percentage)), 
        4
    ) as regression_slope_degree_salary,
    
    ROUND(AVG(salary_lpa) - 
        ((COUNT(*) * SUM(degree_percentage * salary_lpa) - SUM(degree_percentage) * SUM(salary_lpa)) / 
         (COUNT(*) * SUM(degree_percentage * degree_percentage) - SUM(degree_percentage) * SUM(degree_percentage))) * AVG(degree_percentage), 
        2
    ) as regression_intercept

FROM campus_placement_practice
WHERE placed = 1;

-- 9. Performance Optimization Practice

-- Query optimization examples with indexes
-- These would be CREATE INDEX statements in practice:
/*
CREATE INDEX idx_campus_specialization ON campus_placement_practice(specialization);
CREATE INDEX idx_campus_placed ON campus_placement_practice(placed);
CREATE INDEX idx_campus_degree_salary ON campus_placement_practice(degree_percentage, salary_lpa);
CREATE INDEX idx_stock_date ON stock_prices_practice(date);
CREATE INDEX idx_student_cgpa ON student_placement_practice(cgpa);
*/

-- Optimized query examples:
SELECT /*+ USE_INDEX(campus_placement_practice, idx_campus_specialization) */
    specialization,
    COUNT(*) as student_count,
    ROUND(AVG(salary_lpa), 2) as avg_salary
FROM campus_placement_practice
WHERE placed = 1
GROUP BY specialization
ORDER BY avg_salary DESC;

-- 10. Data Visualization Preparation

-- Binning and grouping for charts
SELECT 
    CONCAT(FLOOR(degree_percentage / 5) * 5, '-', FLOOR(degree_percentage / 5) * 5 + 4) as percentage_bin,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 1) as placement_rate,
    ROUND(AVG(salary_lpa), 2) as avg_salary
FROM campus_placement_practice
GROUP BY FLOOR(degree_percentage / 5)
ORDER BY FLOOR(degree_percentage / 5);

-- Time-based aggregation for trend charts
SELECT 
    YEAR(date) as year,
    MONTH(date) as month,
    ROUND(AVG(close_price), 2) as avg_price,
    ROUND(MIN(close_price), 2) as min_price,
    ROUND(MAX(close_price), 2) as max_price,
    ROUND(AVG(volume), 0) as avg_volume
FROM stock_prices_practice
GROUP BY YEAR(date), MONTH(date)
ORDER BY year, month;

-- 11. Business Intelligence Dashboard Queries

-- Executive dashboard with multiple metrics
SELECT 
    'Placement Analytics' as dashboard_section,
    metric_name,
    metric_value,
    metric_percentage
FROM (
    SELECT 
        'Total Students' as metric_name,
        CAST(COUNT(*) AS CHAR) as metric_value,
        NULL as metric_percentage
    FROM campus_placement_practice
    
    UNION ALL
    
    SELECT 
        'Placement Rate' as metric_name,
        CAST(SUM(placed)) as metric_value,
        ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as metric_percentage
    FROM campus_placement_practice
    
    UNION ALL
    
    SELECT 
        'Avg Salary (Placed)' as metric_name,
        CONCAT('₹', ROUND(AVG(salary_lpa), 2), ' LPA') as metric_value,
        NULL as metric_percentage
    FROM campus_placement_practice
    WHERE placed = 1
    
    UNION ALL
    
    SELECT 
        'Top Specialization' as metric_name,
        specialization as metric_value,
        ROUND(AVG(salary_lpa), 2) as metric_percentage
    FROM campus_placement_practice
    WHERE placed = 1
    GROUP BY specialization
    ORDER BY AVG(salary_lpa) DESC
    LIMIT 1
) as dashboard_metrics

UNION ALL

SELECT 
    'Stock Market Analytics' as dashboard_section,
    metric_name,
    metric_value,
    metric_percentage
FROM (
    SELECT 
        'Total Trading Days' as metric_name,
        CAST(COUNT(*) AS CHAR) as metric_value,
        NULL as metric_percentage
    FROM stock_prices_practice
    
    UNION ALL
    
    SELECT 
        'Price Range' as metric_name,
        CONCAT('₹', ROUND(MIN(close_price), 2), ' - ₹', ROUND(MAX(close_price), 2)) as metric_value,
        NULL as metric_percentage
    FROM stock_prices_practice
    
    UNION ALL
    
    SELECT 
        'Best Performing Month' as metric_name,
        CONCAT(MONTHNAME(date), ' ', YEAR(date)) as metric_value,
        ROUND(AVG(close_price), 2) as metric_percentage
    FROM stock_prices_practice
    GROUP BY YEAR(date), MONTH(date)
    ORDER BY AVG(close_price) DESC
    LIMIT 1
) as stock_metrics;

-- 12. Data Science Preparation Queries

-- Feature engineering examples
SELECT 
    student_id,
    degree_percentage,
    internships_count,
    CASE WHEN internships_count > 0 THEN 1 ELSE 0 END as has_internship,
    LOG(degree_percentage + 1) as log_degree,  -- Log transformation
    POWER(degree_percentage, 2) as degree_squared,  -- Polynomial feature
    degree_percentage * internships_count as degree_internship_interaction,
    CASE 
        WHEN degree_percentage >= 80 THEN 'High'
        WHEN degree_percentage >= 60 THEN 'Medium'
        ELSE 'Low'
    END as degree_category
FROM campus_placement_practice
WHERE placed = 1
ORDER BY student_id;

-- Sample data verification
-- SELECT 'Campus Placement Sample:' as dataset;
-- SELECT * FROM campus_placement_practice LIMIT 5;
-- 
-- SELECT 'Stock Prices Sample:' as dataset;
-- SELECT * FROM stock_prices_practice ORDER BY date DESC LIMIT 5;
-- 
-- SELECT 'Student Placement Sample:' as dataset;
-- SELECT * FROM student_placement_practice LIMIT 5;