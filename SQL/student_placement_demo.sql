-- Student Placement Prediction SQL Demo
-- Dataset: realistic_placement_data.csv (Simple CGPA, IQ, Placement data)

-- 1. Table Creation
CREATE DATABASE IF NOT EXISTS student_placement;
USE student_placement;

CREATE TABLE placement_data (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    cgpa DECIMAL(3,2),
    iq INT,
    placed BOOLEAN
);

-- Note: In practice, load data with:
-- LOAD DATA INFILE 'realistic_placement_data.csv'
-- INTO TABLE placement_data
-- FIELDS TERMINATED BY ','
-- LINES TERMINATED BY '\n'
-- IGNORE 1 ROWS
-- (cgpa, iq, placed);

-- 2. Basic Data Exploration

-- Overall statistics
SELECT 
    COUNT(*) as total_students,
    SUM(placed) as placed_students,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM placement_data;

-- CGPA distribution
SELECT 
    CASE 
        WHEN cgpa >= 9.0 THEN '9.0+'
        WHEN cgpa >= 8.0 THEN '8.0-8.9'
        WHEN cgpa >= 7.0 THEN '7.0-7.9'
        WHEN cgpa >= 6.0 THEN '6.0-6.9'
        ELSE '<6.0'
    END as cgpa_range,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM placement_data
GROUP BY 
    CASE 
        WHEN cgpa >= 9.0 THEN '9.0+'
        WHEN cgpa >= 8.0 THEN '8.0-8.9'
        WHEN cgpa >= 7.0 THEN '7.0-7.9'
        WHEN cgpa >= 6.0 THEN '6.0-6.9'
        ELSE '<6.0'
    END
ORDER BY MIN(cgpa);

-- IQ distribution
SELECT 
    CASE 
        WHEN iq >= 120 THEN '120+ (High)'
        WHEN iq >= 110 THEN '110-119 (Above Average)'
        WHEN iq >= 100 THEN '100-109 (Average)'
        WHEN iq >= 90 THEN '90-99 (Below Average)'
        ELSE '<90 (Low)'
    END as iq_category,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM placement_data
GROUP BY 
    CASE 
        WHEN iq >= 120 THEN '120+ (High)'
        WHEN iq >= 110 THEN '110-119 (Above Average)'
        WHEN iq >= 100 THEN '100-109 (Average)'
        WHEN iq >= 90 THEN '90-99 (Below Average)'
        ELSE '<90 (Low)'
    END
ORDER BY MIN(iq);

-- 3. Correlation Analysis

-- CGPA vs Placement correlation
SELECT 
    ROUND(AVG(CASE WHEN placed = 1 THEN cgpa ELSE NULL END), 2) as avg_cgpa_placed,
    ROUND(AVG(CASE WHEN placed = 0 THEN cgpa ELSE NULL END), 2) as avg_cgpa_not_placed,
    ROUND(AVG(cgpa), 2) as overall_avg_cgpa
FROM placement_data;

-- IQ vs Placement correlation
SELECT 
    ROUND(AVG(CASE WHEN placed = 1 THEN iq ELSE NULL END), 2) as avg_iq_placed,
    ROUND(AVG(CASE WHEN placed = 0 THEN iq ELSE NULL END), 2) as avg_iq_not_placed,
    ROUND(AVG(iq), 2) as overall_avg_iq
FROM placement_data;

-- Combined factor analysis
SELECT 
    CASE 
        WHEN cgpa >= 8.0 AND iq >= 110 THEN 'High CGPA + High IQ'
        WHEN cgpa >= 8.0 AND iq < 110 THEN 'High CGPA + Low IQ'
        WHEN cgpa < 8.0 AND iq >= 110 THEN 'Low CGPA + High IQ'
        ELSE 'Low CGPA + Low IQ'
    END as profile,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM placement_data
GROUP BY 
    CASE 
        WHEN cgpa >= 8.0 AND iq >= 110 THEN 'High CGPA + High IQ'
        WHEN cgpa >= 8.0 AND iq < 110 THEN 'High CGPA + Low IQ'
        WHEN cgpa < 8.0 AND iq >= 110 THEN 'Low CGPA + High IQ'
        ELSE 'Low CGPA + Low IQ'
    END
ORDER BY placement_rate DESC;

-- 4. Predictive Modeling Queries

-- Students likely to get placed (high CGPA or high IQ)
SELECT 
    student_id,
    cgpa,
    iq,
    CASE 
        WHEN cgpa >= 8.5 OR iq >= 120 THEN 'Very High Chance'
        WHEN cgpa >= 7.5 OR iq >= 110 THEN 'High Chance'
        WHEN cgpa >= 6.5 OR iq >= 100 THEN 'Medium Chance'
        ELSE 'Low Chance'
    END as placement_probability
FROM placement_data
WHERE placed = 0
ORDER BY 
    CASE 
        WHEN cgpa >= 8.5 OR iq >= 120 THEN 1
        WHEN cgpa >= 7.5 OR iq >= 110 THEN 2
        WHEN cgpa >= 6.5 OR iq >= 100 THEN 3
        ELSE 4
    END;

-- Students at risk (low metrics)
SELECT 
    student_id,
    cgpa,
    iq
FROM placement_data
WHERE placed = 0 AND (cgpa < 6.0 OR iq < 95)
ORDER BY cgpa ASC, iq ASC;

-- 5. Threshold Analysis

-- Optimal CGPA threshold for placement
SELECT 
    FLOOR(cgpa * 2) / 2 as cgpa_threshold,  -- Group by 0.5 increments
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM placement_data
GROUP BY FLOOR(cgpa * 2) / 2
ORDER BY cgpa_threshold;

-- Optimal IQ threshold for placement
SELECT 
    FLOOR(iq / 5) * 5 as iq_threshold,  -- Group by 5-point increments
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM placement_data
GROUP BY FLOOR(iq / 5) * 5
ORDER BY iq_threshold;

-- 6. Performance Comparison

-- Top performers vs others
SELECT 
    'Top Performers (CGPA>=8.5 OR IQ>=120)' as category,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM placement_data
WHERE cgpa >= 8.5 OR iq >= 120

UNION ALL

SELECT 
    'Average Performers' as category,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM placement_data
WHERE (cgpa >= 7.0 AND cgpa < 8.5) OR (iq >= 100 AND iq < 120)

UNION ALL

SELECT 
    'Low Performers' as category,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM placement_data
WHERE cgpa < 7.0 AND iq < 100;

-- 7. Statistical Analysis

-- Correlation strength analysis
SELECT 
    ROUND(
        (COUNT(*) * SUM(cgpa * iq) - SUM(cgpa) * SUM(iq)) / 
        SQRT((COUNT(*) * SUM(cgpa * cgpa) - SUM(cgpa) * SUM(cgpa)) * 
             (COUNT(*) * SUM(iq * iq) - SUM(iq) * SUM(iq))), 
        4
    ) as correlation_coefficient
FROM placement_data;

-- Placement probability by CGPA ranges
SELECT 
    CASE 
        WHEN cgpa >= 9.0 THEN '9.0-10.0'
        WHEN cgpa >= 8.0 THEN '8.0-8.9'
        WHEN cgpa >= 7.0 THEN '7.0-7.9'
        WHEN cgpa >= 6.0 THEN '6.0-6.9'
        ELSE 'Below 6.0'
    END as cgpa_group,
    ROUND(AVG(CASE WHEN placed = 1 THEN 1.0 ELSE 0.0 END), 3) as placement_probability,
    COUNT(*) as total_students
FROM placement_data
GROUP BY 
    CASE 
        WHEN cgpa >= 9.0 THEN '9.0-10.0'
        WHEN cgpa >= 8.0 THEN '8.0-8.9'
        WHEN cgpa >= 7.0 THEN '7.0-7.9'
        WHEN cgpa >= 6.0 THEN '6.0-6.9'
        ELSE 'Below 6.0'
    END
ORDER BY MIN(cgpa);

-- 8. Machine Learning Preparation Queries

-- Feature engineering examples
SELECT 
    student_id,
    cgpa,
    iq,
    placed,
    CASE WHEN cgpa >= 8.0 THEN 1 ELSE 0 END as high_cgpa_flag,
    CASE WHEN iq >= 110 THEN 1 ELSE 0 END as high_iq_flag,
    cgpa * iq as cgpa_iq_interaction,
    ROUND((cgpa - 7.0) / 2.0, 2) as normalized_cgpa,  -- Assuming mean=7.0, std=2.0
    ROUND((iq - 105) / 15.0, 2) as normalized_iq      -- Assuming mean=105, std=15
FROM placement_data
ORDER BY student_id;

-- Training/testing split simulation
SELECT 
    student_id,
    cgpa,
    iq,
    placed,
    CASE 
        WHEN student_id % 5 = 0 THEN 'TEST'
        ELSE 'TRAIN'
    END as dataset_type
FROM placement_data
ORDER BY student_id;

-- 9. Business Intelligence Queries

-- Placement success factors ranking
SELECT 
    'CGPA Impact' as factor,
    ROUND(AVG(CASE WHEN placed = 1 THEN cgpa ELSE NULL END) - AVG(CASE WHEN placed = 0 THEN cgpa ELSE NULL END), 2) as difference
FROM placement_data

UNION ALL

SELECT 
    'IQ Impact' as factor,
    ROUND(AVG(CASE WHEN placed = 1 THEN iq ELSE NULL END) - AVG(CASE WHEN placed = 0 THEN iq ELSE NULL END), 2) as difference
FROM placement_data

ORDER BY ABS(difference) DESC;

-- 10. Comprehensive Dashboard

-- Executive summary
SELECT 
    'Total Students' as metric,
    COUNT(*) as value,
    NULL as percentage
FROM placement_data

UNION ALL

SELECT 
    'Placement Rate' as metric,
    SUM(placed) as value,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as percentage
FROM placement_data

UNION ALL

SELECT 
    'Average CGPA (Placed)' as metric,
    ROUND(AVG(CASE WHEN placed = 1 THEN cgpa ELSE NULL END), 2) as value,
    NULL as percentage
FROM placement_data

UNION ALL

SELECT 
    'Average IQ (Placed)' as metric,
    ROUND(AVG(CASE WHEN placed = 1 THEN iq ELSE NULL END), 2) as value,
    NULL as percentage
FROM placement_data

UNION ALL

SELECT 
    'High Potential Students' as metric,
    COUNT(*) as value,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM placement_data), 2) as percentage
FROM placement_data
WHERE cgpa >= 8.0 OR iq >= 115;

-- Sample verification query
-- SELECT * FROM placement_data ORDER BY student_id LIMIT 10;