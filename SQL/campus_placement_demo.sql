-- Campus Placement Data Analysis SQL Demo
-- Dataset: campus_placement_data.csv (100,000+ student records)

-- 1. Basic Table Creation and Data Loading
-- First, create the table structure
CREATE DATABASE IF NOT EXISTS campus_analytics;
USE campus_analytics;

CREATE TABLE campus_placement (
    student_id VARCHAR(10) PRIMARY KEY,
    gender ENUM('Male', 'Female'),
    age INT,
    city_tier VARCHAR(10),
    ssc_percentage DECIMAL(5,2),
    ssc_board VARCHAR(20),
    hsc_percentage DECIMAL(5,2),
    hsc_board VARCHAR(20),
    hsc_stream VARCHAR(20),
    degree_percentage DECIMAL(5,2),
    degree_field VARCHAR(20),
    mba_percentage DECIMAL(5,2),
    specialization VARCHAR(30),
    internships_count INT,
    projects_count INT,
    certifications_count INT,
    technical_skills_score DECIMAL(3,1),
    soft_skills_score DECIMAL(3,1),
    aptitude_score DECIMAL(5,1),
    communication_score DECIMAL(3,1),
    work_experience_months INT,
    leadership_roles INT,
    extracurricular_activities INT,
    backlogs INT,
    placed BOOLEAN,
    salary_lpa DECIMAL(6,2)
);

-- Note: In practice, you would load CSV data using:
-- LOAD DATA INFILE 'campus_placement_data.csv' 
-- INTO TABLE campus_placement 
-- FIELDS TERMINATED BY ',' 
-- ENCLOSED BY '"' 
-- LINES TERMINATED BY '\n'
-- IGNORE 1 ROWS;

-- 2. Basic Data Exploration Queries

-- Get total number of students
SELECT COUNT(*) AS total_students FROM campus_placement;

-- Check placement statistics
SELECT 
    placed,
    COUNT(*) as student_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM campus_placement), 2) as percentage
FROM campus_placement 
GROUP BY placed;

-- Average salary of placed students
SELECT ROUND(AVG(salary_lpa), 2) AS avg_salary_lpa 
FROM campus_placement 
WHERE placed = 1;

-- 3. Advanced Analytics Queries

-- Top 10 highest paid students by specialization
SELECT 
    specialization,
    student_id,
    salary_lpa,
    degree_percentage,
    technical_skills_score
FROM campus_placement 
WHERE placed = 1
ORDER BY salary_lpa DESC 
LIMIT 10;

-- Placement rate by degree field
SELECT 
    degree_field,
    COUNT(*) as total_students,
    SUM(placed) as placed_students,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM campus_placement
GROUP BY degree_field
ORDER BY placement_rate DESC;

-- Salary analysis by city tier
SELECT 
    city_tier,
    COUNT(*) as student_count,
    ROUND(AVG(salary_lpa), 2) as avg_salary,
    ROUND(MIN(salary_lpa), 2) as min_salary,
    ROUND(MAX(salary_lpa), 2) as max_salary
FROM campus_placement
WHERE placed = 1
GROUP BY city_tier
ORDER BY avg_salary DESC;

-- Impact of internships on placement
SELECT 
    internships_count,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM campus_placement
GROUP BY internships_count
ORDER BY internships_count;

-- 4. Skills Analysis

-- Technical vs Soft skills correlation with salary
SELECT 
    CASE 
        WHEN technical_skills_score >= 8 THEN 'High Technical'
        WHEN technical_skills_score >= 5 THEN 'Medium Technical'
        ELSE 'Low Technical'
    END as technical_level,
    CASE 
        WHEN soft_skills_score >= 8 THEN 'High Soft Skills'
        WHEN soft_skills_score >= 5 THEN 'Medium Soft Skills'
        ELSE 'Low Soft Skills'
    END as soft_skills_level,
    COUNT(*) as student_count,
    ROUND(AVG(salary_lpa), 2) as avg_salary
FROM campus_placement
WHERE placed = 1
GROUP BY technical_level, soft_skills_level
ORDER BY avg_salary DESC;

-- 5. Academic Performance Analysis

-- SSC/HSC percentage impact on placement
SELECT 
    CASE 
        WHEN ssc_percentage >= 80 THEN '80%+'
        WHEN ssc_percentage >= 70 THEN '70-79%'
        WHEN ssc_percentage >= 60 THEN '60-69%'
        ELSE '<60%'
    END as ssc_range,
    CASE 
        WHEN hsc_percentage >= 80 THEN '80%+'
        WHEN hsc_percentage >= 70 THEN '70-79%'
        WHEN hsc_percentage >= 60 THEN '60-69%'
        ELSE '<60%'
    END as hsc_range,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM campus_placement
GROUP BY ssc_range, hsc_range
ORDER BY placement_rate DESC;

-- 6. Gender and Diversity Analysis

-- Placement statistics by gender
SELECT 
    gender,
    COUNT(*) as total_students,
    SUM(placed) as placed_students,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate,
    ROUND(AVG(salary_lpa), 2) as avg_salary
FROM campus_placement
GROUP BY gender;

-- Age distribution analysis
SELECT 
    age,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM campus_placement
GROUP BY age
ORDER BY age;

-- 7. Predictive Analysis Queries

-- Students with high potential (not yet placed but strong metrics)
SELECT 
    student_id,
    degree_field,
    degree_percentage,
    technical_skills_score,
    internships_count,
    projects_count,
    CASE 
        WHEN degree_percentage >= 75 AND technical_skills_score >= 7 AND internships_count >= 2 
        THEN 'High Potential'
        WHEN degree_percentage >= 65 AND technical_skills_score >= 5 AND internships_count >= 1
        THEN 'Medium Potential'
        ELSE 'Low Potential'
    END as potential_level
FROM campus_placement
WHERE placed = 0
ORDER BY degree_percentage DESC, technical_skills_score DESC;

-- 8. Board and Stream Analysis

-- SSC Board performance comparison
SELECT 
    ssc_board,
    COUNT(*) as student_count,
    ROUND(AVG(ssc_percentage), 2) as avg_ssc_percentage,
    ROUND(AVG(degree_percentage), 2) as avg_degree_percentage,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM campus_placement
GROUP BY ssc_board
ORDER BY placement_rate DESC;

-- HSC Stream impact on placement
SELECT 
    hsc_stream,
    COUNT(*) as student_count,
    ROUND(AVG(hsc_percentage), 2) as avg_hsc_percentage,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate
FROM campus_placement
GROUP BY hsc_stream
ORDER BY placement_rate DESC;

-- 9. Experience and Leadership Analysis

-- Work experience impact on salary
SELECT 
    CASE 
        WHEN work_experience_months = 0 THEN 'No Experience'
        WHEN work_experience_months <= 12 THEN '0-1 Year'
        WHEN work_experience_months <= 24 THEN '1-2 Years'
        ELSE '2+ Years'
    END as experience_level,
    COUNT(*) as student_count,
    ROUND(AVG(salary_lpa), 2) as avg_salary
FROM campus_placement
WHERE placed = 1
GROUP BY experience_level
ORDER BY avg_salary DESC;

-- Leadership roles correlation
SELECT 
    leadership_roles,
    COUNT(*) as student_count,
    SUM(placed) as placed_count,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate,
    ROUND(AVG(salary_lpa), 2) as avg_salary
FROM campus_placement
GROUP BY leadership_roles
ORDER BY leadership_roles;

-- 10. Comprehensive Dashboard Query

-- Executive summary dashboard
SELECT 
    'Overall Statistics' as metric,
    COUNT(*) as total_students,
    SUM(placed) as placed_students,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate,
    ROUND(AVG(salary_lpa), 2) as avg_salary_lpa,
    ROUND(AVG(degree_percentage), 2) as avg_academic_performance
FROM campus_placement

UNION ALL

SELECT 
    'Top Performers (>80% academic)' as metric,
    COUNT(*) as total_students,
    SUM(placed) as placed_students,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate,
    ROUND(AVG(salary_lpa), 2) as avg_salary_lpa,
    ROUND(AVG(degree_percentage), 2) as avg_academic_performance
FROM campus_placement
WHERE degree_percentage > 80

UNION ALL

SELECT 
    'High Skill Candidates' as metric,
    COUNT(*) as total_students,
    SUM(placed) as placed_students,
    ROUND(SUM(placed) * 100.0 / COUNT(*), 2) as placement_rate,
    ROUND(AVG(salary_lpa), 2) as avg_salary_lpa,
    ROUND(AVG(degree_percentage), 2) as avg_academic_performance
FROM campus_placement
WHERE technical_skills_score >= 8 AND soft_skills_score >= 8;

-- Sample data for testing (first 10 records)
-- SELECT * FROM campus_placement LIMIT 10;