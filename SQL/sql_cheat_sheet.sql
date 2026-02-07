-- Quick Reference SQL Cheat Sheet
-- Essential SQL Commands for Data Analysis

-- ==========================================
-- DATABASE OPERATIONS
-- ==========================================

-- Create Database
CREATE DATABASE IF NOT EXISTS my_database;

-- Use Database
USE my_database;

-- Drop Database
DROP DATABASE IF EXISTS my_database;

-- ==========================================
-- TABLE CREATION AND MANAGEMENT
-- ==========================================

-- Create Table with Common Data Types
CREATE TABLE students (
    student_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    age INT CHECK (age >= 18),
    gender ENUM('Male', 'Female', 'Other'),
    enrollment_date DATE DEFAULT CURRENT_DATE,
    gpa DECIMAL(3,2),
    is_active BOOLEAN DEFAULT TRUE
);

-- Add Column
ALTER TABLE students ADD COLUMN phone VARCHAR(15);

-- Modify Column
ALTER TABLE students MODIFY COLUMN gpa DECIMAL(4,2);

-- Drop Column
ALTER TABLE students DROP COLUMN phone;

-- Drop Table
DROP TABLE IF EXISTS students;

-- ==========================================
-- BASIC DATA INSERTION
-- ==========================================

-- Insert Single Row
INSERT INTO students (name, email, age, gender, gpa) 
VALUES ('John Doe', 'john@email.com', 20, 'Male', 3.75);

-- Insert Multiple Rows
INSERT INTO students (name, email, age, gender, gpa) VALUES
('Alice Smith', 'alice@email.com', 19, 'Female', 3.92),
('Bob Johnson', 'bob@email.com', 21, 'Male', 3.45),
('Carol Brown', 'carol@email.com', 20, 'Female', 3.88);

-- ==========================================
-- BASIC SELECT QUERIES
-- ==========================================

-- Select All Columns
SELECT * FROM students;

-- Select Specific Columns
SELECT name, email, gpa FROM students;

-- Select with Aliases
SELECT name AS student_name, gpa AS grade_point_average FROM students;

-- ==========================================
-- FILTERING DATA (WHERE CLAUSE)
-- ==========================================

-- Basic Comparisons
SELECT * FROM students WHERE age = 20;
SELECT * FROM students WHERE gpa > 3.5;
SELECT * FROM students WHERE age >= 18 AND age <= 25;
SELECT * FROM students WHERE gpa BETWEEN 3.0 AND 4.0;

-- String Operations
SELECT * FROM students WHERE name LIKE 'A%';          -- Names starting with A
SELECT * FROM students WHERE name LIKE '%son';        -- Names ending with 'son'
SELECT * FROM students WHERE email LIKE '%@gmail.com'; -- Gmail users
SELECT * FROM students WHERE name LIKE 'J_hn';        -- Names like John, Jahn, etc.

-- Multiple Conditions
SELECT * FROM students WHERE age > 18 AND gpa > 3.5;
SELECT * FROM students WHERE gender = 'Female' OR gpa > 3.8;
SELECT * FROM students WHERE (age < 20 AND gpa > 3.7) OR (age > 22 AND gpa > 3.5);

-- NULL Checks
SELECT * FROM students WHERE email IS NULL;
SELECT * FROM students WHERE phone IS NOT NULL;

-- ==========================================
-- SORTING DATA (ORDER BY)
-- ==========================================

-- Single Column Sorting
SELECT * FROM students ORDER BY gpa DESC;        -- Highest GPA first
SELECT * FROM students ORDER BY name ASC;        -- Alphabetical order
SELECT * FROM students ORDER BY age;             -- Default ascending

-- Multi-Column Sorting
SELECT * FROM students ORDER BY gpa DESC, age ASC;  -- By GPA, then age

-- Limit Results
SELECT * FROM students ORDER BY gpa DESC LIMIT 10;  -- Top 10 students
SELECT * FROM students LIMIT 5 OFFSET 10;           -- Skip 10, get next 5

-- ==========================================
-- AGGREGATE FUNCTIONS
-- ==========================================

-- Basic Aggregations
SELECT 
    COUNT(*) AS total_students,
    COUNT(email) AS students_with_email,
    AVG(gpa) AS average_gpa,
    MAX(gpa) AS highest_gpa,
    MIN(age) AS youngest_age,
    SUM(age) AS sum_of_ages
FROM students;

-- COUNT with Conditions
SELECT COUNT(*) FROM students WHERE gpa > 3.5;

-- DISTINCT Values
SELECT COUNT(DISTINCT gender) AS gender_count FROM students;
SELECT DISTINCT gender FROM students;

-- ==========================================
-- GROUPING DATA (GROUP BY)
-- ==========================================

-- Group by Single Column
SELECT gender, COUNT(*) as count FROM students GROUP BY gender;

-- Group by Multiple Columns
SELECT gender, age, COUNT(*) as count FROM students GROUP BY gender, age;

-- Group with Aggregates
SELECT 
    gender,
    COUNT(*) as student_count,
    AVG(gpa) as avg_gpa,
    MAX(gpa) as max_gpa
FROM students 
GROUP BY gender;

-- ==========================================
-- FILTERING GROUPS (HAVING)
-- ==========================================

-- HAVING vs WHERE
SELECT gender, AVG(gpa) as avg_gpa
FROM students
GROUP BY gender
HAVING AVG(gpa) > 3.5;  -- Filter groups after aggregation

-- Complex HAVING
SELECT age, COUNT(*) as student_count
FROM students
GROUP BY age
HAVING COUNT(*) > 2 AND AVG(gpa) > 3.0;

-- ==========================================
-- JOINS
-- ==========================================

-- Sample tables for JOIN examples
CREATE TABLE courses (
    course_id INT PRIMARY KEY,
    course_name VARCHAR(100),
    credits INT
);

CREATE TABLE enrollments (
    student_id INT,
    course_id INT,
    grade CHAR(2),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

-- INNER JOIN
SELECT s.name, c.course_name, e.grade
FROM students s
INNER JOIN enrollments e ON s.student_id = e.student_id
INNER JOIN courses c ON e.course_id = c.course_id;

-- LEFT JOIN
SELECT s.name, c.course_name
FROM students s
LEFT JOIN enrollments e ON s.student_id = e.student_id
LEFT JOIN courses c ON e.course_id = c.course_id;

-- RIGHT JOIN
SELECT c.course_name, COUNT(e.student_id) as enrollment_count
FROM courses c
RIGHT JOIN enrollments e ON c.course_id = e.course_id
GROUP BY c.course_name;

-- ==========================================
-- SUBQUERIES
-- ==========================================

-- Subquery in WHERE
SELECT * FROM students 
WHERE gpa > (SELECT AVG(gpa) FROM students);

-- Subquery in SELECT
SELECT 
    name,
    gpa,
    (SELECT AVG(gpa) FROM students) as overall_avg,
    gpa - (SELECT AVG(gpa) FROM students) as difference_from_avg
FROM students;

-- EXISTS Subquery
SELECT * FROM students s
WHERE EXISTS (
    SELECT 1 FROM enrollments e 
    WHERE e.student_id = s.student_id
);

-- ==========================================
-- STRING FUNCTIONS
-- ==========================================

SELECT 
    name,
    UPPER(name) as name_upper,
    LOWER(name) as name_lower,
    LENGTH(name) as name_length,
    SUBSTRING(name, 1, 5) as first_5_chars,
    CONCAT(name, ' - ', email) as name_email,
    REPLACE(email, '@', ' [at] ') as safe_email
FROM students;

-- ==========================================
-- DATE/TIME FUNCTIONS
-- ==========================================

-- Current date/time
SELECT 
    CURRENT_DATE() as today,
    CURRENT_TIME() as now_time,
    NOW() as current_datetime;

-- Date calculations
SELECT 
    name,
    enrollment_date,
    DATEDIFF(CURRENT_DATE(), enrollment_date) as days_enrolled,
    YEAR(enrollment_date) as enrollment_year,
    MONTH(enrollment_date) as enrollment_month
FROM students;

-- ==========================================
-- MATHEMATICAL FUNCTIONS
-- ==========================================

SELECT 
    gpa,
    ROUND(gpa, 1) as rounded_gpa,
    CEIL(gpa) as ceil_gpa,
    FLOOR(gpa) as floor_gpa,
    ABS(gpa - 3.0) as distance_from_3,
    POWER(gpa, 2) as gpa_squared,
    SQRT(gpa) as gpa_sqrt
FROM students;

-- ==========================================
-- CASE STATEMENTS
-- ==========================================

SELECT 
    name,
    gpa,
    CASE 
        WHEN gpa >= 3.8 THEN 'Excellent'
        WHEN gpa >= 3.5 THEN 'Good'
        WHEN gpa >= 3.0 THEN 'Average'
        WHEN gpa >= 2.5 THEN 'Below Average'
        ELSE 'Needs Improvement'
    END as performance_grade
FROM students;

-- ==========================================
-- WINDOW FUNCTIONS
-- ==========================================

-- ROW_NUMBER()
SELECT 
    name,
    gpa,
    ROW_NUMBER() OVER (ORDER BY gpa DESC) as rank_by_gpa
FROM students;

-- RANK() with PARTITION
SELECT 
    gender,
    name,
    gpa,
    RANK() OVER (PARTITION BY gender ORDER BY gpa DESC) as rank_within_gender
FROM students;

-- Running totals
SELECT 
    name,
    gpa,
    SUM(gpa) OVER (ORDER BY gpa DESC) as running_total_gpa
FROM students;

-- ==========================================
-- COMMON TABLE EXPRESSIONS (CTE)
-- ==========================================

WITH high_performers AS (
    SELECT * FROM students WHERE gpa > 3.7
)
SELECT 
    hp.name,
    hp.gpa,
    COUNT(e.course_id) as courses_taken
FROM high_performers hp
LEFT JOIN enrollments e ON hp.student_id = e.student_id
GROUP BY hp.student_id, hp.name, hp.gpa;

-- ==========================================
-- DATA MODIFICATION
-- ==========================================

-- UPDATE
UPDATE students 
SET gpa = 4.0 
WHERE student_id = 1;

-- UPDATE with calculation
UPDATE students 
SET gpa = gpa + 0.1 
WHERE gpa < 3.5;

-- DELETE
DELETE FROM students WHERE student_id = 1;
DELETE FROM students WHERE gpa < 2.0;

-- ==========================================
-- TRANSACTION CONTROL
-- ==========================================

START TRANSACTION;
    UPDATE students SET gpa = 3.9 WHERE student_id = 1;
    INSERT INTO enrollments VALUES (1, 101, 'A');
COMMIT;  -- or ROLLBACK;

-- ==========================================
-- USEFUL TIPS AND BEST PRACTICES
-- ==========================================

-- 1. Always use LIMIT when testing queries
SELECT * FROM students LIMIT 5;

-- 2. Use EXPLAIN to analyze query performance
EXPLAIN SELECT * FROM students WHERE gpa > 3.5;

-- 3. Proper indexing
CREATE INDEX idx_student_gpa ON students(gpa);
CREATE INDEX idx_student_name ON students(name);

-- 4. Handle NULLs explicitly
SELECT COALESCE(phone, 'Not Provided') as contact_number FROM students;

-- 5. Use parameterized queries to prevent SQL injection
-- SELECT * FROM students WHERE student_id = ?;

-- Sample data verification queries
-- SELECT COUNT(*) FROM students;
-- SELECT * FROM students ORDER BY student_id LIMIT 10;