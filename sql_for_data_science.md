# SQL Questions

> 🎯 **Data Science Interview Questions & Answers** — Part of the [complete interview prep series](./README.md)


## Table of Contents

- [SQL operations](#)
- [String Manipulations](#)
- [Join and Sub-queries](#) 

---

Consider you have a `worker` table with following fields:
- first_name
- last_name
- salary
- worker_id(Primary Key)
- department
- department_name

Along with you have some meta data tables like `title` and `bonus` which contains following fields:

- `title`

## SQL Questions

### Q: How do you retrieve the first name from the Worker table using an alias "WORKER NAME"?

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT first_name AS worker_name FROM Worker;
```

</details>

---

### Q: How can you convert the first name from the Worker table to uppercase?

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT UPPER(first_name) AS first_name FROM Worker;
```

</details>

---

### Q: What SQL query would you use to fetch distinct department names from the Worker table?

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT DISTINCT department FROM Worker;
```

</details>

---

### Q: How can you select the first three characters of the first name from the Worker table?

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT SUBSTRING(first_name, 1, 3) AS first_name FROM Worker;
```

</details>

---

### Q: Write a query to find the position of 's' in the first name "Manish" within the Worker table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT POSITION('s' IN first_name) AS position_of_s FROM Worker WHERE first_name = 'Manish';
```

</details>

---

### Q: How do you trim whitespace from the right side of the first name in the Worker table?

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT RTRIM(first_name) AS trimmed_first_name FROM Worker;
```

</details>

---

### Q: Write a query to remove whitespace from the left side of the department field in the Worker table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT LTRIM(department) AS trimmed_department FROM Worker;
```

</details>

---

### Q: How can you fetch unique department names from the Worker table and display their lengths?

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT DISTINCT department, LENGTH(department) AS department_length FROM Worker;
```

</details>

---

### Q: What query would replace 'a' with 'A' in the first name from the Worker table?

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT REPLACE(first_name, 'a', 'A') AS first_name FROM Worker;
```

</details>

---

### Q: How do you concatenate the first name and last name from the Worker table into a single column "COMPLETE NAME"?

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT CONCAT(first_name, ' ', last_name) AS complete_name FROM Worker;
```

</details>

---

### Q: Write a query to list all worker details from the Worker table ordered by first name in ascending order.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker ORDER BY first_name ASC;
```

</details>

---

### Q: How can you list all worker details from the Worker table ordered by first name ascending and department descending?

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker ORDER BY first_name ASC, department DESC;
```

</details>

---

### Q: Write a query to fetch details for Workers with the first names "

Manish" and "Arhan" from the Worker table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE first_name IN ('Manish', 'Arhan');
```

</details>

---

### Q: Write a query to list details of workers excluding first names "Manish" and "Arhan" from the Worker table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE first_name NOT IN ('Manish', 'Arhan');
```

</details>

---

### Q: Write a query to fetch details of Workers with the department name "Admin".

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE department_name = 'Admin';
```

</details>

---

### Q: Write a query to fetch details of Workers whose first name contains 'a'.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE first_name LIKE '%a%';
```

</details>

---

### Q: Write a query to fetch details of Workers whose first name ends with 'a'.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE first_name LIKE '%a';
```

</details>

---

### Q: Write a query to fetch details of Workers whose first name ends with 'h' and contains six alphabets.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE first_name LIKE '%h' AND CHAR_LENGTH(first_name) = 6;
```

</details>

---

### Q: Write a query to fetch details of Workers whose salary lies between 100000 and 500000.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE salary BETWEEN 100000 AND 500000;
```

</details>

---

### Q: Write a query to list Workers who joined in February 2014.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
-- Assuming there's a joining_date field in the Worker table, which is not listed in the initial table information.
    -- SELECT  FROM Worker WHERE YEAR(joining_date) = 2014 AND MONTH(joining_date) = 2;
    -- This answer is based on an assumed field not explicitly mentioned in the table schema provided.
```

</details>

---

### Q: Write a query to fetch the count of employees working in the department 'Admin'.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT COUNT() FROM Worker WHERE department_name = 'Admin';
```

</details>

---

### Q: Write a query to fetch worker names with salaries between 50000 and 100000.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT CONCAT(first_name, ' ', last_name) AS full_name, salary FROM Worker WHERE salary BETWEEN 50000 AND 100000;
```

</details>

---

### Q: Write a query to fetch the number of workers for each department in descending order.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT department, COUNT() AS num_workers FROM Worker GROUpBY department ORDER BY num_workers DESC;
```

</details>

---

### Q: Write a query to list details of Workers who are also Managers, assuming a title table contains info about worker titles.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT w. FROM Worker w JOIN title t ON w.worker_id = t.worker_id WHERE t.worker_title = 'Manager';
```

</details>

---

### Q: Write a query to count the number of titles in the organization of different types.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT worker_title, COUNT() FROM title GROUpBY worker_title HAVING COUNT() > 1;
```

</details>

---

### Q: Write a query to show only odd rows from the Worker table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE MOD(worker_id, 2) = 1;
```

</details>

---

### Q: Write a query to show only even rows from the Worker table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE MOD(worker_id, 2) = 0;
```

</details>

---

### Q: Write a query to clone a new table from another table (e.g., worker_clone from worker).

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
-- Step1: Create a clone table with the same structure as Worker
    CREATE TABLE worker_clone LIKE Worker;
    -- Step2: Copy all data from Worker to worker_clone
    INSERT INTO worker_clone SELECT  FROM Worker;
```

</details>

---

### Q: Write a query to fetch intersecting records of two tables (worker and worker_clone).

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT w. FROM Worker w INNER JOIN worker_clone wc ON w.worker_id = wc.worker_id;
```

</details>

---

### Q: Write a query to show records from one table that another table does not have.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT w. FROM Worker w LEFT JOIN worker_clone wc ON w.worker_id = wc.worker_id WHERE wc.worker_id IS NULL;
```

</details>

---

### Q: Write a query to show the current date and time.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT NOW();
```

</details>

---

### Q: Write a query to show the topn (e.g., 5) records of a table ordered by descending salary.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT 

 FROM Worker ORDER BY salary DESC LIMIT 5;
```

</details>

---

### Q: Write a query to determine the nth (e.g., 5th) highest salary from a table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT DISTINCT salary FROM Worker ORDER BY salary DESC LIMIT 1 OFFSET 4;
```

</details>

---

### Q: Write a query to find the 5th highest salary without using the LIMIT keyword.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT salary FROM Worker w1 WHERE 4 = (SELECT COUNT(DISTINCT w2.salary) FROM Worker w2 WHERE w2.salary > w1.salary);
```

</details>

---

### Q: Write a query to list employees with the same salary.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT w1. FROM Worker w1, Worker w2 WHERE w1.salary = w2.salary AND w1.worker_id != w2.worker_id;
```

</details>

---

### Q: Write a query to show the second highest salary from a table using a sub-query.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT MAX(salary) FROM Worker WHERE salary NOT IN (SELECT MAX(salary) FROM Worker);
```

</details>

---

### Q: Write a query to show one row twice in results from a table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE worker_id = (SELECT MIN(worker_id) FROM Worker) UNION ALL SELECT  FROM Worker WHERE worker_id = (SELECT MIN(worker_id) FROM Worker);
```

</details>

---

### Q: Write a query to list worker ids who do not receive a bonus.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT worker_id FROM Worker WHERE worker_id NOT IN (SELECT worker_id FROM bonus);
```

</details>

---

### Q: Write a query to fetch the first 50% records from a table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker WHERE worker_id <= (SELECT FLOOR(COUNT() / 2) FROM Worker);
```

</details>

---

### Q: Write a query to fetch the departments that have less than 4 people in them.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT department, COUNT() AS dept_count FROM Worker GROUpBY department HAVING dept_count < 4;
```

</details>

---

### Q: Write a query to show all departments along with the number of people in there.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT department, COUNT() AS dept_count FROM Worker GROUpBY department;
```

</details>

---

### Q: Write a query to show the last record from a table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker ORDER BY worker_id DESC LIMIT 1;
```

</details>

---

### Q: Write a query to fetch the first row of a table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker ORDER BY worker_id ASC LIMIT 1;
```

</details>

---

### Q: Write a query to fetch the last five records from a table.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT  FROM Worker ORDER BY worker_id DESC LIMIT 5;
```

</details>

---

### Q: Write a query to print the names of employees having the highest salary in each department.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT w.department, w.first_name, w.salary FROM Worker w INNER JOIN (SELECT department, MAX(salary) AS max_salary FROM Worker GROUpBY department) AS dept_max ON w.department = dept_max.department AND w.salary = dept_max.max_salary;
```

</details>

---

### Q: Write a query to fetch three max salaries from a table using a co-related subquery.

<details>
<summary><b>💡 Show Answer</b></summary>

```sql
SELECT DISTINCT salary FROM Worker w1 WHERE 3 >= (SELECT COUNT(DISTINCT w2.salary) FROM Worker w2 WHERE w2.salary >= w1.salary) ORDER BY salary DESC;
```

</details>

---

### Q: Write the order of executions of SQL operations.

<details>
<summary><b>💡 Show Answer</b></summary>

**Execution Order:**

1. FROM
2. ON
3. WHERE
4. GROUP BY
5. Aggregation functions (COUNT, SUM, MIN/MAX, AVG)
6. HAVING
7. SELECT
8. DISTINCT
9. UNION, INTERSECT, EXCEPT
10. ORDER BY
11. LIMIT, TOP

</details>

---

---

[⬆️ Back to Top](#table-of-contents) | [🏠 Back to Main Index](./README.md)
