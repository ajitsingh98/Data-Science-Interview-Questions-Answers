# Data Science Interview Questions And Answers

Topics
---

- [SQL Questions](#sql-questions)

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

Q. How do you retrieve the first name from the Worker table using an alias "WORKER NAME"?

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT <b>first_name</b> AS worker_name FROM <b>Worker</b>;
    </p>
</details>

---

Q. How can you convert the first name from the Worker table to uppercase?

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT <b>UPPER(first_name)</b> AS first_name FROM <b>Worker</b>;
    </p>
</details>

---

Q. What SQL query would you use to fetch distinct department names from the Worker table?

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT <b>DISTINCT department</b> FROM <b>Worker</b>;
    </p>
</details>

---

Q. How can you select the first three characters of the first name from the Worker table?

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT <b>SUBSTRING(first_name, 1, 3)</b> AS first_name FROM <b>Worker</b>;
    </p>
</details>

---

Q. Write a query to find the position of 's' in the first name "Manish" within the Worker table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT <b>POSITION('s' IN first_name)</b> AS position_of_s FROM <b>Worker</b> WHERE first_name = 'Manish';
    </p>
</details>

---

Q. How do you trim whitespace from the right side of the first name in the Worker table?

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT <b>RTRIM(first_name)</b> AS trimmed_first_name FROM <b>Worker</b>;
    </p>
</details>

---

7. Write a query to remove whitespace from the left side of the department field in the Worker table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT <b>LTRIM(department)</b> AS trimmed_department FROM <b>Worker</b>;
    </p>
</details>

---

8. How can you fetch unique department names from the Worker table and display their lengths?

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT <b>DISTINCT department</b>, <b>LENGTH(department)</b> AS department_length FROM <b>Worker</b>;
    </p>
</details>

---

9. What query would replace 'a' with 'A' in the first name from the Worker table?

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT <b>REPLACE(first_name, 'a', 'A')</b> AS first_name FROM <b>Worker</b>;
    </p>
</details>

---

10. How do you concatenate the first name and last name from the Worker table into a single column "COMPLETE NAME"?

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT <b>CONCAT(first_name, ' ', last_name)</b> AS complete_name FROM <b>Worker</b>;
    </p>
</details>

---

11. Write a query to list all worker details from the Worker table ordered by first name in ascending order.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> ORDER BY first_name ASC;
    </p>
</details>

---

12. How can you list all worker details from the Worker table ordered by first name ascending and department descending?

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> ORDER BY first_name ASC, department DESC;
    </p>
</details>

---

13. Write a query to fetch details for Workers with the first names "

Manish" and "Arhan" from the Worker table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE first_name IN ('Manish', 'Arhan');
    </p>
</details>

---

14. Write a query to list details of workers excluding first names "Manish" and "Arhan" from the Worker table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE first_name NOT IN ('Manish', 'Arhan');
    </p>
</details>

---

15. Write a query to fetch details of Workers with the department name "Admin".

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE department_name = 'Admin';
    </p>
</details>

---

16. Write a query to fetch details of Workers whose first name contains 'a'.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE first_name LIKE '%a%';
    </p>
</details>

---

17. Write a query to fetch details of Workers whose first name ends with 'a'.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE first_name LIKE '%a';
    </p>
</details>

---

18. Write a query to fetch details of Workers whose first name ends with 'h' and contains six alphabets.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE first_name LIKE '%h' AND CHAR_LENGTH(first_name) = 6;
    </p>
</details>

---

19. Write a query to fetch details of Workers whose salary lies between 100000 and 500000.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE salary BETWEEN 100000 AND 500000;
    </p>
</details>

---

20. Write a query to list Workers who joined in February 2014.

<details ><summary><b>Answer</b></summary>
    <p>
    -- Assuming there's a joining_date field in the Worker table, which is not listed in the initial table information.
    -- SELECT  FROM <b>Worker</b> WHERE YEAR(joining_date) = 2014 AND MONTH(joining_date) = 2;
    -- This answer is based on an assumed field not explicitly mentioned in the table schema provided.
    </p>
</details>

---

21. Write a query to fetch the count of employees working in the department 'Admin'.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT COUNT() FROM <b>Worker</b> WHERE department_name = 'Admin';
    </p>
</details>

---

22. Write a query to fetch worker names with salaries between 50000 and 100000.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT CONCAT(first_name, ' ', last_name) AS full_name, salary FROM <b>Worker</b> WHERE salary BETWEEN 50000 AND 100000;
    </p>
</details>

---

23. Write a query to fetch the number of workers for each department in descending order.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT department, COUNT() AS num_workers FROM <b>Worker</b> GROUpBY department ORDER BY num_workers DESC;
    </p>
</details>

---

24. Write a query to list details of Workers who are also Managers, assuming a title table contains info about worker titles.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT w. FROM <b>Worker w</b> JOIN <b>title t</b> ON w.worker_id = t.worker_id WHERE t.worker_title = 'Manager';
    </p>
</details>

---

25. Write a query to count the number of titles in the organization of different types.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT worker_title, COUNT() FROM <b>title</b> GROUpBY worker_title HAVING COUNT() > 1;
    </p>
</details>

---

26. Write a query to show only odd rows from the Worker table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE MOD(worker_id, 2) = 1;
    </p>
</details>

---

27. Write a query to show only even rows from the Worker table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE MOD(worker_id, 2) = 0;
    </p>
</details>

---

28. Write a query to clone a new table from another table (e.g., worker_clone from worker).

<details ><summary><b>Answer</b></summary>
    <p>
    -- Step1: Create a clone table with the same structure as Worker
    CREATE TABLE <b>worker_clone</b> LIKE <b>Worker</b>;
    -- Step2: Copy all data from Worker to worker_clone
    INSERT INTO <b>worker_clone</b> SELECT  FROM <b>Worker</b>;
    </p>
</details>

---

29. Write a query to fetch intersecting records of two tables (worker and worker_clone).

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT w. FROM <b>Worker w</b> INNER JOIN <b>worker_clone wc</b> ON w.worker_id = wc.worker_id;
    </p>
</details>

---

30. Write a query to show records from one table that another table does not have.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT w. FROM <b>Worker w</b> LEFT JOIN <b>worker_clone wc</b> ON w.worker_id = wc.worker_id WHERE wc.worker_id IS NULL;
    </p>
</details>

---

31. Write a query to show the current date and time.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT NOW();
    </p>
</details>

---

32. Write a query to show the topn (e.g., 5) records of a table ordered by descending salary.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT 

 FROM <b>Worker</b> ORDER BY salary DESC LIMIT 5;
    </p>
</details>

---

33. Write a query to determine the nth (e.g., 5th) highest salary from a table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT DISTINCT salary FROM <b>Worker</b> ORDER BY salary DESC LIMIT 1 OFFSET 4;
    </p>
</details>

---

34. Write a query to find the 5th highest salary without using the LIMIT keyword.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT salary FROM <b>Worker</b> w1 WHERE 4 = (SELECT COUNT(DISTINCT w2.salary) FROM <b>Worker</b> w2 WHERE w2.salary > w1.salary);
    </p>
</details>

---

35. Write a query to list employees with the same salary.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT w1. FROM <b>Worker</b> w1, <b>Worker</b> w2 WHERE w1.salary = w2.salary AND w1.worker_id != w2.worker_id;
    </p>
</details>

---

36. Write a query to show the second highest salary from a table using a sub-query.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT MAX(salary) FROM <b>Worker</b> WHERE salary NOT IN (SELECT MAX(salary) FROM Worker);
    </p>
</details>

---

37. Write a query to show one row twice in results from a table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE worker_id = (SELECT MIN(worker_id) FROM Worker) UNION ALL SELECT  FROM <b>Worker</b> WHERE worker_id = (SELECT MIN(worker_id) FROM Worker);
    </p>
</details>

---

38. Write a query to list worker ids who do not receive a bonus.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT worker_id FROM <b>Worker</b> WHERE worker_id NOT IN (SELECT worker_id FROM <b>bonus</b>);
    </p>
</details>

---

39. Write a query to fetch the first 50% records from a table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> WHERE worker_id <= (SELECT FLOOR(COUNT() / 2) FROM Worker);
    </p>
</details>

---

40. Write a query to fetch the departments that have less than 4 people in them.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT department, COUNT() AS dept_count FROM <b>Worker</b> GROUpBY department HAVING dept_count < 4;
    </p>
</details>

---

41. Write a query to show all departments along with the number of people in there.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT department, COUNT() AS dept_count FROM <b>Worker</b> GROUpBY department;
    </p>
</details>

---

42. Write a query to show the last record from a table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> ORDER BY worker_id DESC LIMIT 1;
    </p>
</details>

---

43. Write a query to fetch the first row of a table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> ORDER BY worker_id ASC LIMIT 1;
    </p>
</details>

---

44. Write a query to fetch the last five records from a table.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT  FROM <b>Worker</b> ORDER BY worker_id DESC LIMIT 5;
    </p>
</details>

---

45. Write a query to print the names of employees having the highest salary in each department.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT w.department, w.first_name, w.salary FROM <b>Worker w</b> INNER JOIN (SELECT department, MAX(salary) AS max_salary FROM Worker GROUpBY department) AS dept_max ON w.department = dept_max.department AND w.salary = dept_max.max_salary;
    </p>
</details>

---

46. Write a query to fetch three max salaries from a table using a co-related subquery.

<details ><summary><b>Answer</b></summary>
    <p>
    SELECT DISTINCT salary FROM <b>Worker w1</b> WHERE 3 >= (SELECT COUNT(DISTINCT w2.salary) FROM <b>Worker w2</b> WHERE w2.salary >= w1.salary) ORDER BY salary DESC;
    </p>
</details>

---