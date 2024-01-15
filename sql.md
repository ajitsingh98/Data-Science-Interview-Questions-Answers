# Data Science Interview Questions And Answers

Topics
---

- [SQL Questions](#sql-questions)


## SQL Questions

1. Write an SQL query to fetch `FIRST NAME` from `Worker` table using the alias name as `<WORKER NAME>`.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b>first_name</b> as worker_name from <b>Worker</b>;
    </p>
</details>

---

2. Write an SQL query to fetch `FIRST NAME` from `Worker` table in upper case.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b>upper(first_name) as first_name</b> from <b>Worker</b>;
    </p>
</details>

---

3. Write an SQL query to fetch unique values of `DEPARTMENT` from `Worker` table.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b>distinct department</b> from <b>Worker</b>;
    </p>
</details>

---

4. Write an SQL query to print the first three characters of `FIRST NAME` from `Worker` table.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b>SUBSTR(first_name, 1, 3)</b> from <b>Worker</b>;
    </p>
</details>

---

5. Write an SQL query to find the position of the alphabet ('s') in the first name column 'Manish' from `Worker` table.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b>POSITION('s' IN first_name)</b> AS position_of_s FROM <b>Worker</b> WHERE first_name = 'Manish';
    </p>
</details>

---

6. Write an SQL query to print the `FIRST NAME` from `Worker` table after removing white spaces from the right side.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b>RTRIM(first_name)</b> AS trimmed_first_name FROM <b>Worker</b>;
    </p>
</details>

---

7. Write an SQL query to print the `DEPARTMENT` from `Worker` table after removing white spaces from the left
side.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b>LTRIM(department)</b> AS trimmed_department FROM <b>Worker</b>;
    </p>
</details>

---

8. Write an SQL query that fetches the unique values of `DEPARTMENT` from `Worker` table and prints its
length. 

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b>LENGTH(distinct department)</b> AS department_length FROM <b>Worker</b>;
    </p>
</details>

---


9. Write an SQL query to print the `FIRST NAME` from `Worker` table after replacing 'a' with 'A'.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b>REPLACE(first_name, 'a', 'A') </b> as first_name FROM <b>Worker</b>;
    </p>
</details>

---


10. Write an SQL query to print the `FIRST NAME` and `LAST NAME` from Worker table into a single column
`COMPLETE NAME`.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b>REPLACE(first_name, 'a', 'A') </b> as first_name FROM <b>Worker</b>;
    </p>
</details>

---

11. Write an SQL query to print all Worker details from the `Worker` table order by `FIRST_NAME` Ascending.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b> * </b> FROM <b>Worker</b> ORDER BY first_name;
    </p>
</details>

---

12. Write an SQL query to print all Worker details from the `Worker` table order by `FIRST NAME` Ascending and `DEPARTMENT` Descending. 

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b> * </b> FROM <b>Worker</b> ORDER BY first_name ASC, department DESC;
    </p>
</details>

---

13. Write an SQL query to print details for Workers with the `first name` as "Manish" and "Arhan" from `Worker` table.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b> * </b> FROM <b>Worker</b> ORDER BY first_name ASC, department DESC;
    </p>
</details>

---

14. Write an SQL query to print details of workers excluding `first names` as "Manish" and "Arhan" from `Worker` table.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b> * </b> FROM <b>Worker</b> where first_name not in ('Manish', 'Arhan');
    </p>
</details>

---

15. Write an SQL query to print details of Workers with `DEPARTMENT` name as "Admin".

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b> * </b> FROM <b>Worker</b> where department_name not in ('Manish', 'Arhan');
    </p>
</details>

---

16. Write an SQL query to print details of the Workers whose `FIRST NAME` contains 'a'.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b> * </b> FROM <b>Worker</b> where first_name like '%a%';
    </p>
</details>

---

17. Write an SQL query to print details of the Workers whose `FIRST NAME` ends with 'a'.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b> * </b> FROM <b>Worker</b> where first_name like '%a';
    </p>
</details>

---

18. Write an SQL query to print details of the Workers whose `FIRST NAME` ends with 'h' and contains six
alphabets.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
    SELECT <b> * </b> FROM <b>Worker</b> where first_name like '%h' and LENGTH(first_name)=6;
    </p>
</details>

---

19. Write an SQL query to print details of the Workers whose `SALARY` lies between `100000` and `500000`.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
   SELECT * FROM Worker WHERE SALARY BETWEEN 100000 AND 500000;
    </p>
</details>

---

20. Write an SQL query to print details of the Workers who have joined(`joining_date`) in Feb' 2014.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
   SELECT * FROM Worker WHERE YEAR(joining_date)=2014 and MONTH(joining_date)=2;
    </p>
</details>

---

21. Write an SQL query to fetch the count of employees working in the department 'Admin'.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
   SELECT <b>COUNT(*)</b> FROM Worker WHERE YEAR(joining_date)=2014 and MONTH(joining_date)=2;
    </p>
</details>

---

22. Write an SQL query to fetch worker full names with salaries >= `50000` and <= `100000`.

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
   SELECT <b>CONCAT(first_name, ' ', last_name)</b> AS full_name, SALARY FROM Worker WHERE SALARY BETWEEN 50000 AND 100000;
    </p>
</details>

---

23. Write an SQL query to fetch the no. of workers for each department in the descending order. 

<details style='color: red;'><summary><b>Answer</b></summary>
    <p style='color: red'>
   SELECT <b>DEPARTMENT, COUNT(*) AS worker_count</b> FROM Worker GROUP BY DEPARTMENT ORDER BY worker_count DESC;
    </p>
</details>

---

24. Write an SQL query to print details of the Workers who are also Managers.
25. Write an SQL query to fetch number (more than `1`) of same titles in the ORG of different types.
26. Write an SQL query to show only odd rows from a table. 
27. Write an SQL query to show only even rows from a table.
28. Write an SQL query to clone a new table from another table.
29. Write an SQL query to fetch intersecting records of two tables.
30. Write an SQL query to show records from one table that another table does not have.
31. Write an SQL query to show the current date and time. 
32. Write an SQL query to show the top `n` (say `5`) records of a table order by descending salary.
33. Write an SQL query to determine the nth (say n=5) highest salary from a table.
34. Write an SQL query to determine the `5th` highest salary without using LIMIT keyword.
35. Write an SQL query to fetch the list of employees with the same salary. 
36. Write an SQL query to show the second highest salary from a table using sub-query.
37. Write an SQL query to show one row twice in results from a table.
38. Write an SQL query to list worker id who does not get bonus. 
39. Write an SQL query to fetch the first `50%` records from a table. 
40. Write an SQL query to fetch the `departments` that have less than `4` people in it. 
41. Write an SQL query to show all departments along with the number of people in there.
42. Write an SQL query to show the last record from a table.
43. Write an SQL query to fetch the first row of a table.
44. Write an SQL query to fetch the last five records from a table.
45. Write an SQL query to print the name of employees having the highest salary in each department. 
46. Write an SQL query to fetch three max salaries from a table using co-related subquery.