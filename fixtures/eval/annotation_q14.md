# Query 14: window functions ranking partitioning

**Domain:** sql
**Query ID:** q_sql_001
**Candidates:** 43
**Grading:** 0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant

---

## [1/43] google_ds_l4_l5_guide

| Field | Value |
|-------|-------|
| **Pages** | 29-29 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.770 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `881ea733-a47c-41b5-aa65-13a3eb31021a` |
| **YOUR GRADE** | ____ |

**Full Text (199 chars):**

```
Chapter 7. 2.1 Window Functions (10 Problems)
Window functions are critical for ranking, time-series analysis, and partition-based aggregations. They appear in approximately 30% of DS SQL interviews.
```

---

## [2/43] AWS Certified Data Engineer Study Guide: Associate (DEA-C01) Exam

| Field | Value |
|-------|-------|
| **Pages** | 567-568 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.765 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `4eae900f-b59b-45fe-9f54-ab1a9a50cf69` |
| **YOUR GRADE** | ____ |

**Full Text (1444 chars):**

```
Window Functions: The Analytical Powerhouse
Window functions represent one of SQL's most powerful features for analytical work, yet they often remain underutilized by analysts who haven't fully explored their capabilities. These functions allow analysts to perform calculations across sets of rows while still returning detailed, row-level data -a capability that transforms how we can analyze trends and patterns in our data. The true elegance of window functions lies in their ability to maintain the granularity of our data while simultaneously providing contextual information. Consider the challenge of analyzing employee salaries within departments. We might want to understand not just individual salaries, but how each employee's compensation compares to their peers:
SELECT department, employee_name, salary, AVG(salary) OVER (PARTITION BY department) as dept_avg_salary, salary - AVG(salary) OVER (PARTITION BY department) as diff_from_avg, PERCENT_RANK() OVER (PARTITION BY department ORDER BY salary) as salary_percentile FROM employees;
This query exemplifies the analytical power of window functions. For each employee, we can see their absolute salary, their department's average, their deviation from this average, and their percentile ranking-all in a single query. The PARTITION BY clause creates these window calculations within the context of each department, allowing for meaningful comparisons within relevant peer groups.
```

---

## [3/43] AWS Certified Data Engineer Study Guide: Associate (DEA-C01) Exam

| Field | Value |
|-------|-------|
| **Pages** | 657-658 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.765 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `ed95e62f-bbaf-46f7-8d26-15fa2ecd776a` |
| **YOUR GRADE** | ____ |

**Full Text (1443 chars):**

```
Window Functions: The Analytical Powerhouse
Window functions represent one of SQL's most powerful features for analytical work, yet they often remain underutilized by analysts who haven't fully explored their capabilities. These functions allow analysts to perform calculations across sets of rows while still returning detailed, row-level data-a capability that transforms how we can analyze trends and patterns in our data. The true elegance of window functions lies in their ability to maintain the granularity of our data while simultaneously providing contextual information. Consider the challenge of analyzing employee salaries within departments. We might want to understand not just individual salaries, but how each employee's compensation compares to their peers:
SELECT department, employee_name, salary, AVG(salary) OVER (PARTITION BY department) as dept_avg_salary, salary - AVG(salary) OVER (PARTITION BY department) as diff_from_avg, PERCENT_RANK() OVER (PARTITION BY department ORDER BY salary) as salary_percentile FROM employees;
This query exemplifies the analytical power of window functions. For each employee, we can see their absolute salary, their department's average, their deviation from this average, and their percentile ranking-all in a single query. The PARTITION BY clause creates these window calculations within the context of each department, allowing for meaningful comparisons within relevant peer groups.
```

---

## [4/43] Data Analysis with Python and PySpark

| Field | Value |
|-------|-------|
| **Pages** | 261-261 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.754 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `216832ef-0a6c-4784-b40f-b6219f7c5700` |
| **YOUR GRADE** | ____ |

**Full Text (1236 chars):**

```
10.6 Summary
- Window functions are functions that are applied over a portion of a data frame called a window frame. They can perform aggregation, ranking, or analytical operations. A window function will return the data frame with the same number of records, unlike its siblings the groupby-aggregate operation and the grouped map UDF.
- A window frame is defined through a window spec. A window spec mandates how the data frame is split ( ), how it's ordered ( ) and how it's partitionBy() orderBy() portioned ( ). rowsBetween()/rangeBetween()
- By default, an unordered window frame will be unbounded, meaning that the window frame will be equal to the window partition for every record. An ordered window frame will be growing to the left, meaning that each record will have a window frame ranging from the first record in the window partition to the current record.
- A window can be bounded by row, meaning that the records included in the window frame are tied to the row boundaries passed as parameters (with the range boundaries added to the row number of the current row), or by range, meaning that the records included in the window frame depend on the value of the current row (with the range boundaries added to the value).
```

---

## [5/43] Vol 4: SQL Mastery

| Field | Value |
|-------|-------|
| **Pages** | 300-300 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.754 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `3a592adc-4619-4a30-b18a-10518e675f38` |
| **YOUR GRADE** | ____ |

**Full Text (231 chars):**

```
Chapter 3 (Tier 2 - Intermediate)
Window functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD), PARTITION BY, Running totals, Moving averages, CTEs, CROSS JOIN, Date functions (DATE_DIFF, DATE_TRUNC), Complex joins, Anti-joins, PIVOT
```

---

## [6/43] Vol 4: SQL Mastery

| Field | Value |
|-------|-------|
| **Pages** | 285-285 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.754 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `c0919b5c-3f90-4fe3-b2b2-fba50ed7e1f0` |
| **YOUR GRADE** | ____ |

**Full Text (417 chars):**

```
CORRECT
```
SELECT name, salary, department, RANK() OVER ( PARTITION BY department ORDER BY salary DESC ) AS rank FROM employees; --Now ranks within each department
```
Why it matters : Without PARTITION BY, the window spans the entire table. This is THE most common window function mistake in Google interviews. Remember : PARTITION BY is like GROUP BY for window functions - it creates separate 'windows' per group.
```

---

## [7/43] Vol 4: SQL Mastery

| Field | Value |
|-------|-------|
| **Pages** | 292-292 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.752 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `eb7f9ae1-5a52-47db-942f-d5790e4b21d8` |
| **YOUR GRADE** | ____ |

**Full Text (450 chars):**

```
D.7.2 Window Functions
- glyph[square] PARTITION BY included if grouping needed
- glyph[square] ROWS BETWEEN uses (N-1) PRECEDING for N-period window
- glyph[square] Correct ranking function (ROW_NUMBER vs RANK vs DENSE_RANK)
- glyph[square] LAST_VALUE has explicit UNBOUNDED FOLLOWING frame
- glyph[square] ROWS vs RANGE choice is intentional (ROWS for physical, RANGE for value-based)
- glyph[square] Window function vs GROUP BY decision is correct
```

---

## [8/43] Data Analysis with Python and PySpark

| Field | Value |
|-------|-------|
| **Pages** | 242-242 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.750 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `612e06bc-973e-4237-9619-bd16caed9199` |
| **YOUR GRADE** | ____ |

**Full Text (1231 chars):**

```
10.2.1 Ranking functions: quick, who's first?
This section covers ranking functions: non-consecutive ranks with ,  consecutive ranks rank() with , percentile ranks with , tiles with , and finally a dense_rank() percent_rank() ntile() bare row number with . Ranking functions are used for getting the top (or bottom) row_number() record for each window partition, or more genreally to get an order according to some columns value. For example, if you wanted to get the for  each  station/month, a top three hottest days ranking function would make this a walk in the park. Because their behavior is quite close to one another, they are better introduced in one fell swoop. Have no fear, I promise it won't read like an old technical manual.
Ranking functions have one sole purpose in life: rank records based on the value of a field. Because of this, we need to order the values within a window. Enter the method for orderBy() windows. In , I create a new window which partition the data listing 10.9 temp_per_month_asc frame  according  to  the column,  ordering  each  record  in  the  partition  according  to  the mo column. Just like when ordering a data frame, will sort the values in count_temp orderBy() ascending order.
TIP
```

---

## [9/43] Vol 4: SQL Mastery

| Field | Value |
|-------|-------|
| **Pages** | 293-293 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.750 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `f57a1087-6a15-4f76-b145-d30f1d857198` |
| **YOUR GRADE** | ____ |

**Full Text (319 chars):**

```
Most Common Mistake
Most Common Mistake in Google L4 Interviews :
Forgetting PARTITION BY in window functions
Practice writing window functions until PARTITION BY becomes automatic:
```
-- Default template for ranking within groups: DENSE_RANK() OVER ( PARTITION BY group_column ORDER BY value_column DESC ) AS rank
```
```

---

## [10/43] Vol 4: SQL Mastery

| Field | Value |
|-------|-------|
| **Pages** | 190-190 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.749 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `301ade94-88ce-4c03-b3e1-94d2a20c6d69` |
| **YOUR GRADE** | ____ |

**Full Text (402 chars):**

```
7.16.3 Common Mistakes
```
Mistake 3: Not Partitioning Window Functions -- WRONG: Computes median across ALL companies, not per company WITH ranked AS ( SELECT id, company, salary, ROW_NUMBER() OVER ( ORDER BY salary) AS rn, --Missing PARTITION BY! COUNT (*) OVER () AS cnt FROM Employee ) --... rest of query
```
Why wrong : Without PARTITION BY company , ranks and counts are global, not per-company.
```

---

## [11/43] Data Analysis with Python and PySpark

| Field | Value |
|-------|-------|
| **Pages** | 232-232 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.740 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `8f061c56-1353-45c0-ad39-963e2f474f5d` |
| **YOUR GRADE** | ____ |

**Full Text (1502 chars):**

```
This chapter covers
UDF ( )  transformations,  both  seen  in  chapter  9.  Both  grouped-aggregate groupBy().apply() methods  and  group  map  UDF  rely  on for  splitting  the  data  frame  based  on  a partitioning predicate. A group aggregate transformation will yield one record per grouping, while a group map UDF allows for any shape of resulting data frame; a window function always keeps the dimensions of the data frame intact. Window functions have a secret weapon in the window that  we  define  within  a  partition:  it  determines  which  records  are  included  in  the frame application of the function.
Window functions are mostly for creating new columns, so they leverage some familiar methods, such  as and .  Because  we  already  are  familiar  with  the  syntax  for select() withColumn() adding columns, I approach this chapter differently. First, we look at how we can emulate a simple window function by relying on some concepts we already know, such as groupbys and joins.  Following  this,  we  get  familiar  with  the  two  components  of  a  window  function:  the window spec and the function. I then apply and dissect the three main types of window functions (summarizing, ranking, and analytical). Equipped with the building blocks of window function application, we break open the window spec by introducing ordered and bounded window, where I  introduce  the  concept  of  a  window  frame.  Finally,  we  go  full  circle  and  introduce  UDF  as window functions.
```

---

## [12/43] Vol 4: SQL Mastery

| Field | Value |
|-------|-------|
| **Pages** | 259-259 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.734 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `9ad685e4-3571-41d2-85e4-ea097b211ed0` |
| **YOUR GRADE** | ____ |

**Full Text (679 chars):**

```
Daily Schedule
- Monday (2 hrs) :
- -Read Chapter 1, Section 1.5 (Window Functions) carefully
- -Problem 26 (Rank Scores) - THE canonical window function problem
- Tuesday (2 hrs) :
- -Problems 27-28 (Nth Highest Salary, Department Highest Salary)
- -Focus on PARTITION BY
- Wednesday (2 hrs) :
- -Problems 29-30 (Department Top 3, Second Highest Salary)
- -Practice 'top N per group' pattern
- Thursday (2 hrs) :
- -Problems 31-33 (Game Play, Friend Requests, Consecutive Numbers)
- Friday (1 hr) :
- -Problems 34-35
- -Review all Week 4 problems
- Weekend (1 hr) :
- Window Function Bootcamp : Write 5 ranking queries from memory
- -Practice explaining PARTITION BY vs GROUP BY
```

---

## [13/43] google_ds_l4_l5_guide

| Field | Value |
|-------|-------|
| **Pages** | 30-30 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.730 |
| **Found In** | hybrid |
| **Chunk ID** | `27f46bba-ccb2-4ebf-9a47-6726ca2656e4` |
| **YOUR GRADE** | ____ |

**Full Text (280 chars):**

```
Problem 2.1.1: ROW_NUMBER Rankings
Difficulty: Easy | Time Target: 12 min
Pattern: Window function with PARTITION BY
DS Context: Common in user activity analysis, product rankings, top-N queries
DataLemur Equivalent: "Spotify Top 5 Artists" (Medium) - uses similar ranking pattern
```

---

## [14/43] Vol 4: SQL Mastery

| Field | Value |
|-------|-------|
| **Pages** | 63-64 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.730 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `554dad99-abe6-4cb5-93bd-c49ce13af31a` |
| **YOUR GRADE** | ____ |

**Full Text (419 chars):**

```
Solution:
Approach: Use ROW_NUMBER() window function with PARTITION BY.
```
WITH ranked_ads AS ( SELECT campaign_id, ad_id, clicks, impressions, clicks * 1.0 / impressions AS ctr, ROW_NUMBER() OVER ( PARTITION BY campaign_id ORDER BY clicks * 1.0 / impressions DESC ) AS rn FROM ad_performance WHERE impressions >= 100 ) SELECT campaign_id, ad_id,
```
```
ctr FROM ranked_ads WHERE rn <= 3 ORDER BY campaign_id, rn;
```
```

---

## [15/43] Volume 2: The Google Data Scientist Interview Workbook

| Field | Value |
|-------|-------|
| **Pages** | 67-67 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.729 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `246cdaf0-c605-4640-bb71-e553d5968055` |
| **YOUR GRADE** | ____ |

**Full Text (797 chars):**

```
29.1. Session Structure: The 80/20 Split
```
1 18:00-18:30 (30 min) - LEARN 2 3 - Watch: "SQL Window Functions Explained" (YouTube, 20 min) 4 - Read: Interview Query window functions guide (10 min) 5 - Key concepts: ROW_NUMBER(), RANK(), LAG(), LEAD() 6 7 18:30-20:30 (120 min) - DO 8 9 - Problem 1: ROW_NUMBER() to rank sales (15 min attempt, 5 min solution, 10 min redo) 10 - Problem 2: LAG() to calculate day-over-day change (20 min) 11 - Problem 3: RANK() with PARTITION BY (20 min) 12 - Problem 4: Combining window functions (25 min) 13 - Problem 5: Complex window function question (25 min) 14 15 20:30-20:45 (15 min) - REVIEW 16 17 - Create Anki flashcards: 18   - "What's difference between ROW_NUMBER() and RANK()?" 19   - "When do you use PARTITION BY?" 20   - "How does LAG() work?"
```
```

---

## [16/43] volume2_google_ds_interview_workbook_FINAL

| Field | Value |
|-------|-------|
| **Pages** | 42-42 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.721 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `bc2e6149-9f10-4a22-bbb1-ee8445f1819b` |
| **YOUR GRADE** | ____ |

**Full Text (732 chars):**

```
Example: Week 1, Monday (SQL Window Functions)
```
18:00-18:30 (30 min) -LEARN -Watch: "SQL Window Functions Explained" (YouTube, 20 min) -Read: Interview Query window functions guide (10 min) -Key concepts: ROW_NUMBER(), RANK(), LAG(), LEAD() 18:30-20:30 (120 min) -DO -Problem 1: ROW_NUMBER() to rank sales (15 min attempt, 5 min solution, 10 min redo) -Problem 2: LAG() to calculate day-over-day change (20 min) -Problem 3: RANK() with PARTITION BY (20 min) -Problem 4: Combining window functions (25 min) -Problem 5: Complex window function question (25 min) 20:30-20:45 (15 min) -REVIEW -Create Anki flashcards: - "What's difference between ROW_NUMBER() and RANK()?" -"When do you use PARTITION BY?" -"How does LAG() work?"
```
```

---

## [17/43] Vol 4: SQL Mastery

| Field | Value |
|-------|-------|
| **Pages** | 21-21 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.717 |
| **Found In** | fts, vector, hybrid, citations |
| **Chunk ID** | `a9282daa-117c-4fdf-95a9-b9739e3724c0` |
| **YOUR GRADE** | ____ |

**Full Text (283 chars):**

```
1.6 Window Functions
```
Interview Tip [
```
Window Functions Critical for Interviews] Window functions are the most important SQL concept for Google interviews . They're like pandas' .transform() and .rank() combined. Master ROW_NUMBER, RANK, LAG, LEAD, and SUM/AVG OVER partitions.
```

---

## [18/43] Deep Learning

| Field | Value |
|-------|-------|
| **Pages** | 638-638 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.715 |
| **Found In** | vector |
| **Chunk ID** | `c8ca4d7e-802b-4a98-98d0-b483c2dc5347` |
| **YOUR GRADE** | ____ |

**Full Text (1093 chars):**

```
18.7 Estimating the Partition Function
While much of this chapter is dedicated to describing methods that avoid needing to compute the intractable partition function Z ( θ ) associated with an undirected graphical model, in this section we discuss several methods for directly estimating the partition function.
Estimating the partition function can be important because we require it if we wish to compute the normalized likelihood of data. This is often important in evaluating the model, monitoring training performance, and comparing models to each other.
For example, imagine we have two models: model M A defining a probability distribution p A ( x ; θ A ) = 1 Z A ˜ p A ( x ; θ A ) and model M B defining a probability distribution p B ( x ; θ B ) = 1 Z B ˜ p B ( x ; θ B ) . A common way to compare the models is to evaluate and compare the likelihood that both models assign to an i.i.d. test dataset. Suppose the test set consists of m examples { x (1) , . . . , x ( ) m } . If  i p A ( x ( ) i ; θ A ) >  i p B ( x ( ) i ; θ B ) or equivalently if  
<!-- formula-not-decoded -->
```

---

## [19/43] Ace the Data Science Interview

| Field | Value |
|-------|-------|
| **Pages** | 160-160 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.712 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `45de15cc-d76b-4def-8360-cbb96d2c94c2` |
| **YOUR GRADE** | ____ |

**Full Text (304 chars):**

```
RANK
Say that for each user, we wanted to rank posts by their length. We can use the window function RANK() to rank the posts by length for each user:
```
SELECT * U RANK () OVER ({ PARTITION BY user id ORDER BY LENGTH (body) DESC ) AS rank KF' ROM users u LEFT JOIN posts p ON u.user_ id = p.user_id
```
```

---

## [20/43] Vol 4: SQL Mastery

| Field | Value |
|-------|-------|
| **Pages** | 70-70 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.704 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `e4bd7d65-88f5-42d9-8880-aad79c0985c9` |
| **YOUR GRADE** | ____ |

**Full Text (691 chars):**

```
3.8 Tier 2 Summary
Problems Completed : 40 Critical Concepts Mastered :
- glyph[check] Window Functions - Ranking : ROW_NUMBER, RANK, DENSE_RANK
- glyph[check] Window Functions - LAG/LEAD : Previous/next values
- glyph[check] Window Functions - Running Totals : Cumulative SUM
- glyph[check] Window Functions - Moving Averages : ROWS BETWEEN
- glyph[check] PARTITION BY : Grouping for window functions
- glyph[check] Complex CTEs : Multiple WITH clauses
- glyph[check] Self-joins : Table joined to itself
- glyph[check] Date manipulation : DATE_DIFF, DATE_TRUNC, DATE_ADD
- glyph[check] CROSS JOIN : Cartesian products
- glyph[check] Anti-join patterns : NOT EXISTS, NOT IN, LEFT JOIN + NULL
```

---

## [21/43] Mykel J. Kochenderfer, Tim A. Wheeler - Algorithms for Optimization-The MIT Press (2019)

| Field | Value |
|-------|-------|
| **Pages** | 509-509 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.700 |
| **Found In** | vector |
| **Chunk ID** | `fc58c11e-a08e-4aab-a05d-d56bf12b69bc` |
| **YOUR GRADE** | ____ |

**Full Text (843 chars):**

```
Bibliography
74. D.R. Jones, C. D. Perttunen, and B. E. Stuckman, ''Lipschitzian Optimization Without the Lipschitz Constant,'' Journal of Optimization Theory and Application , vol. 79, no. 1, pp. 157-181, 1993 (cit. on p. 108).
75. D. Jones and M. Tamiz, Practical Goal Programming . Springer, 2010 (cit. on p. 219).
76. A. B. Kahn, ''Topological Sorting of Large Networks,'' Communications of the ACM , vol. 5, no. 11, pp. 558-562, 1962 (cit. on p. 390).
77. L. Kallmeyer, Parsing Beyond Context-Free Grammars . Springer, 2010 (cit. on p. 361).
78. L. V. Kantorovich, ''A New Method of Solving Some Classes of Extremal Problems,'' in Proceedings of the USSR Academy of Sciences , vol. 28, 1940 (cit. on p. 3).
79. A.F. Kaupe Jr, ''Algorithm 178: Direct Search,'' Communications of the ACM , vol. 6, no. 6, pp. 313-314, 1963 (cit. on p. 104).
```

---

## [22/43] (Use R!) Eric D. Kolaczyk, Gábor Csárdi - Statistical Analysis Of Network Data With R-Springer (2020)

| Field | Value |
|-------|-------|
| **Pages** | 112-112 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.693 |
| **Found In** | vector |
| **Chunk ID** | `6279264b-3c80-436f-8331-70c3f7b20e96` |
| **YOUR GRADE** | ____ |

**Full Text (676 chars):**

```
6.3.2 Model Fitting
```
#6.16 1 > cl.labs <- apply(Z,1,which.max)
```
Thus stochastic block models may be used as a model-based method of graph partitioning, complementing the other methods introduced in Sect.4.4. Note that in this case the evidence for class membership assignment appears to be uniformly strong (with maximum posterior probability exceeding 85%) across vertices.
```
#6.17 1 > nv <- vcount(fblog) 2 > summary(Z[cbind(1:nv,cl.labs)]) 3 Min. 1st Qu. Median Mean 3rd Qu. Max. 4 0.8586 0.9953 0.9953 0.9938 0.9953 0.9953
```
It is also of interest to examine the parameter estimates associated with this model. For example, estimates of the class proportions α q
```

---

## [23/43] N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting

| Field | Value |
|-------|-------|
| **Pages** | 2-2 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.691 |
| **Found In** | vector |
| **Chunk ID** | `181112a9-0a38-4214-b8f1-b9ac2c8b886e` |
| **YOUR GRADE** | ____ |

**Full Text (750 chars):**

```
2 PROBLEM STATEMENT
We consider the univariate point forecasting problem in discrete time. Given a lengthH forecast horizon a lengthT observed series history [ y 1 , . . . , yT ] ∈ R T , the task is to predict the vector of future values y ∈ R H = [ yT + 1 , yT + 2 , . . . , yT + H ] . For simplicity, we will later consider a lookback window of length t ≤ T ending with the last observed value yT to serve as model input, and denoted x ∈ R t = [ yT -t + 1 , . . . , yT ] . We denote ̂ y the forecast of y . The following metrics are commonly used to evaluate forecasting performance (Hyndman & Koehler, 2006; Makridakis & Hibon, 2000; Makridakis et al., 2018b; Athanasopoulos et al., 2011):
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
```

---

## [24/43] Ace the Data Science Interview

| Field | Value |
|-------|-------|
| **Pages** | 179-180 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.689 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `3f268598-0f8c-48e0-9c32-e347c3866c69` |
| **YOUR GRADE** | ____ |

**Full Text (980 chars):**

```
Solution #8.12
First, we calculate a subquery with total spend by product and category using SUM and GROUP BY. Note that we must filter by a 2020 transaction date. Then, using this subquery, we utilize a window function to calculate the rankings (by spend) for each product category using the RANK window function over the existing sums in the previous subquery. For the window function, we PARTITION by category and ORDER by product spend. Finally, we use this result and then filter for a rank less than or equal to 3 as shown below.
```
WITH product category spend AS ( SELECT product id, category id, SUM(spend) AS total product_spend FROM product_spend WERERE transaction date BETWEEN '2020-01-01' AND '2020-12-31' GROUP BY product id, category id ), top spend AS ( SELECT p-.*, RANK() OVER ( PARTITION BY category id ORDER BY total product spend DESC ) AS rnk F ROS
```
```
product category spend p ) SELECT * FROM top spend WHERE rnk <= 3 ORDER BY category id, rnk DESC
```
```

---

## [25/43] Volume 1: The Google Data Scientist Interview Guide

| Field | Value |
|-------|-------|
| **Pages** | 127-127 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.688 |
| **Found In** | fts, citations |
| **Chunk ID** | `4cad25c9-8a28-484a-b677-368fafb022fa` |
| **YOUR GRADE** | ____ |

**Full Text (1064 chars):**

```
80.3. SQL Problem 2: Top 3 Users by Monthly Spend (L4 Level - Window Functions)
1. CTE vs Subquery :  "I  used  CTEs  for  readability.  For  complex  queries  with  multiple  steps, CTEs make logic easier to follow and debug."
2. Window Function Choice : "ROW_NUMBER ensures exactly 3 users per month. If two users tied for 3rd place, RANK would return 4 rows for that month, which violates the 'top 3' requirement."
3. DATE_TRUNC : "Using DATE_TRUNC('month', order_date) standardizes all dates in a month to  the  first  day,  enabling  clean  grouping.  Alternative:  EXTRACT(YEAR,  MONTH)  but DATE_TRUNC is cleaner."
Common Mistakes:
❌ Using  RANK  instead  of  ROW_NUMBER  (ties  cause  >3  results) ❌ Filtering  before  window function  (WHERE  spend  >  X  before  ranking  breaks  the  partition  logic) ❌ Forgetting  to partition (OVER ORDER BY without PARTITION BY ranks globally, not per month)
Performance :  O(n  log  n)  for  sorting  within  each  partition.  With  proper  indexing  on (order_date, user_id), the GROUP BY aggregation is efficient.
```

---

## [26/43] Adnan Aziz, Tsung-Hsien Lee, Amit Prakash - Elements of Programming Interviews in Python: The Insiders’ Guide-CreateSpace Independent Publishing Platform

| Field | Value |
|-------|-------|
| **Pages** | 323-323 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.687 |
| **Found In** | vector |
| **Chunk ID** | `3f6a6557-bf8c-4028-9dca-53c93742bcbd` |
| **YOUR GRADE** | ____ |

**Full Text (1350 chars):**

```
20.8 lNrprsrlsNr  PecsRANr
- o Disk-based  sorting-we keep the column  vector X in memory and load rows one at a time. Processing  Row f simply  requires adding A;,iX1to X7 for each  i  such that Ai,1is not zero. The advantage  of this approach  is that if the column vector fits in RAM,  the entire computation can be performed on a single machine.  This approach is slow because it uses a single machine and relies on the disk.
- o Partitioned  graph-we use n servers and partition the vertices  (web pages) into n sets. This partition can be computed by partitioning the set of hash codes in such a way that it is easy to determine which vertex maps to which machine. Given this partitioning,  each machine loads its vertices  and their outgoing edges into RAM. Each machine  also loads the portion of the PageRank vector corresponding to the vertices it is responsible  for. Then each machine does a local matrix multiplication. Some of the edges on each machine  may correspond to vertices  that are owned  by other  machines. Hence the result  vector  contains  nonzero  entries for vertices  that are not owned by the local machine.  At the end of the local multiplication it needs to send updates to other hosts so that these values  can be correctly added up. The advantage of this approach  is that it can process  arbitrarily large graphs.
```

---

## [27/43] The Algorithm Design Manual

| Field | Value |
|-------|-------|
| **Pages** | 526-526 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.685 |
| **Found In** | vector |
| **Chunk ID** | `cbd5ff6b-b3f3-49f0-a769-3da25e0945a5` |
| **YOUR GRADE** | ____ |

**Full Text (1215 chars):**

```
17.3 Median and Selection
Notes : The linear expected-time algorithm for median and selection is due to Hoare [Hoa61]. Floyd and Rivest [FR75] provide an algorithm that uses fewer comparisons on average. Good expositions on linear-time selection include [BvG99, CLRS09, Raw92], with [Raw92] being particularly enlightening.
Streaming algorithms have extensive applications to large data sets, and are well surveyed by Muthukrishnan [Mut05] and Cormode [CH09].
A sport of considerable theoretical interest is determining exactly how many comparisons are sufficient to find the median of n items. The linear-time algorithm of Blum et al. [BFP + 72] proves that c · n comparisons suffice, but we want to know what c is. Dor and Zwick [DZ99] proved that 2 . 95 n comparisons suffice to find the median. These algorithms attempt to minimize the number of element comparisons but not the total number of operations, and hence do not lead to faster algorithms in practice. They also hold the current best lower bound of (2 + /epsilon1 ) comparisons for median finding [DZ01].
Tight combinatorial bounds for selection problems are presented in Aigner [Aig88]. An optimal algorithm for computing the mode is given by [DM80].
```

---

## [28/43] Direct Preference Optimization: Your Language Model is Secretly a Reward Model

| Field | Value |
|-------|-------|
| **Pages** | 6-6 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.682 |
| **Found In** | vector |
| **Chunk ID** | `30c03b8d-2e6b-4293-bc24-52b1ab362f8c` |
| **YOUR GRADE** | ____ |

**Full Text (640 chars):**

```
5.1 Your Language Model Is Secretly a Reward Model
<!-- formula-not-decoded -->
i.e., π ( y | x ) is a valid distribution (probabilities are positive and sum to 1). However, following Eq. 4, we can see that Eq. 9 is the partition function of the optimal policy induced by the reward function r ( x, y ) . The key insight of the DPO algorithm is that we can impose certain constraints on the under-constrained Plackett-Luce (and Bradley-Terry in particular) family of preference models, such that we preserve the class of representable reward models, but explicitly make the optimal policy in Eq. 4 analytically tractable for all prompts x .
```

---

## [29/43] arXiv:2307.03172

| Field | Value |
|-------|-------|
| **Pages** | 11-11 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.679 |
| **Found In** | vector |
| **Chunk ID** | `084a194e-98e0-4221-8844-2d407fda65b0` |
| **YOUR GRADE** | ____ |

**Full Text (864 chars):**

```
References
- Avi Arampatzis, Jaap Kamps, and Stephen Robertson. 2009. Where to stop reading a ranked list? threshold optimization using truncated score distributions. In Proc. of SIGIR .
- Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. ArXiv:2004.05150.
- Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instructionfinetuned language models. ArXiv:2210.11416.
```

---

## [30/43] Volume 1: The Google Data Scientist Interview Guide

| Field | Value |
|-------|-------|
| **Pages** | 126-127 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.679 |
| **Found In** | fts |
| **Chunk ID** | `0e3be6fd-95b1-4ffa-877c-01318b9e367c` |
| **YOUR GRADE** | ____ |

**Full Text (1130 chars):**

```
80.3. SQL Problem 2: Top 3 Users by Monthly Spend (L4 Level - Window Functions)
Problem Statement:
Using the same orders table, find the top 3 users by total spend for each month in 2025.
Solution Approach (critical for L4+):
1. Extract year-month from order_date
2. Aggregate spend by user and month
3. Use window function (RANK or ROW_NUMBER) partitioned by month
4. Filter to rank ≤ 3
Full SQL Solution:
```
1 WITH monthly_spend AS ( 2     SELECT 3         user_id, 4         DATE_TRUNC('month', order_date) AS month, 5         SUM(amount) AS total_spend 6     FROM orders 7     WHERE order_date >= '2025-01-01' AND order_date < '2026-01-01' 8     GROUP BY user_id, DATE_TRUNC('month', order_date) 9 ), 10 ranked_users AS ( 11     SELECT 12         user_id, 13         month, 14         total_spend, 15         ROW_NUMBER() OVER ( 16             PARTITION BY month 17             ORDER BY total_spend DESC 18         ) AS spend_rank 19     FROM monthly_spend 20 ) 21 SELECT 22     user_id, 23     month, 24     total_spend, 25     spend_rank 26 FROM ranked_users 27 WHERE spend_rank <= 3
```
[3]
L4 Interview Discussion Points:
```

---

## [31/43] Graph-Powered Machine Learning

| Field | Value |
|-------|-------|
| **Pages** | 490-490 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.677 |
| **Found In** | vector |
| **Chunk ID** | `d9a5b0c8-32c5-4d4e-9939-31a49d7339fc` |
| **YOUR GRADE** | ____ |

**Full Text (1191 chars):**

```
index

memory-based approach to collaborative filtering 167 meta information 123 methodological approach to big data challenges 40 minutes-to-milliseconds performance 66 mixed hybridization strategy 258 model-based approach to collaborative filtering 168 model-based learning 433 - 434 modeling contextual 249 - 251, 1 = node clustering 352 nodes 53 , 84. memory-based approach to collaborative filtering 167 meta information 123 methodological approach to big data challenges 40 minutes-to-milliseconds performance 66 mixed hybridization strategy 258 model-based approach to collaborative filtering 168 model-based learning 433 - 434 modeling contextual 249 - 251, 2 = . , 1 = non-native graph databases 61 - 66. , 2 = . , 1 = O. , 2 = . , 1 = online learning 433 optimization, session-based. , 2 = . , 1 = k-NN -. , 2 = . , 1 = first 221 223. , 2 = . , 1 = second 223 - 224. , 2 = . hybrid recommendation engines 256 -, 1 = out-degree metrics 17 , 330. hybrid recommendation engines 256 -, 2 = . 257 phase 9, 1 = . 257 phase 9, 2 = . , 1 = P. , 2 = . user 136 - 142 of networks 17 - 22, 1 = . user 136 - 142 of networks 17 - 22, 2 = . , 1 = PageRank. , 2 = . models monitoring subject, 1 =
```

---

## [32/43] Graph-Powered_Machine_Learning (5)

| Field | Value |
|-------|-------|
| **Pages** | 490-490 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.677 |
| **Found In** | vector |
| **Chunk ID** | `b54025e1-337d-465f-8f93-92a46b47005a` |
| **YOUR GRADE** | ____ |

**Full Text (1191 chars):**

```
index

memory-based approach to collaborative filtering 167 meta information 123 methodological approach to big data challenges 40 minutes-to-milliseconds performance 66 mixed hybridization strategy 258 model-based approach to collaborative filtering 168 model-based learning 433 - 434 modeling contextual 249 - 251, 1 = node clustering 352 nodes 53 , 84. memory-based approach to collaborative filtering 167 meta information 123 methodological approach to big data challenges 40 minutes-to-milliseconds performance 66 mixed hybridization strategy 258 model-based approach to collaborative filtering 168 model-based learning 433 - 434 modeling contextual 249 - 251, 2 = . , 1 = non-native graph databases 61 - 66. , 2 = . , 1 = O. , 2 = . , 1 = online learning 433 optimization, session-based. , 2 = . , 1 = k-NN -. , 2 = . , 1 = first 221 223. , 2 = . , 1 = second 223 - 224. , 2 = . hybrid recommendation engines 256 -, 1 = out-degree metrics 17 , 330. hybrid recommendation engines 256 -, 2 = . 257 phase 9, 1 = . 257 phase 9, 2 = . , 1 = P. , 2 = . user 136 - 142 of networks 17 - 22, 1 = . user 136 - 142 of networks 17 - 22, 2 = . , 1 = PageRank. , 2 = . models monitoring subject, 1 =
```

---

## [33/43] Graph-Powered Machine Learning

| Field | Value |
|-------|-------|
| **Pages** | 490-490 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.677 |
| **Found In** | vector |
| **Chunk ID** | `ec419f9f-dc48-4e84-8497-66549dfb5b80` |
| **YOUR GRADE** | ____ |

**Full Text (1191 chars):**

```
index

memory-based approach to collaborative filtering 167 meta information 123 methodological approach to big data challenges 40 minutes-to-milliseconds performance 66 mixed hybridization strategy 258 model-based approach to collaborative filtering 168 model-based learning 433 - 434 modeling contextual 249 - 251, 1 = node clustering 352 nodes 53 , 84. memory-based approach to collaborative filtering 167 meta information 123 methodological approach to big data challenges 40 minutes-to-milliseconds performance 66 mixed hybridization strategy 258 model-based approach to collaborative filtering 168 model-based learning 433 - 434 modeling contextual 249 - 251, 2 = . , 1 = non-native graph databases 61 - 66. , 2 = . , 1 = O. , 2 = . , 1 = online learning 433 optimization, session-based. , 2 = . , 1 = k-NN -. , 2 = . , 1 = first 221 223. , 2 = . , 1 = second 223 - 224. , 2 = . hybrid recommendation engines 256 -, 1 = out-degree metrics 17 , 330. hybrid recommendation engines 256 -, 2 = . 257 phase 9, 1 = . 257 phase 9, 2 = . , 1 = P. , 2 = . user 136 - 142 of networks 17 - 22, 1 = . user 136 - 142 of networks 17 - 22, 2 = . , 1 = PageRank. , 2 = . models monitoring subject, 1 =
```

---

## [34/43] Genetic Algorithms in Search, Optimization, and Machine Learning-Addison

| Field | Value |
|-------|-------|
| **Pages** | 127-128 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.676 |
| **Found In** | vector |
| **Chunk ID** | `228102a9-7ac8-4a29-9e2e-6a49d1ba9efe` |
| **YOUR GRADE** | ____ |

**Full Text (766 chars):**

```
DEJONGANDFUNCTIONOPTIMIZATION
FIGURE4.10 Inverted,two-dimensionalversionsofDeJong's(1975)testfunctionsF3andF4.Reprintedbypermission.
FIGURE4.11 Invertedversion ofDeJong's(1975) testfunctionF5.Reprinted by permission.
cluding thecurrenttrial.DeJongactuallypresentedamoregeneralversionof this criterion,which permitted nonuniform weighting oftrials;however,he adoptedauniformweightingthroughouthisstudy.
Healsodefinedtheperformancemeasurex*(s),theoff-lineperformanceof strategysonenvironmenteasfollows:
<!-- formula-not-decoded -->
wheref(t)=best{f（1）,f(2)..,f(t)}.Inwords,the off-lineperformance is arunningaverageofthebestperformancevaluestoaparticulartime.Again,a nonuniformlyweightedversionofthiscriterionwasalsoproposed,butuniform trialweightingwasusedthroughout.
```

---

## [35/43] Deep Learning for Time Series Forecasting: Tutorial and Literature Survey

| Field | Value |
|-------|-------|
| **Pages** | 28-28 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.675 |
| **Found In** | vector |
| **Chunk ID** | `5611d910-24a1-4bce-b9f1-c5e28ffe1885` |
| **YOUR GRADE** | ____ |

**Full Text (1143 chars):**

```
References
- [91] Tim Januschowski, Jan Gasthaus, Yuyang Wang, David Salinas, Valentin Flunkert, Michael Bohlke-Schneider, and Laurent Callot. Criteria for classifying forecasting methods. International Journal of Forecasting , 2019.
- [92] Tim Januschowski, Yuyang Wang, Hilaf Hasson, Timo Erkkila, Kari Torkkila, and Jan Gasthaus. Forecasting with trees. International Journal of Forecasting , 2021.
- [93] Yunho Jeon and Sihyeon Seong. Robust recurrent network model for intermittent time-series forecasting. International Journal of Forecasting , 2021.
- [94] Michael I. Jordan. Serial order: A parallel, distributed processing approach. Technical report, Institute for Cognitive Science, University of California, San Diego, 1986.
- [95] Michael I. Jordan. Serial order: A parallel, distributed processing approach. In Advances in Connectionist Theory: Speech . Erlbaum, 1989.
- [96] Kelvin Kan, Franc ¸ois-Xavier Aubet, Tim Januschowski, Youngsuk Park, Konstantinos Benidis, Lars Ruthotto, and Jan Gasthaus. Multivariate quantile function forecaster. In The 25th International Conference on Artificial Intelligence and Statistics , 2022.
```

---

## [36/43] Koller Friedman Pgm 2009

| Field | Value |
|-------|-------|
| **Pages** | 580-580 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.675 |
| **Found In** | vector |
| **Chunk ID** | `546ee9af-0b6a-4a75-aebc-b2e4b4274e98` |
| **YOUR GRADE** | ____ |

**Full Text (1361 chars):**

```
12.7 Relevant Literature
Algorithms that improve convergence are particularly relevant for the high-dimensional, multimodal distributions that often arise in the setting of graphical models. Some methods for addressing this issue use larger, nonlocal steps in the search space, which are helpful in breaking out of local optima; for example, for pairwise MRFs where all variables have a uniform set of values, Swendsen and Wang (1987) and Barbu and Zhu (2005) propose moves that simultaneously flip an entire subgraph from one value to another. Higdon (1998) discusses the general idea of introducing auxiliary variables as a mechanism for taking larger steps in the space. The temperature-based methods draw on the idea of simulated annealing (Kirkpatrick et al. 1983). These methods include simulated tempering (Marinari and Parisi 1992; Geyer and Thompson 1995) in which the state of the model is augmented with a temperature variable for purposes of sampling; parallel tempering (Swendsen and Wang 1986; Geyer 1991) runs multiple chains at different temperatures at the same time and allows chains to exchange datapoints; tempered transitions (Neal 1996) proposes a new sample by moving up and down the temperature schedules; and annealed importance sampling Neal (2001) uses a similar approach in combination with an importance sampling reweighting scheme.
```

---

## [37/43] Koller Friedman Pgm 2009

| Field | Value |
|-------|-------|
| **Pages** | 632-633 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.670 |
| **Found In** | vector |
| **Chunk ID** | `ee06f142-6e98-4e44-92c6-c8f1cc318ae7` |
| **YOUR GRADE** | ____ |

**Full Text (1391 chars):**

```
13.7 Local Search Algorithms glyph[star]
The application of search techniques to the MAP problem is a fairly straightforward process: The search space is defined by the possible assignments ξ to X , and log ˜ P ( ξ ) is the score; we omit details. Although generally less powerful than the methods we described earlier, these methods do have some advantages. For example, the beam search method of appendix A.4.2 provides a useful alternative in cases where the complete model is too large to fit into memory; see exercise 15.10. We also note that branch-and-bound does provide a simple method for finding the K most likely assignment; see exercise 13.18. This algorithm requires at least as much computation time as the clique tree-based algorithm, but significantly less space.
These methods have much greater applicability in the context of marginal MAP problem, where most other methods are not (currently) applicable. Here, we search over the space of assignments y to the max-variables Y . Here, we conduct the search so that we can fix some or all of the max-variables to have a concrete assignment. As we show, this allows us to remove the constraint on the variable elimination ordering, allowing an unrestricted ordering to be used.
search operator tabu search
dynamic programming
Here, we search over the space of assignments y for those that maximize
<!-- formula-not-decoded -->
```

---

## [38/43] Barber D., Cemgil A.T., Chiappa S. (eds.) - Bayesian Time Series Models-Cambridge University Press

| Field | Value |
|-------|-------|
| **Pages** | 227-227 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.668 |
| **Found In** | vector |
| **Chunk ID** | `2d61c65e-7ae1-44e0-9ea6-5aa21a7e1bc5` |
| **YOUR GRADE** | ____ |

**Full Text (1201 chars):**

```
10.3.2 Segment neighbourhood search
References [7] and [6] consider an alternative search algorithm for changepoint detection, namely the segment neighbourhood approach (also referred to as global segmentation). The basic principle of this approach is to define some measure of data fit, R ( ), for a segment. For inference via penalised likelihood we would set R ( ys : t ) to be minus the maximum loglikelihood value for data ys : t given it comes from a single segment. That is
<!-- formula-not-decoded -->
We then set a maximum number of segments, M , corresponding to at most M 1 changepoints.
The segment neighbourhood search then uses a dynamic programming algorithm to find the best partition of the data into m + 1 segments for m = 0 ; : : : ; M 1. The best partition is found by minimising the cost function P m i = 0 R ( y i : i + 1 ) for a partition with changepoints at positions 1 ; 2 ; : : : ; m . Thus for R ( ) defined in Eq. (10.2), this would give the partition of the data with m changepoints that maximises the log-likelihood. The algorithm will output the best partition for m = 0 ; : : : ; M 1, and the corresponding minimum value of the cost function, which we denote c m 1 n .
```

---

## [39/43] Undergraduate Topics in Computer Science Kent D Lee Steve Hubbard - Data Structures and Algorithms with Python-Springer 2015

| Field | Value |
|-------|-------|
| **Pages** | 366-366 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.668 |
| **Found In** | vector |
| **Chunk ID** | `4020dc72-8a31-4dd8-9e6f-de621d3db80a` |
| **YOUR GRADE** | ____ |

**Full Text (840 chars):**

```
Bibliography
1. Adelson-Veskii G, Landis EM (1962) An algorithm for the organization of information. Proc USSR Acad Sci 146:263-266
2. Carlis J, Maguire J (2000) Mastering data modeling: a user-driven approach. Addison-Wesley http://www.amazon.com/Mastering-Data-Modeling-User-Driven-Approach/dp/020170045X/ ref=sr_1_1?s=books&ie=UTF8&qid=1404178333&sr=1-1
3. Coppin B (2004) Artificial intelligence illuminated. Jones and Bartlett, USA
4. Dijkstra EW (1959) A note on two problems in connexion with graphs. Nume Math 1:269-271
5. Kruskal JB (1956) On the shortest spanning tree of a graph and the traveling salesman problem. Proc Am Math Soc 7:48-50
6. Lutz M (2013) Learning Python. O'Reilly Media http://www.amazon.com/LearningPython-Edition-Mark-Lutz/dp/1449355730/ref=sr_1_1?ie=UTF8&qid=1398871248&sr=8-1 &keywords=learning+python+lutz
```

---

## [40/43] Mykel J. Kochenderfer, Tim A. Wheeler - Algorithms for Optimization-The MIT Press (2019)

| Field | Value |
|-------|-------|
| **Pages** | 508-508 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.667 |
| **Found In** | vector |
| **Chunk ID** | `7fb915e2-d87f-4640-8361-117a41bd895e` |
| **YOUR GRADE** | ____ |

**Full Text (744 chars):**

```
Bibliography
70. G. Hinton and S. Roweis, ''Stochastic Neighbor Embedding,'' in Advances in Neural Information Processing Systems (NIPS) , 2003 (cit. on p. 125).
71. R. Hooke and T. A. Jeeves, ''Direct Search Solution of Numerical and Statistical Problems,'' Journal of the ACM (JACM) , vol. 8, no. 2, pp. 212-229, 1961 (cit. on p. 102).
72. H. Ishibuchi and T. Murata, ''A Multi-Objective Genetic Local Search Algorithm and Its Application to Flowshop Scheduling,'' IEEE Transactions on Systems, Man, and Cybernetics , vol. 28, no. 3, pp. 392-403, 1998 (cit. on p. 225).
73. V. S. Iyengar, J. Lee, and M. Campbell, ''Q-EVAL: Evaluating Multiple Attribute Items Using Queries,'' in ACM Conference on Electronic Commerce , 2001 (cit. on p. 229).
```

---

## [41/43] (Use R!) Eric D. Kolaczyk, Gábor Csárdi - Statistical Analysis Of Network Data With R-Springer (2020)

| Field | Value |
|-------|-------|
| **Pages** | 114-115 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.666 |
| **Found In** | vector |
| **Chunk ID** | `c009c451-de5f-4efb-bedb-64b6d6e58208` |
| **YOUR GRADE** | ____ |

**Full Text (900 chars):**

```
6.3.3 Goodness-of-Fit
clip(0,nv+1,nv+1,0) 13 > abline(v=c(0.5,cl.lim,nv+0.5), 14 + h=c(0.5,cl.lim,nv+0.5),col="red")
```
The clustering remarked upon earlier, into six larger classes and four smaller classes, is now evident to the eye. Furthermore, while it appears that the vertices in some of these classes are primarily connected with other vertices within their respective classes, among those other classes in which vertices show a propensity towards interclass connections there seems to be, in some cases, a tendency towards connecting selectively with vertices of only certain other classes.
Finally, it is of interest to consider to what extent the graph partitioning induced by the vertex class assignments (i.e., into ten classes) matches the grouping of these blogs according to their political party status (i.e., according to nine parties). This comparison is summarized in Fig.6.3.
```
```

---

## [42/43] Volume 2: The Google Data Scientist Interview Workbook

| Field | Value |
|-------|-------|
| **Pages** | 100-100 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.659 |
| **Found In** | fts |
| **Chunk ID** | `53e20ad7-42e3-450f-be97-476bad521c68` |
| **YOUR GRADE** | ____ |

**Full Text (172 chars):**

```
SQL Cards:
- Difference between ROW_NUMBER, RANK, DENSE_RANK
- Window function syntax template
- PARTITION BY vs GROUP BY
- LAG vs LEAD
- Frame clause syntax (ROWS BETWEEN)
```

---

## [43/43] Vol 4: SQL Mastery

| Field | Value |
|-------|-------|
| **Pages** | 271-272 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.643 |
| **Found In** | fts |
| **Chunk ID** | `3cd77d68-6a60-41dc-8590-f062fa1f2353` |
| **YOUR GRADE** | ____ |

**Full Text (492 chars):**

```
QUALIFY (Filter Window Results)
```
-- BigQuery-specific: Filter AFTER window function evaluation SELECT name,
```
```
salary, department, RANK() OVER (PARTITION BY department ORDER BY salary DESC ) AS rank FROM employees QUALIFY rank <= 3; -- Only top 3 per department -- Equivalent without QUALIFY (standard SQL): WITH ranked AS ( SELECT name, salary, department, RANK() OVER (PARTITION BY department ORDER BY salary DESC ) AS rank FROM employees ) SELECT * FROM ranked WHERE rank <= 3;
```
```

---
