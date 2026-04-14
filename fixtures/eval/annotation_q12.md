# Query 12: hypothesis testing p-value

**Domain:** statistics
**Query ID:** q_stat_004
**Candidates:** 52
**Grading:** 0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant

---

## [1/52] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 139-139 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.791 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `63f98fb8-4565-409c-ae44-431d3608a95a` |
| **YOUR GRADE** | ____ |

**Full Text (1369 chars):**

```
15.2 Statistical Hypothesis Testing
A statistical hypothesis test makes an assumption about the outcome, called the null hypothesis. For example, the null hypothesis for the Pearson's correlation test is that there is no relationship between two variables. The null hypothesis for the Student's t-test is that there is no difference between the means of two populations. The test is often interpreted using a p-value, which is the probability of observing the result given that the null hypothesis is true, not the reverse, as is often the case with misinterpretations.
- p-value (p) : Probability of obtaining a result equal to or more extreme than was observed in the data.
In interpreting the p-value of a significance test, you must specify a significance level, often referred to as the Greek lower case letter alpha ( α ). A common value for the significance level is 5% written as 0.05. The p-value is interested in the context of the chosen significance level. A result of a significance test is claimed to be statistically significant if the p-value is less than the significance level. This means that the null hypothesis (that there is no result) is rejected.
- p-value ≤ alpha : significant result, reject null hypothesis, distributions differ (H1).
- p-value > alpha : not significant result, fail to reject null hypothesis, distributions same (H0).
Where:
```

---

## [2/52] Regression Modeling with Actuarial and Financial Applications

| Field | Value |
|-------|-------|
| **Pages** | 570-570 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.790 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `723d9157-9c3f-44fd-9787-705ef824c1a9` |
| **YOUR GRADE** | ____ |

**Full Text (433 chars):**

```
A1.3 Testing Hypotheses
p -value. Another useful concept in hypothesis testing is the p-value , which is shorthand for probability value . For a data set, a p -value is defined as the smallest significance level for which the null hypothesis would be rejected. The p -value is a useful summary statistic for the data analyst to report because it allows the reader to understand the strength of the deviation from the null hypothesis.
```

---

## [3/52] Ace the Data Science Interview

| Field | Value |
|-------|-------|
| **Pages** | 75-75 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.786 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `09191fff-047e-4563-b6af-e1f5df24552c` |
| **YOUR GRADE** | ____ |

**Full Text (1244 chars):**

```
Solution #6.6
The process of testing whether data supports particular hypotheses is called hypothesis testing and involves measuring parameters of a population's probability distribution. This process typically employs at least two groups - one a control that receives no treatment, and the other group(s), which do receive the treatment(s) of interest. Examples could be the height of two groups of people, the conversion rates for particular user flows in a product, etc. Testing also involves two hypotheses the null hypothesis, which assumes no significant difference between the groups, and the alternative hypothesis, which assumes a significant difference in the measured parameter(s) as a consequence of the treatment.
A p-value is the probability of observing the given test results under the null hypothesis assumptions. The lower this probability, the higher the chance that the null hypothesis should be rejected. If the p-value is lower than the predetermined significance level a, generally set at 0.05, then it indicates that the null hypothesis should be rejected in favor of the alternative hypothesis. Otherwise, the null hypothesis cannot be rejected, and it cannot be concluded that the treatment has any significant effect.
```

---

## [4/52] David M Diez, Christopher D Barr, Mine Çetinkaya-Rundel - OpenIntro Statistics-OpenIntro, Inc. (2015)

| Field | Value |
|-------|-------|
| **Pages** | 286-286 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.782 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `b3571817-4fa2-4574-a6a9-412fbf1aa4a8` |
| **YOUR GRADE** | ____ |

**Full Text (863 chars):**

```
6.2.4 More on 2-proportion hypothesis tests (special topic)
<!-- formula-not-decoded -->
In this hypothesis test, because the null is that p 1 -p 2 = 0 . 03, the sample proportions were used for the standard error calculation rather than a pooled proportion.
Next, we compute the test statistic and use it to find the p-value, which is depicted in Figure 6.5.
<!-- formula-not-decoded -->
Using the normal model for this test statistic, we identify the right tail area as 0.006. Since this is a one-sided test, this single tail area is also the p-value, and we reject the null hypothesis because 0.006 is less than 0.05. That is, we have statistically significant evidence that the higher-quality blades actually do pass inspection more than 3% as often as the currently used blades. Based on these results, management will approve the switch to the new supplier.
```

---

## [5/52] (Springer Texts in Statistics'',) Larry Wasserman - All of Statistics  A Concise Course in Statistical Inference (Springer Texts in Statistics)-Springer (2003)

| Field | Value |
|-------|-------|
| **Pages** | 8-8 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.782 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `ddc06ec1-3b17-4895-b688-727759d18187` |
| **YOUR GRADE** | ____ |

**Full Text (714 chars):**

```
I Probability

 Contents = . .. ,  = .. , xv = .. , xv = 142. , xv = . .. , xv = . , xv = . , xv = . , xv = . , xv = . ,  = . ,  = . ,  = 9.14. ,  = Exercises . . . . .. ,  = . .. ,  = . .. ,  = . ,  = . . . .. ,  = . .. ,  = . . .. ,  = . .. ,  = .. ,  = . ,  = .. ,  = .. ,  = . ,  = .. , Contents = . ,  = .. , xv = .. , xv = . , xv = .. , xv = 146. , xv = 146. , xv = 146. , xv = 146. , xv = 146. ,  = . ,  = . 10,  = Hypothesis Testing and p-values 149. 10,  = . 10,  = . 10,  = . 10,  = . 10,  = . 10,  = . 10,  = . 10,  = . 10,  = . 10,  = . 10,  = . 10,  = . 10,  = . 10,  = . 10, Contents = . 10,  = . 10, xv = . 10, xv = . 10, xv = . 10, xv = . 10, xv = . 10, xv = . 10, xv = . 10, xv = . 10,  = . 10,  =
```

---

## [6/52] openintro-statistics

| Field | Value |
|-------|-------|
| **Pages** | 194-194 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.779 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `a94af778-2d13-4248-9e1c-d90c5be2feb9` |
| **YOUR GRADE** | ____ |

**Full Text (298 chars):**

```
5.3.4 Formal testing using p-values
The p-value is a way of quantifying the strength of the evidence against the null hypothesis and in favor of the alternative hypothesis. Statistical hypothesis testing typically uses the p-value method rather than making a decision based on confidence intervals.
```

---

## [7/52] David M Diez, Christopher D Barr, Mine Çetinkaya-Rundel - OpenIntro Statistics-OpenIntro, Inc. (2015)

| Field | Value |
|-------|-------|
| **Pages** | 186-186 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.779 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `9f5eee8d-06c9-41ef-8e0b-9b1621791fc0` |
| **YOUR GRADE** | ____ |

**Full Text (1357 chars):**

```
p-value
The p-value is the probability of observing data at least as favorable to the alternative hypothesis as our current data set, if the null hypothesis is true. We typically use a summary statistic of the data, in this chapter the sample mean, to help compute the p-value and evaluate the hypotheses.
- ⊙ Guided Practice 4.26 A poll by the National Sleep Foundation found that college students average about 7 hours of sleep per night. Researchers at a rural school are interested in showing that students at their school sleep longer than seven hours on average, and they would like to demonstrate this using a sample of students. What would be an appropriate skeptical position for this research? 22
We can set up the null hypothesis for this test as a skeptical perspective: the students at this school average 7 hours of sleep per night. The alternative hypothesis takes a new form reflecting the interests of the research: the students average more than 7 hours of sleep. We can write these hypotheses as
<!-- formula-not-decoded -->
Using µ > 7 as the alternative is an example of a one-sided hypothesis test. In this investigation, there is no apparent interest in learning whether the mean is less than 7 hours. 23 Earlier we encountered a two-sided hypothesis where we looked for any clear difference, greater than or less than the null value.
```

---

## [8/52] (Springer Texts in Statistics'',) Larry Wasserman - All of Statistics  A Concise Course in Statistical Inference (Springer Texts in Statistics)-Springer (2003)

| Field | Value |
|-------|-------|
| **Pages** | 162-163 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.776 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `fcc75e2a-c5f5-4731-8d6b-7ae5f0daa4c3` |
| **YOUR GRADE** | ____ |

**Full Text (1267 chars):**

```
Hypothesis Testing and p-values
Suppose we want to know if exposure to asbestos is associated with lung disease. We take some rats and randomly divide them into two groups. We expose one group to asbestos and leave the second group unexposed. Then we compare the disease rate in the two groups. Consider the following two hypotheses:
The Null Hypothesis : The disease rate is the same in the two groups.
The Alternative Hypothesis : The disease rate is not the same in the two groups.
If the exposed group has a much higher rate of disease than the unexposed group then we will reject the null hypothesis and conclude that the evidence favors the alternative hypothesis. This is an example of hypothesis testing.
More formally, suppose that we partition the parameter space Θ into two disjoint sets Θ 0 and Θ 1 and that we wish to test
<!-- formula-not-decoded -->
We call H 0 the null hypothesis and H 1 the alternative hypothesis .
Let X be a random variable and let X be the range of X . We test a hypothesis by finding an appropriate subset of outcomes R ⊂ X called the rejection
TABLE 10.1. Summary of outcomes of hypothesis testing.

, 1 = Retain Null. , 2 = Reject Null. H 0 true, 1 = √. H 0 true, 2 = type I error. H 1 true, 1 = type II error. H 1 true, 2 = √
```

---

## [9/52] book2

| Field | Value |
|-------|-------|
| **Pages** | 115-115 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.773 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `b64f2161-733a-4ed1-a197-6a26ad060d61` |
| **YOUR GRADE** | ____ |

**Full Text (1323 chars):**

```
3.3.5.2 p-values
The frequentist approach to hypothesis testing, known as null hypothesis significance testing or NHST , is to define a decision procedure for deciding whether to accept or reject the null hypothesis H 0 based on whether some observed test statistic t ( D ) is likely or not under the sampling distribution of the null model. We describe this procedure in more detail in Section 3.10.1.
Rather than accepting or rejecting the null hypothesis, we can compute a quantity related to how likely the null hypothesis is to be true. In particular, we can compute a quantity called a p-value , which is defined as
<!-- formula-not-decoded -->
A p-value is often interpreted as the likelihood of the data under the null hypothesis, so small values are interpreted to mean that H 0 is unlikely, and therefore that H 1 is likely. The reasoning is roughly as follows:
where ˜ D ∼ H 0 is hypothetical future data. That is, the p-value is just the tail probability of observing the value t ( D ) under the sampling distribution. (Note that the p-value does not explicitly depend on a model of the data, but most common test statistics implicitly define a model, as we discuss in Section 3.10.3.)
If H 0 is true, then this test statistic would probably not occur. This statistic did occur. Therefore H 0 is probably false.
```

---

## [10/52] ROS

| Field | Value |
|-------|-------|
| **Pages** | 72-72 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.772 |
| **Found In** | vector |
| **Chunk ID** | `bf5da163-bb76-4029-8626-aa6d36dcf265` |
| **YOUR GRADE** | ____ |

**Full Text (462 chars):**

```
Hypothesis testing: general formulation
In the simplest form of hypothesis testing, the null hypothesis H 0 represents a particular probability model, p ( y ) , with potential replication data y rep . To perform a hypothesis test, we must define a test statistic T , which is a function of the data. For any given data y , the p -value is then Pr ( T ( y rep ) T ( y )) : the probability of observing, under the model, something as or more extreme than the data.
```

---

## [11/52] Statistics Slam Dunk

| Field | Value |
|-------|-------|
| **Pages** | 158-158 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.770 |
| **Found In** | fts, vector, hybrid, citations |
| **Chunk ID** | `9688c92e-164a-4563-847d-ba59888300cf` |
| **YOUR GRADE** | ____ |

**Full Text (1373 chars):**

```
Hypothesis testing and p-values
Let's take a brief pause to raise a few additional points around hypothesis testing and p-values. Hypothesis testing, or statistical inference, is all about testing an assumption and drawing a conclusion from one or more data series. Hypothesis testing essentially evaluates how unusual or not so unusual the results are and whether they are too extreme or improbable to be the outcome of chance.
Our starting assumption should always be what's known as the null hypothesis, designated as H 0 , which suggests that nothing statistically significant or out of the ordinary  exists  in  one  variable  or  between  two  data  series.  We  therefore  require extraordinary evidence to reject the null hypothesis and to instead accept the alternative hypothesis, designated as H 1 .
That evidence is the p-value and specifically the generally accepted 5% threshold for significance. While 5% might be somewhat arbitrary, we can agree that it's a very low number, so we're setting a high bar to overturn or reject a null hypothesis.
As  previously  mentioned,  linear  modeling  expects  variables  to  be  normally  distributed, so any predictors that have Shapiro-Wilk test results where the p-value is less than or equal to 0.05 will be withheld from model development. There will be no data transformations or other corrective action applied.
```

---

## [12/52] Statistics_Slam_Dunk_2

| Field | Value |
|-------|-------|
| **Pages** | 158-158 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.770 |
| **Found In** | vector, hybrid, citations |
| **Chunk ID** | `b53ecdbd-0142-4ce0-964f-3e1a11e01afa` |
| **YOUR GRADE** | ____ |

**Full Text (1373 chars):**

```
Hypothesis testing and p-values
Let's take a brief pause to raise a few additional points around hypothesis testing and p-values. Hypothesis testing, or statistical inference, is all about testing an assumption and drawing a conclusion from one or more data series. Hypothesis testing essentially evaluates how unusual or not so unusual the results are and whether they are too extreme or improbable to be the outcome of chance.
Our starting assumption should always be what's known as the null hypothesis, designated as H 0 , which suggests that nothing statistically significant or out of the ordinary  exists  in  one  variable  or  between  two  data  series.  We  therefore  require extraordinary evidence to reject the null hypothesis and to instead accept the alternative hypothesis, designated as H 1 .
That evidence is the p-value and specifically the generally accepted 5% threshold for significance. While 5% might be somewhat arbitrary, we can agree that it's a very low number, so we're setting a high bar to overturn or reject a null hypothesis.
As  previously  mentioned,  linear  modeling  expects  variables  to  be  normally  distributed, so any predictors that have Shapiro-Wilk test results where the p-value is less than or equal to 0.05 will be withheld from model development. There will be no data transformations or other corrective action applied.
```

---

## [13/52] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 6-6 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.768 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `661a69b4-3f2b-4ff9-a22e-40a6ba7653a6` |
| **YOUR GRADE** | ____ |

**Full Text (820 chars):**

```
CONTENTS

16.1, 16 Hypothesis testing = Introduction . . . . . . . . . . . . . . . .. 16.1, 16 Hypothesis testing = . . . . . . . . . . . . . .. 16.1, 487 = 487. 16.2, 16 Hypothesis testing = General hypothesis tests . . . . . . . . .. 16.2, 16 Hypothesis testing = . . . . . . . . . . . . . .. 16.2, 487 = 494. 16.3, 16 Hypothesis testing = Connection with confidence intervals . .. 16.3, 16 Hypothesis testing = . . . . . . . . . . . . . .. 16.3, 487 = 495. 16.4, 16 Hypothesis testing = Data example . . . . . . . . . . . . . . .. 16.4, 16 Hypothesis testing = . . . . . . . . . . . . . .. 16.4, 487 = 496. 16.5, 16 Hypothesis testing = P-values . . . . . . . . . . . . . . . . . .. 16.5, 16 Hypothesis testing = . . . . . . . . . . . . . .. 16.5, 487 = 499. 16.6, 16 Hypothesis testing = Discussion . . . . . . . . .
```

---

## [14/52] Data Science Bookcamp

| Field | Value |
|-------|-------|
| **Pages** | 147-147 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.768 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `85936194-c1cf-44d7-b8f8-dafb66d35b5a` |
| **YOUR GRADE** | ____ |

**Full Text (730 chars):**

```
Listing 7.6 Computing the null-hypothesis-driven p-value
```
p_value = prob_low_grade + prob_high_grade assert p_value == 2 * prob_high_grade print(f"The p-value is {p_value}") The p-value is 0.08968602177036457
```
Under the null hypothesis, there is approximately a 9% chance of observing the grade extreme at random. It's therefore plausible that the null hypothesis is true and the extreme test average is just a random fluctuation. We haven't definitively proved this, but  our  calculations  raise  serious  doubts  about  restructuring  North  Dakota's  fifthgrade curriculum. What if the average of the South Dakotan class had equaled 85%, not 84%? Let's check if that slight grade shift would have influenced our p-value.
```

---

## [15/52] openintro-statistics

| Field | Value |
|-------|-------|
| **Pages** | 271-271 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.766 |
| **Found In** | vector, hybrid, citations |
| **Chunk ID** | `7b522ca7-d00f-480a-9265-ca5b12968191` |
| **YOUR GRADE** | ____ |

**Full Text (1014 chars):**

```
EXAMPLE 7.25
Complete the hypothesis test started in Example 7.23 and Guided Practice 7.24. Use a significance level of α = 0 . 05. For reference, ¯ x n -¯ x s = 0 . 40, SE = 0 . 26, and the sample sizes were n n = 100 and n s = 50.
We can find the test statistic for this test using the values from Guided Practice 7.24:
<!-- formula-not-decoded -->
The p-value is represented by the two shaded tails in the following plot:
We find the single tail area using software (or the t -table in Appendix C.2). We'll use the smaller of n n -1 = 99 and n s -1 = 49 as the degrees of freedom: df = 49. The one tail area is 0.065; doubling this value gives the two-tail area and p-value, 0.135.
The p-value is larger than the significance value, 0.05, so we do not reject the null hypothesis. There is insufficient evidence to say there is a difference in average birth weight of newborns from North Carolina mothers who did smoke during pregnancy and newborns from North Carolina mothers who did not smoke during pregnancy.
```

---

## [16/52] Introduction to Mathematical Statistics

| Field | Value |
|-------|-------|
| **Pages** | 301-302 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.765 |
| **Found In** | vector, hybrid, citations |
| **Chunk ID** | `bc1701be-7384-4540-81aa-ad0a54d81f1a` |
| **YOUR GRADE** | ____ |

**Full Text (838 chars):**

```
4.7 Chi-Square Tests
Thus the hypothesis to be tested is that p 1 , p 2 , p 3 , and p 4 = 1 -p 1 -p 2 -p 3 have the preceding values in a multinomial distribution with k = 4. This hypothesis is to be tested at an approximate 0.025 significance level by repeating the random experiment n = 80 independent times under the same conditions. Here the np i 0 for i = 1 , 2 , 3 , 4, are, respectively, 5, 15, 25, and 35. Suppose the observed frequencies of A 1 , A 2 , A 3 , and A 4 are 6, 18, 20, and 36, respectively. Then the observed value
of Q 3 = 4 ( X i np i 0 ) 2 / ( np i 0
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
The following R segment calculates the test and p -value:
x=c(6,18,20,36); ps=c(1,3,5,7)/16; chisq.test(x,p=ps)
X-squared = 1.8286, df = 3, p-value = 0.6087
Hence, we fail to reject H 0 at level 0 . 0250.
```

---

## [17/52] Robert Johansson - Numerical Python: Scientific Computing and Data Science Applications with Numpy SciPy and Matplotlib-Apress 2019

| Field | Value |
|-------|-------|
| **Pages** | 475-475 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.765 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `5e3d4dbc-8563-48ff-9e80-17741b7753fe` |
| **YOUR GRADE** | ____ |

**Full Text (1471 chars):**

```
Hypothesis Testing
Once H 0 and HA are defined, the data that support the test must be collected, for example, through measurements, observations, or a survey. The next step is to find a test statistics that can be computed from the data and whose probability distribution function can be found under the null hypothesis. Next we can evaluate the data by computing the probability (the p-value ) of obtaining the observed value of the test statistics (or a more extreme one) using the distribution function that is implied by the null hypothesis. If the p -value is smaller than a predetermined threshold, known as the significance level, and denoted by α (typically 5% or 1%), we can conclude that the observed data is unlikely to have been described by the distribution corresponding to the null hypothesis. In that case, we can therefore reject the null hypothesis in favor of the alternative hypothesis. The steps for carrying out a hypothesis test are summarized in the following list:
1. Formulate the null hypothesis and the alternative hypothesis.
2. Select a test statistics such that its sampling distribution under the null hypothesis is known (exactly or approximately).
3. Collect data.
4. Compute the test statistics from the data and calculate its p -value under the null hypothesis.
5. If the p -value is smaller than the predetermined significance level α , we reject the null hypothesis. If the p -value is larger, we fail to reject the null hypothesis.
```

---

## [18/52] openintro-statistics

| Field | Value |
|-------|-------|
| **Pages** | 274-274 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.764 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `d44bd1cc-4ba4-450e-abd0-19dba80b3f91` |
| **YOUR GRADE** | ____ |

**Full Text (576 chars):**

```
Exercises
- (a) Are there any underlying structures in these data that should be considered in an analysis? Explain.
- (b) What are the hypotheses for evaluating whether the number of people out on Friday the 6 th is different than the number out on Friday the 13 th ?
- (c) Check conditions to carry out the hypothesis test from part (b).
- (d) Calculate the test statistic and the p-value.
- (e) What is the conclusion of the hypothesis test?
- (f) Interpret the p-value in this context.
- (g) What type of error might have been made in the conclusion of your test? Explain.
```

---

## [19/52] David M Diez, Christopher D Barr, Mine Çetinkaya-Rundel - OpenIntro Statistics-OpenIntro, Inc. (2015)

| Field | Value |
|-------|-------|
| **Pages** | 189-189 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.763 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `da787bc9-48ca-46e7-a8f4-298a0be6f4b6` |
| **YOUR GRADE** | ____ |

**Full Text (1142 chars):**

```
TIP: It is useful to first draw a picture to find the p-value
Figure 4.16: To identify the p-value, the distribution of the sample mean is considered as if the null hypothesis was true. Then the p-value is defined and computed as the probability of the observed ¯ x or an ¯ x even more favorable to H A under this distribution.
- ⊙ Guided Practice 4.28 If the null hypothesis is true, how often should the p-value be less than 0.05? 25
- ⊙ Guided Practice 4.29 Suppose we had used a significance level of 0.01 in the sleep study. Would the evidence have been strong enough to reject the null hypothesis? (The p-value was 0.007.) What if the significance level was α = 0 . 001? 26
- ⊙ Guided Practice 4.30 Ebay might be interested in showing that buyers on its site tend to pay less than they would for the corresponding new item on Amazon. We'll research this topic for one particular product: a video game called Mario Kart for the Nintendo Wii. During early October 2009, Amazon sold this game for $ 46.99. Set up an appropriate (one-sided!) hypothesis test to check the claim that Ebay buyers pay less during auctions at this same time. 27
```

---

## [20/52] book1 1

| Field | Value |
|-------|-------|
| **Pages** | 230-230 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.763 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `302a4ecf-436e-4ee4-ab78-9e66527c8047` |
| **YOUR GRADE** | ____ |

**Full Text (1091 chars):**

```
5.5.3 Null hypothesis significance testing (NHST) and p-values
In the above decision-theoretic (or Neyman-Pearson) approach to hypothesis testing, we had to specify a null hypothesis H 0 as well as an alternative hypothesis H 1 so that we can compute p ( D| H 0 ) and p ( D| H 1 ) . In some cases, it is difficult to define an alternative hypothesis, and we just want to test if a simple null hypothesis is 'plausible' given some data. To do this, we can define a test statistic test ( D ) , and then we can compare its observed value to the value we would expect if the data came from the null hypothesis, test ( ˜ D ) where ˜ D ∼ H 0 . If the observed value is unexpected given H 0 , we reject the null hypothesis . To quantify this, we compute the probability of seeing a test value that is as large or larger than the observed value (assuming that larger values make H 1 more likely). More precisely, we define the p-value to be the probability, under the null hypothesis, of observing a test statistic that is as large or larger than that actually observed:
<!-- formula-not-decoded -->
```

---

## [21/52] Data Science Bookcamp

| Field | Value |
|-------|-------|
| **Pages** | 164-164 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.761 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `1894f505-ba53-48e3-be02-a8008c68e929` |
| **YOUR GRADE** | ____ |

**Full Text (1457 chars):**

```
Summary
-  Statistical  hypothesis  testing  requires  us  to  choose  between  two  competing hypotheses. According to the null hypothesis , a pair of populations are identical. According to the alternative hypothesis , the pair of populations are not identical.
-  To evaluate the null hypothesis, we must compute a p-value . The p-value equals the probability of observing our data when the null hypothesis is true. The null hypothesis  is  rejected  if  the  p-value  is  lower  than  a  specified significance  level threshold. Typically, the significance level is set to 0.05.
-  If we reject the null hypothesis, and the null hypothesis is true, we commit a type I error .  If  we fail to reject the null hypothesis and the alternative hypothesis is true, we commit a type II error.
-  Data dredging increases our risk of type I errors. In data dredging, an experiment is repeated until the p-value falls below the significance level. We can minimize data  dredging  by  carrying  out  a Bonferroni  correction ,  in  which  the  significance level is divided by the experiment count.
-  We can compare a sample mean to a population mean and variance by relying on the central limit theorem. The population variance is needed to compute the SEM. If we're not provided with the population variance, we can estimate the SEM using bootstrapping with replacement .
-  We can compare the means of two distinct samples by running a permutation test .
```

---

## [22/52] Linear Mixed Models: A Practical Guide Using Statistical Software: Third Edition

| Field | Value |
|-------|-------|
| **Pages** | 394-394 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.758 |
| **Found In** | vector |
| **Chunk ID** | `33ddaf69-7bbd-4cd2-9a9a-480bcecfaead` |
| **YOUR GRADE** | ____ |

**Full Text (756 chars):**

```
7.5.1 Likelihood Ratio Tests for Random Effects
TABLE 7.5 : Summary of Hypothesis Test Results for the Dental Veneer Analysis7.1, Test = LRT. 7.1, Estima- tion Method = REML. 7.1, Models Compared (Nested vs. Reference) = 7.1A vs. 7.1. 7.1, Test Statistic Values (Calculation) = χ 2 (0 : 1) = 11 . 2 (858.3 - 847.1). 7.1, p -Value = < . 001. 7.2, Test = LRT. 7.2, Estima- tion Method = REML. 7.2, Models Compared (Nested vs. Reference) = 7.1 vs. 7.2C. 7.2, Test Statistic Values (Calculation) = χ 2 (1) = 0 . 9 (847.1 - 846.2). 7.2, p -Value = 0 . 34. 7.3, Test = LRT. 7.3, Estima- tion Method = ML. 7.3, Models Compared (Nested vs. Reference) = 7.3 vs. 7.1. 7.3, Test Statistic Values (Calculation) = χ 2 (3) = 1 . 8 (845.5 - 843.7). 7.3, p -Value = 0 . 61
```

---

## [23/52] Linear Mixed Models: A Practical Guide Using Statistical Software: Third Edition

| Field | Value |
|-------|-------|
| **Pages** | 369-369 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.758 |
| **Found In** | vector |
| **Chunk ID** | `2049471f-2d7b-464a-bb9a-ec446b2321dc` |
| **YOUR GRADE** | ____ |

**Full Text (930 chars):**

```
Step 2: Select a structure for the random effects (Model 7.1 vs. Model 7.1A) .
```
title "P-Value for Hypothesis 7.1: Model 7.1A vs 7.1"; data _null_; lrtstat = 858.3 -847.1; df = 1; p_value = 0.5*(1-probchi(lrtstat,df)); format p_value 10.8; put "Hypothesis 7.1: " lrtstat= p_value= ; run;
```
We use the probchi() function to obtain the appropriate p -value for the χ 2 1 distribution and weight it by 0.5. Note that the χ 2 0 distribution is not included in the syntax because it contributes zero to the resulting p -value. This syntax results in the following output in the SAS log: glyph[a0]
```
✄ ✂ Hypothesis 7.1: lrtstat=11.2 p_value=0.00040899
```
✁ Based on the result of this test ( p < . 001), we reject the null hypothesis (that we should delete the random effects associated with the intercepts for teeth nested within patients from the model) and keep Model 7.1 as our preferred model at this stage of the analysis.
```

---

## [24/52] openintro-statistics

| Field | Value |
|-------|-------|
| **Pages** | 234-234 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.756 |
| **Found In** | fts |
| **Chunk ID** | `c7d06ad5-2c2d-4d64-a1ea-d9e0fa870a66` |
| **YOUR GRADE** | ____ |

**Full Text (822 chars):**

```
EXAMPLE 6.32
If the null hypothesis is true, the test statistic X 2 = 5 . 89 would be closely associated with a chisquare distribution with three degrees of freedom. Using this distribution and test statistic, identify the p-value.
The chi-square distribution and p-value are shown in Figure 6.9. Because larger chi-square values correspond to stronger evidence against the null hypothesis, we shade the upper tail to represent the p-value. Using statistical software (or the table in Appendix C.3), we can determine that the area is 0.1171. Generally we do not reject the null hypothesis with such a large p-value. In other words, the data do not provide convincing evidence of racial bias in the juror selection.
Figure 6.9: The p-value for the juror hypothesis test is shaded in the chi-square distribution with df = 3.
```

---

## [25/52] Introduction to Mathematical Statistics

| Field | Value |
|-------|-------|
| **Pages** | 8-8 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.755 |
| **Found In** | vector |
| **Chunk ID** | `80c5fa8b-4bb0-4d4c-9a5a-8d2007c1a2e0` |
| **YOUR GRADE** | ____ |

**Full Text (804 chars):**

```
Contents

 . . . 261. 4.5, 2.6.1 = Introduction to Hypothesis Testing. 4.5, ∗ Multivariate Variance-Covariance Matrix . . . . . . . . . . . 140 = . . . . . . . . . . . . . . . . . . . 267. 4.6, 2.6.1 = Additional Comments About Statistical Tests . . . . . . . . . . .. 4.6, ∗ Multivariate Variance-Covariance Matrix . . . . . . . . . . . 140 = . . 275. , 2.6.1 = 4.6.1. , ∗ Multivariate Variance-Covariance Matrix . . . . . . . . . . . 140 = Observed Significance Level, p -value . . . . . . . . . . . . . . 279. , 2.6.1 = .. , ∗ Multivariate Variance-Covariance Matrix . . . . . . . . . . . 140 = .. , 2.6.1 = . , ∗ Multivariate Variance-Covariance Matrix . . . . . . . . . . . 140 = Accept-Reject Generation Algorithm . . . . . . . . . . . . . . 298. 4.7, 2.6.1 = Chi-Square Tests . . . . . . . . . . .
```

---

## [26/52] openintro-statistics

| Field | Value |
|-------|-------|
| **Pages** | 195-195 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.754 |
| **Found In** | fts |
| **Chunk ID** | `01ea6dac-f20a-4f36-82e1-f89ff5b167e4` |
| **YOUR GRADE** | ____ |

**Full Text (920 chars):**

```
CHECKING SUCCESS-FAILURE AND COMPUTING SE ˆ P FOR A HYPOTHESIS TEST
When using the p-value method to evaluate a hypothesis test, we check the conditions for ˆ p and construct the standard error using the null value, p 0 , instead of using the sample proportion.
In a hypothesis test with a p-value, we are supposing the null hypothesis is true, which is a different mindset than when we compute a confidence interval. This is why we use p 0 instead of ˆ p when we check conditions and compute the standard error in this context.
When we identify the sampling distribution under the null hypothesis, it has a special name: the null distribution . The p-value represents the probability of the observed ˆ p , or a ˆ p that is more extreme, if the null hypothesis were true. To find the p-value, we generally find the null distribution, and then we find a tail area in that distribution corresponding to our point estimate.
```

---

## [27/52] Data Science Bookcamp

| Field | Value |
|-------|-------|
| **Pages** | 149-149 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.752 |
| **Found In** | fts |
| **Chunk ID** | `c2099995-3492-49bc-8f00-62005d061c21` |
| **YOUR GRADE** | ____ |

**Full Text (1488 chars):**

```
7.2 Data dredging: Coming to false conclusions through oversampling
The first roommate fundamentally misconstrued the meaning of the p-value. He wrongly assumed it represents the probability of the null hypothesis being true. In fact, the p-value represents the probability of observing deviations if the null hypothesis is true. The difference between the definitions is subtle but very important: the first definition implies that the null hypothesis is likely to be false if the p-value is low; but the second definition guarantees that we'll eventually observe a low p-value by repeatedly counting candies, even when the null hypothesis is true. Furthermore, the frequency of low p-value observations will equal the p-value itself. Hence, if we open 100 bags of candy, we should expect to observe a p-value of 0.05 approximately five times. By taking random measurements repeatedly, we will eventually obtain a statistically significant result, even if no statistical significance exists!
Running the same experiment too many times increases our risk of type I errors. Let's explore this notion in the context of our fifth-grade exam analysis. Suppose that North Dakota's statewide test performance does not diverge from the exam results in the other 49 states. More precisely, we'll assume that the national mean and variance equal  North  Dakota's population_mean and population_variance exam-grade results. Thus, the null hypothesis is true for all the states in the United States.
```

---

## [28/52] Malcolm Sherrington - Mastering Julia-Packt Publishing 2015

| Field | Value |
|-------|-------|
| **Pages** | 205-205 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.750 |
| **Found In** | vector |
| **Chunk ID** | `fc732b80-772f-458a-a4d9-f5183d81fec3` |
| **YOUR GRADE** | ____ |

**Full Text (841 chars):**

```
Hypothesis testing
```
UnequalVarianceTTest( Two sample t-test (unequal variance) ------------------------------------Population details: parameter of interest:   Mean difference value under h_0:         0 point estimate:          4.4492 95% confidence interval: (-0.1837, 9.0821) Test summary: outcome with 95% confidence: fail to reject h_0 two-sided p-value:       0.05963 (not significant) Details: number of observations:   [56,65] t-statistic:              1.9032531870995715 degrees of freedom:       109.74148002018097 empirical standard error: 2.337668920946911
```
df68107[complete_cases(df68107[[:Written, :Course]]), :]; df68411[complete_cases(df68411[[:Written, :Course]]), :]; We can test the hypothesis that the two distributions have same means with differing variances: float(df68107[:Written]), float64(df68411[:Written]) )
```

---

## [29/52] ROS

| Field | Value |
|-------|-------|
| **Pages** | 80-80 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.749 |
| **Found In** | vector |
| **Chunk ID** | `ce855077-a3e6-46e5-9f68-18a07c0cf3b9` |
| **YOUR GRADE** | ____ |

**Full Text (1280 chars):**

```
4.7 Moving beyond hypothesis testing
- Present all your comparisons. The paper quoted on page 62 leads us through various comparisons and p -values that represent somewhat arbitrary decisions throughout of what to look for. It would be better to display and analyze more data, for example a comparison of respondents in different parts of their cycle on variables such as birth year, party identification, and marital status, along with seeing the distribution of reported days of the menstrual cycle. In this particular case we would not expect to find anything interesting, as any real underlying patterns will be much less than the variation, but speaking generally we recommend displaying more of your data rather than focusing on comparisons that happen to reach statistical significance. The point here is not to get an improved p -value via a multiple comparisons correction but rather to see the big picture of the data. We recognize that, compared to the usual deterministically framed summary, this might represent a larger burden of effort for the consumer of the research as well as the author of the paper.
- Make your data public (subject to any confidentiality restrictions). If the topic is worth studying, you should want others to be able to make rapid progress.
```

---

## [30/52] Regression and Other Stories

| Field | Value |
|-------|-------|
| **Pages** | 84-84 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.749 |
| **Found In** | vector |
| **Chunk ID** | `cedc46e0-f82a-40ba-bd76-5d44ed41e6a0` |
| **YOUR GRADE** | ____ |

**Full Text (1280 chars):**

```
4.7 Moving beyond hypothesis testing
- Present all your comparisons. The paper quoted on page 62 leads us through various comparisons and p -values that represent somewhat arbitrary decisions throughout of what to look for. It would be better to display and analyze more data, for example a comparison of respondents in different parts of their cycle on variables such as birth year, party identification, and marital status, along with seeing the distribution of reported days of the menstrual cycle. In this particular case we would not expect to find anything interesting, as any real underlying patterns will be much less than the variation, but speaking generally we recommend displaying more of your data rather than focusing on comparisons that happen to reach statistical significance. The point here is not to get an improved p -value via a multiple comparisons correction but rather to see the big picture of the data. We recognize that, compared to the usual deterministically framed summary, this might represent a larger burden of effort for the consumer of the research as well as the author of the paper.
- Make your data public (subject to any confidentiality restrictions). If the topic is worth studying, you should want others to be able to make rapid progress.
```

---

## [31/52] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 90-91 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.749 |
| **Found In** | fts |
| **Chunk ID** | `a2526fb4-444f-4dfe-a382-c1c4c1aa6376` |
| **YOUR GRADE** | ____ |

**Full Text (1307 chars):**

```
9.3.1 Interpret the p-value
We describe a finding as statistically significant by interpreting the p-value. For example, we may perform a Student's t-test on two data samples and find that it is unlikely that the samples have the same mean. We reject the null hypothesis that the samples have the same mean at a chosen level of statistical significance (or confidence). A statistical hypothesis test may return a value called p or the p-value. This is a quantity that we can use to interpret or quantify the result of the test and either reject or fail to reject the null hypothesis. This is done by comparing the p-value to a threshold value chosen beforehand called the significance level.
The significance level is often referred to by the Greek lower case letter alpha ( α ). A common value used for alpha is 5% or 0.05. A smaller alpha value suggests a more robust interpretation of the result, such as 1% or 0.01%. The p-value is compared to the pre-chosen alpha value. A result is statistically significant when the p-value is less than or equal to alpha. This signifies a change was detected: that the default or null hypothesis can be rejected.
- p-value ≤ alpha : significant result, reject null hypothesis (H1).
- p-value > alpha : not significant result, fail to reject the null hypothesis (H0).
```

---

## [32/52] Schaum's Outline of Probability and Statistics, Third Edition 2009

| Field | Value |
|-------|-------|
| **Pages** | 241-241 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.749 |
| **Found In** | vector |
| **Chunk ID** | `0b1951c9-4038-4811-8cfc-82621d9f1d87` |
| **YOUR GRADE** | ____ |

**Full Text (1139 chars):**

```
Tests involving student's t distribution
- (c) The P value is . The table in Appendix D shows 0.1 < P < 0.2. By computer software, . P = 0.158 P ( T ≥ 1.45) + P ( T ≤ -1.45)
- 7.19. At an agricultural station it was desired to test the effect of a given fertilizer on wheat production. To accomplish this, 24 plots of land having equal areas were chosen; half of these were treated with the fertilizer and the other half were untreated (control group). Otherwise the conditions were the same. The mean yield of wheat on the untreated plots was 4.8 bushels with a standard deviation of 0.40 bushels, while the mean yield on the treated plots was 5.1 bushels with a standard deviation of 0.36 bushels. Can we conclude that there is a significant improvement in wheat production because of the fertilizer if a significance level of (a) 1%, (b) 5% is used? (c) What is the P value of the test?
If 1 and 2 denote population mean yields of wheat on treated and untreated land, respectively, we have to decide between the hypotheses m m
H 0 : , and the difference is due to chance m 1 = m 2
H 1 : , and the fertilizer improves the yield m 1 > m 2
```

---

## [33/52] Linear Mixed Models: A Practical Guide Using Statistical Software: Third Edition

| Field | Value |
|-------|-------|
| **Pages** | 427-428 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.748 |
| **Found In** | vector |
| **Chunk ID** | `8a1f7a23-f98c-481a-93c1-e99f581c3230` |
| **YOUR GRADE** | ____ |

**Full Text (1045 chars):**

```
8.4.1 SAS
```
title "Hypothesis 8.1"; proc mixed data = satmath covtest; class studid tchrid; model math = year / solution; random int / subject = tchrid; run;
```
The -2 REML log-likelihood value for this reduced two-level model is 2170.3, and the corresponding value for Model 8.1 was 2123.6. We compute the p -value for the likelihood ratio test using the following syntax:
```
title "p-value for Hypothesis 8.1"; data _null_; lrtstat = 2170.3 - 2123.6; df = 1;
```
```
pvalue = 0.5 * (1 - probchi(lrtstat, df)); format pvalue 10.8; put lrtstat = df = pvalue = ; run;
```
We have very strong evidence ( p < 0 . 001) against the null hypothesis in this case, and choose to retain the random student effects in the model; there is clear evidence of substantial between-student variance in performance on the math test, as was apparent in the initial data summary.
We test Hypothesis 8.2 using a similar approach. We first fit a reduced model without the random teacher effects, and then compute the likelihood ratio test statistic and p -value:
```

---

## [34/52] Schaum's Outline of Probability and Statistics, Third Edition 2009

| Field | Value |
|-------|-------|
| **Pages** | 235-235 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.748 |
| **Found In** | vector |
| **Chunk ID** | `e5c2ea70-2141-4ca4-ad3d-adda5dee17be` |
| **YOUR GRADE** | ____ |

**Full Text (1110 chars):**

```
Third method
- (b) The P value of the test is P , which shows that the claim is almost certainly false. That is, if H 0 were true, it is almost certain that a random sample of 200 allergy sufferers who used the medicine would include more than 160 people who found relief. ( Z ≤ -4.73) < 0
- 7.7. The mean lifetime of a sample of 100 fluorescent light bulbs produced by a company is computed to be 1570 hours with a standard deviation of 120 hours. If is the mean lifetime of all the bulbs produced by the company, test the hypothesis hours against the alternative hypothesis hours, using a level of significance of (a) 0.05 and (b) 0.01. (c) Find the P value of the test. m 2 1600 m = 1600 m
We must decide between the two hypotheses
<!-- formula-not-decoded -->
A two-tailed test should be used here since includes values both larger and smaller than 1600. m 2 1600
- (a) For a two-tailed test at a level of significance of 0.05, we have the following decision rule:
- (1) Reject H 0 if the z score of the sample mean is outside the range to 1.96. -1.96
- (2) Accept H 0 (or withhold any decision) otherwise.
```

---

## [35/52] openintro-statistics

| Field | Value |
|-------|-------|
| **Pages** | 239-239 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.745 |
| **Found In** | vector |
| **Chunk ID** | `b33ae31d-9931-48cd-8d4f-932d1aab4e4c` |
| **YOUR GRADE** | ____ |

**Full Text (1083 chars):**

```
Exercises
6.33 Open source textbook. A professor using an open source introductory statistics book predicts that 60% of the students will purchase a hard copy of the book, 25% will print it out from the web, and 15% will read it online. At the end of the semester he asks his students to complete a survey where they indicate what format of the book they used. Of the 126 students, 71 said they bought a hard copy of the book, 30 said they printed it out from the web, and 25 said they read it online.
- (a) State the hypotheses for testing if the professor's predictions were inaccurate.
- (b) How many students did the professor expect to buy the book, print the book, and read the book exclusively online?
- (c) This is an appropriate setting for a chi-square test. List the conditions required for a test and verify they are satisfied.
- (d) Calculate the chi-squared statistic, the degrees of freedom associated with it, and the p-value.
- (e) Based on the p-value calculated in part (d), what is the conclusion of the hypothesis test? Interpret your conclusion in this context.
```

---

## [36/52] Schaum's Outline of Probability and Statistics, Third Edition 2009

| Field | Value |
|-------|-------|
| **Pages** | 240-240 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.744 |
| **Found In** | vector |
| **Chunk ID** | `77797eb9-1f95-455c-8ec2-b90c9c77dbf9` |
| **YOUR GRADE** | ____ |

**Full Text (1025 chars):**

```
Tests involving student's t distribution
- (c) The P value is . The table in Appendix D shows that 0.01 < P < 0.02. Using computer software, we find . P = 0.015 P ( T ≥ 3) + P ( T ≤ -3)
- 7.17. A test of the breaking strengths of 6 ropes manufactured by a company showed a mean breaking strength of 7750 lb and a standard deviation of 145 lb, whereas the manufacturer claimed a mean breaking strength of 8000 lb. Can we support the manufacturér's claim at a level of significance of (a) 0.05, (b) 0.01? (c) What is the P value of the test?
We must decide between the hypotheses
H 0 : lb, and the manufacturer's claim is justified m = 8000
H 1 : lb, and the manufacturer's claim is not justified m < 8000
so that a one-tailed test is required.
Under the hypothesis H 0 , we have
<!-- formula-not-decoded -->
- (a) For a one-tailed test at a 0.05 level of significance, we adopt the decision rule:
- (1) Accept H 0 if T is greater than -t 0.95 , which for 6 -1 = 5 degrees of freedom means T >-2.01.
- (2) Reject H 0 otherwise.
```

---

## [37/52] Why: A Guide to Finding and Using Causes

| Field | Value |
|-------|-------|
| **Pages** | 62-62 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.744 |
| **Found In** | vector |
| **Chunk ID** | `2bf7ecc1-362d-4b2d-8399-6ff2bb441ba2` |
| **YOUR GRADE** | ____ |

**Full Text (1052 chars):**

```
Multiple testing and p-values
Given that the subject in this study was a dead salmon, that seems unlikely. So how can a deceased fish seemingly respond to visual stimuli? The results here would be reported as very significant by any usual threshold, so it's not a matter of trying to exaggerate their significance-but to understand how such a result could be, we need a brief statistical interlude.
Researchers often want to determine if an effect is significant (is a correlation genuine or a statistical artifact?) or whether there's a difference between two groups (are different regions of the brain active when people look at humans versus  at  animals?),  but  need  some  quantitative  measure  to  objectively  determine which of their findings are meaningful. One common measure of significance is what's called a p-value, which is used for comparing two hypotheses (known as the null and alternate hypotheses).
A p-value tells you the probability of seeing a result at least as extreme as the observed result if the null hypothesis were true.
```

---

## [38/52] Linear Mixed Models: A Practical Guide Using Statistical Software: Third Edition

| Field | Value |
|-------|-------|
| **Pages** | 373-373 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.744 |
| **Found In** | vector |
| **Chunk ID** | `2e9bb420-33ee-4978-b5d3-a5d9ec01f7e0` |
| **YOUR GRADE** | ____ |

**Full Text (481 chars):**

```
Step 3: Select a covariance structure for the residual errors (Model 7.1 and Model 7.2A through Model 7.2C) .
```
title "P-Value for Hypothesis 7.2: Model 7.1 vs 7.2C"; data _null_; lrtstat = 847.1 -846.2; df = 1; p_value =(1-probchi(lrtstat,df)); format p_value 10.8; put "Hypothesis 7.2: " df= p_value= ; run;
```
Because the test is not significant ( p = 0 . 34), we keep Model 7.1, with homogeneous residual error variance, as our preferred model at this stage of the analysis.
```

---

## [39/52] Schaum's Outline of Probability and Statistics, Third Edition 2009

| Field | Value |
|-------|-------|
| **Pages** | 235-235 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.742 |
| **Found In** | vector |
| **Chunk ID** | `a4440f95-7c4a-4757-8ced-b11575a7bd92` |
| **YOUR GRADE** | ____ |

**Full Text (813 chars):**

```
Third method
- (b) If the level of significance is 0.01, the range to 1.96 in the decision rule of part (a) is replaced by to 2.58. Then since the z score of lies inside this range, we accept H 0 (or withhold any decision) at a 0.01 level of significance. -2.50 -2.58 -1.96
- (c) The P value of the two-tailed test is , which is the probability that a mean lifetime of less than 1570 hours or more than 1630 hours would occur by chance if H 0 were true. P ( Z ≤ -2.50) + P ( Z ≥ 2.50) = 0.0124
- 7.8. In Problem 7.7 test the hypothesis hours against the alternative hypothesis hours, using a level of significance of (a) 0.05, (b) 0.01. (c) Find the P value of the test. m < 1600 m = 1600
We must decide between the two hypotheses
<!-- formula-not-decoded -->
A one-tailed test should be used here (see Fig. 7-5).
```

---

## [40/52] Introduction to Mathematical Statistics

| Field | Value |
|-------|-------|
| **Pages** | 297-297 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.742 |
| **Found In** | vector |
| **Chunk ID** | `7fb7e4fb-b93d-484e-8916-cbfff11cc360` |
| **YOUR GRADE** | ____ |

**Full Text (818 chars):**

```
4.6.1 Observed Significance Level, p -value
For the p -value, compute each of the one-sided p -values, take the smaller p -value, and double it. For an illustration, in the Darwin example, suppose the the hypotheses are H 0 : µ = 0 versus H 1 : µ = 0. Then the p -value is 2(0 . 0248) = 0 . 0496. As a final note on p -values for two-sided hypotheses, suppose the test statistic can be expressed in terms of a t -test statistic. In this case the p -value can be found equivalently as follows. If d is the realized value of the t -test statistic then the p -value is
/negationslash
<!-- formula-not-decoded -->
where, under H 0 , t has a t -distribution with n -1 degrees of freedom.
In this discussion on p -values, keep in mind that good science dictates that the hypotheses should be known before the data are drawn.
```

---

## [41/52] David M Diez, Christopher D Barr, Mine Çetinkaya-Rundel - OpenIntro Statistics-OpenIntro, Inc. (2015)

| Field | Value |
|-------|-------|
| **Pages** | 188-188 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.738 |
| **Found In** | fts |
| **Chunk ID** | `3ccab9a0-ed36-44b1-9cde-8a520985286f` |
| **YOUR GRADE** | ____ |

**Full Text (1313 chars):**

```
TIP: It is useful to first draw a picture to find the p-value
It is useful to draw a picture of the distribution of ¯ x as though H 0 was true (i.e. µ equals the null value), and shade the region (or regions) of sample means that are at least as favorable to the alternative hypothesis. These shaded regions represent the p-value.
The ideas below review the process of evaluating hypothesis tests with p-values:
- The null hypothesis represents a skeptic's position or a position of no difference. We reject this position only if the evidence strongly favors H A .
- A small p-value means that if the null hypothesis is true, there is a low probability of seeing a point estimate at least as extreme as the one we saw. We interpret this as strong evidence in favor of the alternative.
- We reject the null hypothesis if the p-value is smaller than the significance level, α , which is usually 0.05. Otherwise, we fail to reject H 0 .
- We should always state the conclusion of the hypothesis test in plain language so non-statisticians can also understand the results.
The p-value is constructed in such a way that we can directly compare it to the significance level ( α ) to determine whether or not to reject H 0 . This method ensures that the Type 1 Error rate does not exceed the significance level standard.
```

---

## [42/52] Data Science Bookcamp

| Field | Value |
|-------|-------|
| **Pages** | 147-147 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.736 |
| **Found In** | fts |
| **Chunk ID** | `d534548b-d833-4720-8033-c32a47131d81` |
| **YOUR GRADE** | ____ |

**Full Text (902 chars):**

```
Listing 7.6 Computing the null-hypothesis-driven p-value
```
def compute_p_value(observed_mean, population_mean, sem): mean_diff = abs(population_mean - observed_mean) prob_high = stats.norm.sf(population_mean + mean_diff, population_mean, sem) return 2 * prob_high new_p_value = compute_p_value(85, mean, sem) print(f"The updated p-value is {new_p_value}") The updated p-value is 0.03389485352468927
```

Listing 7.7 Computing the p-value for an adjusted sample mean
A tiny increase in the average grade has caused a threefold decrease in the p-value. Now, under the null hypothesis, there's only a 3.3% chance of observing an average test grade that's  at  least  as  extreme  as  85%.  This  likelihood  is  low,  and  we  might  therefore  be tempted to reject the null hypothesis. Should we accept the alternative hypothesis and invest our time and money in revamping North Dakota's school system?
```

---

## [43/52] (Springer Texts in Statistics'',) Larry Wasserman - All of Statistics  A Concise Course in Statistical Inference (Springer Texts in Statistics)-Springer (2003)

| Field | Value |
|-------|-------|
| **Pages** | 178-179 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.735 |
| **Found In** | fts |
| **Chunk ID** | `5ee6c6c9-8501-4bfa-b35f-51f41e4e29a4` |
| **YOUR GRADE** | ____ |

**Full Text (806 chars):**

```
10.7 Multiple Testing
In some situations we may conduct many hypothesis tests. In example 10.20, there were actually 2,638 genes. If we tested for a difference for each gene, we would be conducting 2,638 separate hypothesis tests. Suppose each test is conducted at level α . For any one test, the chance of a false rejection of the null is α . But the chance of at least one false rejection is much higher. This is the multiple testing problem. The problem comes up in many data mining situations where one may end up testing thousands or even millions of hypotheses. There are many ways to deal with this problem. Here we discuss two methods.
166 10. Hypothesis Testing and p-values
Consider m hypothesis tests:
<!-- formula-not-decoded -->
and let P 1 , . . . , P m denote the m p-values for these tests.
```

---

## [44/52] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 252-252 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.732 |
| **Found In** | fts |
| **Chunk ID** | `27be84c8-386a-4379-b64b-8d164e98cf74` |
| **YOUR GRADE** | ____ |

**Full Text (1033 chars):**

```
28.2 Nonparametric Statistical Significance Tests
In general, each test calculates a test statistic, that must be interpreted with some background in statistics and a deeper knowledge of the statistical test itself. Tests also return a p-value that can be used to interpret the result of the test. The p-value can be thought of as the probability of observing the two data samples given the base assumption (null hypothesis) that the two samples were drawn from a population with the same distribution. The p-value can be interpreted in the context of a chosen significance level called alpha. A common value for alpha is 5% or 0.05. If the p-value is below the significance level, then the test says there is enough evidence to reject the null hypothesis and that the samples were likely drawn from populations with differing distributions.
- p-value ≤ alpha : significant result, reject null hypothesis, distributions differ (H1).
- p-value > alpha : not significant result, fail to reject null hypothesis, distributions same (H0).
```

---

## [45/52] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 108-109 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.730 |
| **Found In** | fts |
| **Chunk ID** | `58d95a52-a9ab-4df1-a049-e73450a6577f` |
| **YOUR GRADE** | ____ |

**Full Text (1131 chars):**

```
11.2 Why Do We Need Critical Values?
Many statistical hypothesis tests return a p-value that is used to interpret the outcome of the test. Some tests do not return a p-value, requiring an alternative method for interpreting the calculated test statistic directly. A statistic calculated by a statistical hypothesis test can be interpreted using critical values from the distribution of the test statistic. Some examples of
statistical hypothesis tests and their distributions from which critical values can be calculated are as follows:
- Z-Test : Gaussian distribution.
- Student's t-Test : Student's t-distribution.
- Chi-Squared Test : Chi-Squared distribution.
- ANOVA : F-distribution.
Critical values are also used when defining intervals for expected (or unexpected) observations in distributions. Calculating and using critical values may be appropriate when quantifying the uncertainty of estimated statistics or intervals such as confidence intervals and tolerance intervals. Note, a p-value can be calculated from a test statistic by retrieving the probability from the test statistics cumulative density function (CDF).
```

---

## [46/52] David M Diez, Christopher D Barr, Mine Çetinkaya-Rundel - OpenIntro Statistics-OpenIntro, Inc. (2015)

| Field | Value |
|-------|-------|
| **Pages** | 305-305 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.728 |
| **Found In** | fts |
| **Chunk ID** | `0bc32702-657b-4a29-b88d-cec5a4895d06` |
| **YOUR GRADE** | ____ |

**Full Text (1163 chars):**

```
One-sided hypothesis test for p with a small sample
The p-value is always derived by analyzing the null distribution of the test statistic. The normal model poorly approximates the null distribution for ˆ p when the success-failure condition is not satisfied. As a substitute, we can generate the null distribution using simulated sample proportions (ˆ p sim ) and use this distribution to compute the tail area, i.e. the p-value.
We continue to use the same rule as before when computing the p-value for a twosided test: double the single tail area, which remains a reasonable approach even when the sampling distribution is asymmetric. However, this can result in p-values larger than 1 when the point estimate is very near the mean in the null distribution; in such cases, we write that the p-value is 1. Also, very large p-values computed in this way (e.g. 0.85), may also be slightly inflated.
Guided Practice 6.48 said the p-value is estimated . It is not exact because the simulated null distribution itself is not exact, only a close approximation. However, we can generate an exact null distribution and p-value using the binomial model from Section 3.4.
```

---

## [47/52] Causal Inference in Python

| Field | Value |
|-------|-------|
| **Pages** | 76-76 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.727 |
| **Found In** | fts |
| **Chunk ID** | `c043e1b9-7767-4bca-8651-52c4437af43c` |
| **YOUR GRADE** | ____ |

**Full Text (1234 chars):**

```
p-values
Previously, I've said that there is less than a 5% chance you would observe such an extreme difference if the conversion of customers that received no email and short email were the same. But can you precisely estimate what that chance is? How likely are you to observe such an extreme value? Enter p-values!
Like with confidence intervals (and most frequentist statistics, as a matter of fact), the true definition of p-values can be very confusing. So, to not take any risks, I'll copy the definition from Wikipedia: 'the p-value is the probability of obtaining test results at least as extreme as the results actually observed during the test, assuming that the null hypothesis is correct. '
To put it more succinctly, the p-value is the probability of seeing such data, if the null hypothesis  were  true  (see  Figure  2-4).  It  measures  how  unlikely  that  measurement you are seeing is, considering that the null hypothesis is true. Naturally, this often gets confused with the probability of the null hypothesis being true. Note the difference here. The p-value is not P H 0 data , but rather P data H 0 .
Figure 2-4. p-value is the probability of seeing a extreme statistic, given that the null hypothesis is true
```

---

## [48/52] Data Science Bookcamp

| Field | Value |
|-------|-------|
| **Pages** | 159-159 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.726 |
| **Found In** | fts |
| **Chunk ID** | `d71daae1-2c2b-436a-b45c-158c6c2a8561` |
| **YOUR GRADE** | ____ |

**Full Text (1266 chars):**

```
Listing 7.24 Using bootstrapping to estimate the SEM
```
estimated_sem = random_variable.std() p_value = compute_p_value(27, 37, estimated_sem) print(f"P-value computed from estimated SEM is approximately {p_value:.2f}") P-value computed from estimated SEM is approximately 0.10
```
As expected, the computed p-value is approximately 0.1. We've shown how bootstrapping with replacement provides us with two divergent approaches for computing the p-value. The first approach requires us to do the following:
- 1 Sample with replacement from the data. Repeat tens of thousands of times to obtain a list of sample means.
- 2 Generate a histogram from the sample means.
- 3 Convert the histogram to a distribution using the stats.rv_histogram method.
- 4 Take  the  area  beneath  the  left  and  right  extremes  of  the  distribution  curve using the survival function and the cumulative distribution function.
Meanwhile, the second approach appears to be slightly simpler:
- 1 Sample with replacement from the data. Repeat tens of thousands of times to obtain a list of sample means.
- 2 Compute the standard deviation of the means to approximate the SEM.
- 3 Use  the  estimated  SEM  to  carry  out  basic  hypothesis  testing  using  our com-pute_p_value function.
```

---

## [49/52] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 217-218 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.704 |
| **Found In** | fts |
| **Chunk ID** | `56c3a768-906c-4014-9f15-370cae00c27e` |
| **YOUR GRADE** | ____ |

**Full Text (1283 chars):**

```
24.5.1 Interpretation of a Test
Before you can apply the statistical tests, you must know how to interpret the results. Each test will return at least two things:
- Statistic : A quantity calculated by the test that can be interpreted in the context of the test via comparing it to critical values from the distribution of the test statistic.
- p-value : Used to interpret the test, in this case whether the sample was drawn from a Gaussian distribution.
Each test calculates a test-specific statistic. This statistic can aid in the interpretation of the result, although it may require a deeper proficiency with statistics and a deeper knowledge of
the specific statistical test. Instead, the p-value can be used to quickly and accurately interpret the statistic in practical applications. The tests assume that the sample was drawn from a Gaussian distribution. Technically this is called the null hypothesis, or H0. A threshold level is chosen called alpha, typically 5% (or 0.05), that is used to interpret the p-value. In the SciPy implementation of these tests, you can interpret the p value as follows.
- p-value ≤ alpha : significant result, reject null hypothesis, not Gaussian (H1).
- p-value > alpha : not significant result, fail to reject null hypothesis, Gaussian (H0).
```

---

## [50/52] openintro-statistics

| Field | Value |
|-------|-------|
| **Pages** | 200-200 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.703 |
| **Found In** | fts |
| **Chunk ID** | `9bc8650e-2c6f-4c8d-be3d-85f5dcd5d0a0` |
| **YOUR GRADE** | ____ |

**Full Text (844 chars):**

```
5.3.7 One-sided hypothesis tests (special topic)
In the entire hypothesis testing procedure, there is only one difference in evaluating a onesided hypothesis test vs a two-sided hypothesis test: how to compute the p-value. In a one-sided hypothesis test, we compute the p-value as the tail area in the direction of the alternative hypothesis only , meaning it is represented by a single tail area. Herein lies the reason why one-sided tests are sometimes interesting: if we don't have to double the tail area to get the p-value, then the p-value is smaller and the level of evidence required to identify an interesting finding in the direction of the alternative hypothesis goes down. However, one-sided tests aren't all sunshine and rainbows: the heavy price paid is that any interesting findings in the opposite direction must be disregarded.
```

---

## [51/52] (Springer Texts in Statistics'',) Larry Wasserman - All of Statistics  A Concise Course in Statistical Inference (Springer Texts in Statistics)-Springer (2003)

| Field | Value |
|-------|-------|
| **Pages** | 170-171 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.689 |
| **Found In** | fts |
| **Chunk ID** | `f83ccbd5-4a70-4bd2-8ed3-a7901c607d9e` |
| **YOUR GRADE** | ____ |

**Full Text (1069 chars):**

```
10.2 p-values
Warning! A large p-value is not strong evidence in favor of H 0 . A large p-value can occur for two reasons: (i) H 0 is true or (ii) H 0 is false but the test has low power.

< . 01, evidence = very strong evidence against H 0. .01 - .05, evidence = strong evidence against H 0. .05 - .10, evidence = weak evidence against H 0. > . 1, evidence = little or no evidence against H 0
Warning! Do not confuse the p-value with P ( H 0 | Data). 2 The p-value is not the probability that the null hypothesis is true.
The following result explains how to compute the p-value.
2 We discuss quantities like P ( H 0 | Data) in the chapter on Bayesian inference.
- 10.12 Theorem. Suppose that the size α test is of the form
<!-- formula-not-decoded -->
Then,
<!-- formula-not-decoded -->
where x n is the observed value of X n . If Θ 0 = { θ 0 } then
<!-- formula-not-decoded -->
We can express Theorem 10.12 as follows:
The p-value is the probability (under H 0 ) of observing a value of the test statistic the same as or more extreme than what was actually observed.
```

---

## [52/52] Statistics Slam Dunk

| Field | Value |
|-------|-------|
| **Pages** | 346-346 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.623 |
| **Found In** | fts |
| **Chunk ID** | `11d1ea8a-e5b1-4334-b225-bbff17b5db25` |
| **YOUR GRADE** | ____ |

**Full Text (1123 chars):**

```
RUNNING THE ALGORITHM
Bear  in  mind  that  neither salarytotal nor wintotal are  normally  distributed.  The K-means algorithm works best -it still otherwise works-when the plotted variables are evenly distributed around their means. A Shapiro-Wilk test returns a p-value that tells us whether or not a numeric variable is normally distributed. Our null hypothesis is a normal  distribution;  we  would  therefore  reject  that  hypothesis  if  Shapiro-Wilk returned a p-value below 5%. That being said, we pass the variables salarytotal and wintotal to the base R shapiro.test() function:
```
shapiro.test(final_kmeans$salarytotal) ##  Shapiro-Wilk normality test ## ## data:  final_kmeans$salarytotal ## W = 0.82724, p-value = 0.0002144 shapiro.test(final_kmeans$wintotal) ##  Shapiro-Wilk normality test ## ## data:  final_kmeans$wintotal ## W = 0.89706, p-value = 0.007124
```
Both tests  returned  p-values  below  the  5%  threshold  for  significance;  so  we  would reject the null hypothesis twice and conclude that both variables aren't normally distributed. This might help explain the anomalies in the results.
```

---
