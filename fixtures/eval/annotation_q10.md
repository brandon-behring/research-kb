# Query 10: sample size power calculation

**Domain:** statistics
**Query ID:** q_stat_002
**Candidates:** 50
**Grading:** 0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant

---

## [1/50] Vol 2: Causal Inference

| Field | Value |
|-------|-------|
| **Pages** | 94-94 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.800 |
| **Found In** | vector |
| **Chunk ID** | `42e44583-7f8b-4937-bb79-325d62f97240` |
| **YOUR GRADE** | ____ |

**Full Text (132 chars):**

```
5.4 Sample Size and Power
Statistical Power : The probability of detecting a true effect. Power = 1 -β = P ( reject H 0 | H 1 true )
```

---

## [2/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 366-367 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.798 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `de88f84a-6713-4426-b81c-2cc6486e4f9a` |
| **YOUR GRADE** | ____ |

**Full Text (1223 chars):**

```
13.1. POWER AND SAMPLE-SIZE CALCULATIONS FOR I NTERACTION FOR CONTINUOUS OUTCOMES
where σ 2 is the variance of the error term in the regression model for Y -that is, the variance of Y conditional on G and E .
To calculate the sample size, we would need to specify (i) the significance level α , the power β , and the magnitude of the interaction Vcts = σ 2 ( 1 π 00 + 1 π 10 + 1 π 01 + 1 π 11 ) = η and (ii) the proportion of subjects in each exposure stratum, π 00, π 10, π 01, π 11.
If instead of calculating the required sample size for a fixed power β , we wanted to calculate the power for a given sample size using the Wald test for the null hypothesis τ 3 = 0 based on the linear regression model, we could proceed as follows. For a fixed sample size n the power to reject the null τ 3 = 0 at significance level α under the alternative that τ 3 = η is given by where /Phi1 -1 is the inverse cumulative distribution function for a standard normal random variable and where Vcts can be calculated as above.
<!-- formula-not-decoded -->
Finally, if the null hypothesis were rejected for extreme values of τ 3 on either side of zero (two-sided test), then the relevant power formula would be
<!-- formula-not-decoded -->
```

---

## [3/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 376-377 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.794 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `3899dc73-13a4-43a4-9be2-4bfe80df459e` |
| **YOUR GRADE** | ____ |

**Full Text (1019 chars):**

```
13.3. POWER AND SAMPLE SIZE CALCULATIONS FOR BINARY OUTCOMES: ADDITIVE INTERACTION
If instead of calculating the required sample size for a fixed power β , we wanted to calculate the power for a given sample size using the Wald test for the null hypothesis θ 3 = 0 based on the linear risk model, we could proceed as follows. For a fixed sample size n , the power to reject the null θ 3 = 0 at significance level α under the alternative that θ 3 = η is given by where
where /Phi1 -1 is the inverse cumulative distribution function for a standard normal random variable and where V can be calculated as above. Below in Section 13.5 we describe how to use a simple Excel spreadsheet to carry out such sample size and power calculations automatically. If the null hypothesis were rejected for extreme values of θ 3 on either side of zero (two-sided test), then the relevant power formula would be
Before moving on, we give a brief example of the use of these formulae for additive interaction.
<!-- formula-not-decoded -->
```

---

## [4/50] David G Kleinbaum Lawrence L Kupper Azhar Nizam Eli S Rosenberg - Applied Regression Analysis and Other Multivariable Methods-Cengage Learning 2013

| Field | Value |
|-------|-------|
| **Pages** | 916-916 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.794 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `4cc771e8-dfe2-49d6-a8ef-810481838274` |
| **YOUR GRADE** | ____ |

**Full Text (1072 chars):**

```
27.5.1    Power and Sample Size Determination for Multiple Linear Regression
1. Determine the critical value Fs , n 2 q 2 s 2 1, 1 2 a .
2. Estimate (or provide values for), using a reasoned approach, r 2 Y | X 1 , p , Xs , X * 1 , p , X * q (the population squared multiple correlation) and r 2 Y 1 X 1 , p , Xs 2 | X * 1 , p , X * q (the population squared multiple partial correlation for variables X 1 ,  .  .  .  , Xs given X * 1 , p , X * q ).
3. Calculate the power as Pr( F l . F 1 2 a ), where F l follows a non-central F distribution with non-centrality parameter
<!-- formula-not-decoded -->
This parameter l captures the effect size that is to be detected. Typically, users rely on software to perform this probability calculation.
There is no straightforward way, in this general setting, to calculate sample sizes for multiple linear regression models. Instead, the power calculation above has to be repeated with different values of n until a sample size that yields an acceptable power is achieved. Computer software makes this iterative approach feasible.
```

---

## [5/50] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 544-544 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.792 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `385a19a7-e926-49a0-99fa-5f2cfa94c4e6` |
| **YOUR GRADE** | ____ |

**Full Text (412 chars):**

```
18.1 Introduction
We will describe the basics of both here. Broadly, in a power calculation, one fixes the sample size and other aspects of the study and calculates an estimate of the power. In a sample size calculation, one fixes the power, then determines the minimum sample size to achieve that power. We will do some examples of both, but we will also dedicate a complete chapter to sample size calculations.
```

---

## [6/50] Volume 1: The Google Data Scientist Interview Guide

| Field | Value |
|-------|-------|
| **Pages** | 499-499 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.790 |
| **Found In** | vector, hybrid |
| **Chunk ID** | `c84ef7e8-9cfe-49a4-898c-0ea98efb91ee` |
| **YOUR GRADE** | ____ |

**Full Text (233 chars):**

```
211.1. 1. Sample Size & Power (Case Study 1)
- Calculate n using: n = 2(Z_ α /2 + Z_ β ) ²σ² / δ²
- Account for high variance metrics (CUPED for variance reduction)
- Consider day-of-week effects, novelty effects in duration planning
```

---

## [7/50] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 559-559 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.789 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `8ece8bf5-d593-4bfe-98d7-738466ab99a9` |
| **YOUR GRADE** | ____ |

**Full Text (874 chars):**

```
19.2.1 Sample size calculation based on the Normal approximation
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
By dividing both the left and right side of the inequality by σ/ √ n/ 2 we obtain
Because it follows that
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
where we used the fact that z 1 -β = -z β due to the symmetry of the standard Normal distribution. Thus, the sample size needed for one group is
<!-- formula-not-decoded -->
and the total sample size for the two groups is
<!-- formula-not-decoded -->
This formula shows what the ingredients are for conducting the sample size calculation. First, we need the size of the test, α , and the power of the test, 1 -β . Suppose that we fix the size at α = 0 . 05 and the power of detecting the alternative at 1 -β = 0 . 9 . Then the first part of the formula can be calculated in R as follows:
```

---

## [8/50] Linear Mixed Models: A Practical Guide Using Statistical Software: Third Edition

| Field | Value |
|-------|-------|
| **Pages** | 448-448 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.788 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `99cd0260-1fac-44de-bb78-73d77a65314e` |
| **YOUR GRADE** | ____ |

**Full Text (1109 chars):**

```
9.1 Introduction
Briefly, the statistical power for a given clustered or longitudinal study design refers to the probability that a study with specified samples sizes at each level of the data hierarchy will be able to detect the effect of interest as being non-zero in an LMM at a given level of significance (e.g., α = 0.05), if the effect is actually non-zero in the true underlying model . That is, if a given model holds in reality, what proportion of studies with specified sample sizes at each level will actually detect the non-zero effect of interest when using the stated significance level? Using this definition of statistical power, we will be able to take advantage of the tremendous flexibility and wide-ranging utility of simulation-based approaches for power analysis, especially when analytic results are not available for direct computation of power.
We first begin with established tools for power and sample size calculation based on known, closed-form analytic results. These tools are straightforward to implement using existing software, but limited to specific types of study designs.
```

---

## [9/50] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 566-566 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.788 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `5dd2d496-d242-48f7-9aac-17bd606474e1` |
| **YOUR GRADE** | ____ |

**Full Text (1026 chars):**

```
19.2.6 R code
Now that we understand where all the sample size calculations for equality in mean are coming from, we can forget everything, and focus on R . One function in R that calculates power in R for a variety of scenarios is power.t.test . To get more information about the function simply type
```
?power.t.test
```
Here it is how to conduct the same calculations described above for the case of two-sample testing with a one sided alternative
```
power.t.test (power = .90, sig.level=0.05, delta = 0.3, sd=1, alternative = "one.sided")
```
Two-sample t test power calculation
```
n = 190.9879 delta = 0.3 sd = 1 sig.level = 0.05 power = 0.9 alternative = one.sided
```
NOTE: n is number in *each* group
The result provides virtually the same result as the one we reported for an effect size f = 0 . 3 with the same size and power. The R function uses slightly different notation than in the previous sections, where delta stands for the difference in means µ 2 -µ 1 and sd stands for σ . Here sd is actually redundant
```

---

## [10/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 377-377 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.787 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `4b03eef5-409e-4f76-ada5-ef6adc234d0f` |
| **YOUR GRADE** | ____ |

**Full Text (814 chars):**

```
13.3. POWER AND SAMPLE SIZE CALCULATIONS FOR BINARY OUTCOMES: ADDITIVE INTERACTION
Example . Suppose we wish to calculate the power of a test at significance level α = 0.05, with n = 4000, with the prevalence of the genetic and environmental factors being π g = 0.5 and π e = 0.3 respectively and assuming these are independent so that /Delta1 = 1, with the probability of the outcome in the reference category of θ 0 = P ( Y = 1 | G = 0, E = 0) = 0.02, with main effects on the risk difference scale of θ 1 = 0.01 and θ 2 = 0.01 and with additive interaction θ 3 = 0.02. We can use equations (13.1) to calculate π 00 = 0.35, π 10 = 0.35, π 01 = 0.15, π 11 = 0.15, and from this we can calculate L ′ , F ′ , J ′ , R ′ and the variance V and the power Power = /Phi1 -1 { -Z 1 -α/ 2 + η √ ( n / V ) } to obtain 0.32.
```

---

## [11/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 365-365 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.786 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `c60a0c88-0515-4839-9a12-7a4d25a7b460` |
| **YOUR GRADE** | ____ |

**Full Text (1562 chars):**

```
Power and Sample-Size Calculations for Interaction Analysis
In this chapter we will present power and sample-size calculations for interaction analyses. Such calculations are important in the planning of studies. If we desire to have a certain power to be able to detect a particular interaction, we may use such calculations to determine how large the study sample must be in order to do so. In many other cases, a study may have been designed to detect a main effect and the sample size fixed accordingly. With this fixed sample size, we may still be interested in the power that we have in a study to detect an interaction of a certain magnitude. The formulae and tools in this chapter allow for such calculations under a range of scenarios. We will first begin with power and sample size calculations for continuous outcomes. We will then move to the setting of binary outcomes andgive power and sample size calculations for binary outcomes on a multiplicative scale using cohort, case-control, or case-only data. After this we will continue with the setting of binary outcomes but will give power and sample-size calculations for additive interaction which, as we have seen, is often more relevant for evaluating the impact of and deciding between interventions; we will give these power and sample-size calculations for additive interaction for both cohort and case-control data. Finally, we will also present power and sample-size calculations for the mechanistic interaction tests that were considered in Chapters 9 and 10. In all of these cases, we will
```

---

## [12/50] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 558-559 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.786 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `af309134-adcd-47d4-96c7-847e4bff0a22` |
| **YOUR GRADE** | ____ |

**Full Text (1163 chars):**

```
19.2.1 Sample size calculation based on the Normal approximation
Here we consider the one-sided alternative because the math is a little easier to follow, but we will discuss the two-sided alternative, as well. Recall that
<!-- formula-not-decoded -->
Here the 2 in front of σ 2 appears because the variances of Y n and X n add up. The rejection of the null hypothesis happens if Y n -X n > C , where the constant C is determined such that the probability of rejecting the null, if the null is true, is small. This probability is denoted by α and is referred to as the size of the test. Under the null hypothesis, Y n -X n ∼ N (0 , 2 σ 2 /n ) and the constant C can be obtained from the formula
<!-- formula-not-decoded -->
Because √ n/ 2( Y n -X n ) /σ ∼ N (0 , 1) it follows that where z 1 -α denotes the 1 -α quantile of the standard Normal distribution. The idea of the sample size calculation is to find the group sample size, n , that ensures a large probability of rejecting the null when the alternative is true µ 2 > µ 1 . This probability is called the power of the test and is denoted by 1 -β . The rejection probability is
<!-- formula-not-decoded -->
```

---

## [13/50] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 142-142 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.785 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `3c0e14dd-05e5-4cfc-8454-54d5445858b4` |
| **YOUR GRADE** | ____ |

**Full Text (922 chars):**

```
15.5 Student's t-Test Power Analysis
```
# estimate sample size via power analysis from statsmodels.stats.power import TTestIndPower # parameters for power analysis effect = 0.8 alpha = 0.05 power = 0.8 # perform power analysis analysis = TTestIndPower() result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha) print( ' Sample Size: %.3f ' % result)
```

Listing 15.2: Example of calculating sample size.
Running the example calculates and prints the estimated number of samples for the experiment as 25. This would be a suggested minimum number of samples required to see an effect of the desired size.
```
Sample Size: 25.525
```

Listing 15.3: Sample output from calculating sample size.
We can go one step further and calculate power curves. Power curves are line plots that show how the change in variables, such as effect size and sample size, impact the power of the statistical test.
```

---

## [14/50] google_ds_l4_l5_guide

| Field | Value |
|-------|-------|
| **Pages** | 395-395 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.785 |
| **Found In** | vector |
| **Chunk ID** | `6dc3695a-20a2-46a6-a6de-40f4619e0ef6` |
| **YOUR GRADE** | ____ |

**Full Text (162 chars):**

```
Sample Size Calculation :
# For β =0.2 (power=0.8) n = 2 * (z_alpha + z_beta)**2 * p_pooled * (1 - p_pooled) / delta**2 print(f"Need {n:.0f} users per group")
```
```

---

## [15/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 372-372 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.784 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `6c6eeb61-0870-4d14-9a94-54910a7abaf2` |
| **YOUR GRADE** | ____ |

**Full Text (797 chars):**

```
13.2.2. Multiplicative Interaction with Case-Control Data
If instead of calculating the required sample size for a fixed power β , we wanted to calculate the power for a given sample size using the Wald test for the null hypothesis γ 3 = 0basedonthelogistic regression model, we could proceed as follows. For a fixed sample size n , the power to reject the null γ 3 = 0at significance level α under the alternative that γ 3 = η is given by where /Phi1 -1 is the inverse cumulative distribution function for a standard normal random variable and where V ∗ mult ( OR ) can be calculated as above. If the null hypothesis were rejected for extreme values of γ 3 on either side of zero (two-sided test), then the relevant power formula would be
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
```

---

## [16/50] ROS

| Field | Value |
|-------|-------|
| **Pages** | 308-308 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.782 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `22cfeaef-7f41-431d-8180-8290e62d20a2` |
| **YOUR GRADE** | ____ |

**Full Text (848 chars):**

```
Sample size to achieve a specified probability of obtaining statistical significance
The conventional level of power in sample size calculations is 80%: the goal is to choose n such that 80% of the possible 95% confidence intervals will not include 0.5. When n is increased, the estimate becomes closer (on average) to the true value, and the width of the confidence interval decreases. Both these effects (decreasing variability of the estimator and narrowing of the confidence interval) can be seen in going from the top half to the bottom half of Figure 16.2.
To find the value of n such that exactly 80% of the estimates will be at least 1.96 standard errors from 0.5, we need
<!-- formula-not-decoded -->
Some algebra then yields ( 1 : 96 + 0 : 84 ) s.e. = 0 : 1. We can then substitute s.e. = 0 : 5 = p n and solve for n , as we discuss next.
```

---

## [17/50] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 560-560 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.781 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `dae60ee4-e706-46af-aa5b-dcc9f0a1b990` |
| **YOUR GRADE** | ____ |

**Full Text (640 chars):**

```
19.2.1 Sample size calculation based on the Normal approximation
```
#Size alpha=0.05 #One minus power beta=0.1 #Corresponding quantiles of the standard N(0,1) z_1_minus_alpha= qnorm (1 -alpha) z_1_minus_beta= qnorm (1 -beta) #Calculate the multiplier in front of the fraction multiplier<-4 * (z_1_minus_alpha + z_1_minus_beta) ^ 2
```
This indicates that the constant 4( z 1 -α + z 1 -β ) 2 is equal to 34.255 for these choices of size and power of the test. The other factor that the sample size depends on is the effect size
<!-- formula-not-decoded -->
Thus, the sample size can be written more compactly as
<!-- formula-not-decoded -->
```

---

## [18/50] David G Kleinbaum Lawrence L Kupper Azhar Nizam Eli S Rosenberg - Applied Regression Analysis and Other Multivariable Methods-Cengage Learning 2013

| Field | Value |
|-------|-------|
| **Pages** | 933-933 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.780 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `46c45c79-3fb6-4b86-b412-dfb63307ba6e` |
| **YOUR GRADE** | ____ |

**Full Text (1531 chars):**

```
27.7  Practical Considerations and Cautions
Another issue of practical importance concerns the increasing availability of software for power and sample size calculations, including SAS and PASS 11 but also including many other shareware or freeware packages and web applets. Many of these programs have easyto-use 'point-and-click' interfaces and thus put rigorous sample size planning tools within the reach of more analysts. At the same time, there is a danger that the applications will be employed inappropriately by users who are not adequately aware of details about the underlying models and the methods of calculation associated with the software. As a further complication, the levels of documentation differ greatly for different software packages and applets. Finally, not all software applications are equally accurate, as some will rely on approximations more than others (Hsieh et al. 1998; Thomas and Krebs 1997). Indeed, the simulation studies of Hsieh et al. showed that results from the simple formulas discussed in Section 27.2 were, in some cases, more accurate than results produced by certain software packages. We think it is essential that users choose well-documented software, in which all the mathematical details of the power and sample size calculations are given; moreover, we encourage users to read and understand the software documentation so that they are well aware of the methodology being implemented, the correct technique for using the software, and any potential limitations of the software.
```

---

## [19/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 375-376 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.780 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `269726d0-9b8b-4182-b109-b080699fcd4e` |
| **YOUR GRADE** | ____ |

**Full Text (861 chars):**

```
13.3. POWER AND SAMPLE SIZE CALCULATIONS FOR BINARY OUTCOMES: ADDITIVE INTERACTION
where Z 1 -α/ 2 and Z β are the (1 -α/ 2)th and β th quantiles, respectively, of the standard normal distribution and where V is the variance of ̂ θ 3 under the alternative that θ 3 = η . The additional computational burden lies in calculating the variance V . This variance V is given by VanderWeele (2012c):
<!-- formula-not-decoded -->
where
<!-- formula-not-decoded -->
Thus to calculate the sample size, we would need to specify (i) the significance level α , the power β , and the magnitude of additive interaction θ 3 = η ; (ii) the proportion of subjects in each exposure stratum, π 00, π 10, π 01, π 11; and (iii) the main effect of the two exposures on the additive scale θ 1 and θ 2 and the baseline risk of the doubly unexposed group θ 0 = P ( Y = 1 | G = 0, E = 0).
```

---

## [20/50] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 547-549 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.779 |
| **Found In** | vector |
| **Chunk ID** | `3c66e4b0-7a12-48b5-838e-dfbbac4a02ff` |
| **YOUR GRADE** | ____ |

**Full Text (1102 chars):**

```
18.2 Power calculation for Normal tests
Consider our previous example from Chapter 16 involving RDI. We were testing H 0 : µ = 30 versus H a : µ > 30 where µ was the population mean respiratory
Figure 18.3: Power to reject the null hypothesis as a function of the number of trials, n , when the true success probability is p = 0 . 7 .
disturbance index. Our test statistic was
<!-- formula-not-decoded -->
and we reject if Z ≥ Z 1 -α . Assume that n is large and that Z is well approximated by a Normal distribution and σ is known. Let µ a be a value of µ under H a that we are assuming to be the true value for the calculations. Then consider:
<!-- formula-not-decoded -->
Thus, we can relate our power to a standard Normal again. Suppose that we wanted to detect an increase in mean RDI of at least 2 events / hour (above 30 ). Assume normality and that the sample in question will have a standard
deviation of 4 . What would the power be if we took a sample size of 16 ? We have that Z α = 1 . 645 and µ a -30 σ/ √ n = 2 / (4 / √ 16) = 2 and therefore P ( Z > 1 . 645 -2) = P ( Z > -0 . 355) = 64% .
```

---

## [21/50] (Quantitative Methodology) Joop J. Hox, Mirjam Moerbeek, Rens Van De Schoot - Multilevel Analysis  Techniques and Applications-Routledge (2017)

| Field | Value |
|-------|-------|
| **Pages** | 237-238 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.778 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `48221c86-e9f7-4d5b-94ce-dc0cdd694067` |
| **YOUR GRADE** | ____ |

**Full Text (1066 chars):**

```
12.2 PoWeR AnALySiS
In this example the sample size to achieve a desired power level was calculated. This should  be  done  during  the  planning  phase  of  a  study  to  get  insight  into  the  number  of subjects that need to be recruited. Such a power analysis is called an a priori power analysis. In many trials the maximum number of subjects is fixed beforehand, for instance because of financial constraints, and the power of the test can be calculated. Low sample sizes often result in low power levels and a decision must be made to search for additional funding such that a sufficient level of power can be achieved, to conduct the study and accept low chances
Figure 12.1 Significance and power in the Z-test.
of finding effects, or not to conduct the study at all. Whenever one wants to calculate the required sample for a given power level or the power for a given sample size, it is necessary to obtain an educated guess based on expert knowledge or findings from the literature to get a plausible estimate of the population value of the effect size.
```

---

## [22/50] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 558-558 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.778 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `a7e007f5-b9d6-4c9e-8594-a28319aa1e65` |
| **YOUR GRADE** | ____ |

**Full Text (854 chars):**

```
19.2 Sample size calculation for continuous data
Wehave already seen how to obtain confidence intervals and conduct tests for the mean of two populations. The sample size calculation is turning the hypothesis testing problem around and asking the question: 'given an effect size (not yet defined) what is the sample size that will be detected with high probability (power).' Consider the case when we observe n subjects both in the first and second group for a total sample size of 2 n . Denote by X 1 , . . . , X n ∼ N ( µ 1 , σ 2 ) and Y 1 , . . . , Y n ∼ N ( µ 2 , σ 2 ) the outcomes of the experiment in the first and second group, respectively. We consider the case of equal and known variances in the two groups, but similar approaches can be used for unequal variances. We are interested in testing the null hypothesis
<!-- formula-not-decoded -->
```

---

## [23/50] google_ds_l4_l5_guide

| Field | Value |
|-------|-------|
| **Pages** | 395-395 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.775 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `dfbb6295-77a1-4b6c-905b-bef32f7a6118` |
| **YOUR GRADE** | ____ |

**Full Text (693 chars):**

```
Sample Size Calculation :
from statsmodels.stats.power import zt_ind_solve_power # Sample size for two-sample t-test effect_size = 0.2  # Cohen's d (small=0.2, medium=0.5, large=0.8) alpha = 0.05 power = 0.8 n_per_group = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power) print(f"Need {n_per_group:.0f} users per group") # Sample size for conversion rate # Baseline conversion: p1 = 0.05, target conversion: p2 = 0.055 (10% lift) from statsmodels.stats.proportion import proportions_ztest # Approximate: n ≈ 2 * (Z_ α /2 + Z_ β ) ² * p(1-p) / (p2 - p1) ² p1 = 0.05 p2 = 0.055 p_pooled = (p1 + p2) / 2 delta = p2 - p1 z_alpha = 1.96  # For α =0.05 (two-tailed) z_beta = 0.84
```

---

## [24/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 378-378 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.770 |
| **Found In** | fts, citations |
| **Chunk ID** | `66602117-d632-440a-a3c0-cded10401fb1` |
| **YOUR GRADE** | ____ |

**Full Text (637 chars):**

```
13.3.2. Additive Interaction in Cohort Studies Using Logistic Regression and RERI
If instead of calculating the required sample size for a given power, we wanted to calculate the power for a given sample-size we could use Power = /Phi1 -1 { -Z 1 -α/ 2 + η √ ( n / VRERI ( OR ) ) } or, for a two-sided test, to detect either positive or negative additive interaction we could use Power = /Phi1 -1 { -Z 1 -α/ 2 + η √ ( n / VRERI ( OR ) ) } + /Phi1 -1 { -Z 1 -α/ 2 -η √ ( n / VRERI ( OR ) ) } . Again, in Section 13.5 we will describe how to use a simple Excel spreadsheet to carry out such sample-size and power calculations automatically.
```

---

## [25/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 368-369 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.769 |
| **Found In** | fts, vector |
| **Chunk ID** | `90345543-344e-469f-9b5d-759f12e4a39c` |
| **YOUR GRADE** | ____ |

**Full Text (1133 chars):**

```
13.2.1. Multiplicative Interaction with Cohort Data
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
If G and E are independent, then /Delta1 = 1 and C simplifies to C = π e / (1 -π e ).
If instead of calculating the required sample size for a fixed power β , we wanted to calculate the power for a given sample size using the Wald test for the null hypothesis γ 3 = 0basedonthelogistic regression model, we could proceed as follows. For a fixed sample size n the power to reject the null γ 3 = 0 at significance level α under
the alternative that γ 3 = η is given by
<!-- formula-not-decoded -->
where /Phi1 -1 is the inverse cumulative distribution function for a standard normal random variable and where Vmult ( OR ) can be calculated as above. In Section 13.5 wewill describe how to use a simple Excel spreadsheet to carry out such sample size and power calculations automatically. 1
Finally, it should be noted that if the null hypothesis were rejected for extreme values of γ 3 on either side of zero (two-sided test), then the relevant power formula would be
<!-- formula-not-decoded -->
```

---

## [26/50] Wiley Series in Probability and Statistics John M Lachin - Biostatistical Methods: The Assessment of Relative Risks 2010 Wiley - libgenlc

| Field | Value |
|-------|-------|
| **Pages** | 129-130 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.768 |
| **Found In** | vector |
| **Chunk ID** | `da09a93f-60d1-4acb-a07e-c94f55d93423` |
| **YOUR GRADE** | ____ |

**Full Text (701 chars):**

```
proc power;
```
twosamplewilcoxon alpha = 0.05 power = 0.9 NTOTAL = . vardist("groupl") = ordinal ((1 2 3) : (0.85 0.10 0.05)) vardist("group2") = ordinal ((1 2 3) : (0.90 0.075 0.025)) variables = "groups" I "group2";
```
The program provides N = 1754 that is negligibly different from the value 1757 provided by the noncentral distribution of the test statistic.
Alternatively, the following statements would provide the computation of power of the test for other sample sizes:
proc power; twosamplewilcoxon alpha = 0.05 NTOTAL = 500 600 1700 power = . vardist..... (etc .) ;
For a sample size of 600, the estimated power is 0.475, equivalent to that computed above from the noncentral distribution.
```

---

## [27/50] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 137-137 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.766 |
| **Found In** | vector |
| **Chunk ID** | `8fab10e6-6f9f-4277-8ec2-99104ceea6ba` |
| **YOUR GRADE** | ____ |

**Full Text (112 chars):**

```
14.7.1 Next
In the next section, you will discover statistical power and how to use it to estimate sample sizes.
```

---

## [28/50] Vol 2: Causal Inference

| Field | Value |
|-------|-------|
| **Pages** | 141-141 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.765 |
| **Found In** | vector |
| **Chunk ID** | `5616ce8d-9f20-4fc7-af46-4317bee75c79` |
| **YOUR GRADE** | ____ |

**Full Text (156 chars):**

```
Problem 7.5 [CI-7.1-calc]
Calculate the sample size per group needed to detect a 2pp lift from a 10% baseline conversion rate with 80% power at α = 0 . 05 .
```

---

## [29/50] Statistical Power Analysis for the Behavioral Sciences

| Field | Value |
|-------|-------|
| **Pages** | 184-184 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.763 |
| **Found In** | vector |
| **Chunk ID** | `2c59e079-3236-4f1b-8b87-3b11eaba42ac` |
| **YOUR GRADE** | ____ |

**Full Text (951 chars):**

```
5.4 SAMPLE SIZE TABLES
$$Power: K: .50 0 .60 0.5 i .70 1.0 1.5 .80 2.5 .85 .90 3.0 3.5 .95 .99 6.0 9.0$$
3.  Desired  Power. As  in  the  previous  chapters,  provision  is  made  for desired  power  values  of .25,  .50,  .60, f, .70  (.05),  .95,  .99.  For  discussion of the  basis  for  selecting  these  values,  the  provision  for  equalizing a  and b risks,  and  the  rationale  of a  proposed  convention  of desired  power  of .80, see Section 2.4.
Summarizing  the  use  of the  following n tables,  the  investigator  finds (a) the table for the significance criterion (a) he is  using, locates (b) the popu­ lation (alternate-hypothetical) value of g and (c) the desired power along the vertical stub. He then  finds n, the necessary sample size to detect g at (when n <50, no  more  than)  the a significance  criterion  with  the  desired  power. If the g value in  his  specifications  is  not provided,  he  locates  the  value  for
```

---

## [30/50] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 142-142 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.763 |
| **Found In** | vector |
| **Chunk ID** | `83fe0cd7-d69a-499c-98da-7c4f24dc89e4` |
| **YOUR GRADE** | ____ |

**Full Text (770 chars):**

```
15.5 Student's t-Test Power Analysis
A note on sample size: the function has an argument called ratio that is the ratio of the number of samples in one sample to the other. If both samples are expected to have the same number of observations, then the ratio is 1.0. If, for example, the second sample is expected to have half as many observations, then the ratio would be 0.5. The TTestIndPower instance must be created, then we can call the solve power() with our arguments to estimate the sample size for the experiment.
```
... # perform power analysis analysis = TTestIndPower() result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
```

Listing 15.1: Function for calculating statistical power.
The complete example is listed below.
```

---

## [31/50] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 573-573 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.761 |
| **Found In** | fts |
| **Chunk ID** | `0947d63d-5c27-4876-a78d-c3b78e12ca35` |
| **YOUR GRADE** | ____ |

**Full Text (688 chars):**

```
19.4 Sample size calculations using exact tests
```
#Calculate the first time the power exceeds 0.9 exceed_diff<-power_binom > 0.9 #Get the sample size and corresponding critical value sample_size_finite<-n[ which.max (exceed_diff)] critical_value_finite<-sn[ which.max (exceed_diff)] #Calculate the proportion difference between exact and asymptotic tests ratio_exact_asymptotic<-round (100 * (sample_size_finite -ss_o_c) /
```
ss_o_c,digits=2)
Thus, the sample size is calculated as the first time the power exceeds 0 . 9 and that happens for 28 at a critical value of 20. Thus, the exact test would require 47.37% more subjects than the asymptotic test. The sample size is still small,
```

---

## [32/50] Statistical Power Analysis for the Behavioral Sciences

| Field | Value |
|-------|-------|
| **Pages** | 469-470 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.760 |
| **Found In** | vector |
| **Chunk ID** | `f271eefb-6625-4f01-b6c6-4680194a616c` |
| **YOUR GRADE** | ____ |

**Full Text (1189 chars):**

```
Illustrative Examples
which,  for these data and subjective Pis,  yields N = .3(248) + .5(120) + .2(78) = 150.  A similar procedure may be used  in  subjectively weighting power values as a function of a fixed N over a range of ESs into a single esti-
mate of power. Of course, these Bayesian-like procedures may be used for any statistical test, not just those of MRC.
Finally, however one proceeds in the end, the generation of tables of N such as the above is recommended for coping with the problem of making decisions about N in experimental planning.
- 9.20 In example 9.5, a Case 0 MRC power analysis was performed on a teaching methods problem, previously analyzed by the analysis of variance methods of Chapter 8 in example 8.1. The original  specifications yielded power =  .51  (.52 in example 8.1) for four groups (hence u  =  3) of 20 cases each (hence N =  80), at a = .OS with f =  .28. When redone as an analysis of variance problem in determining the sample size necessary for power  =  .80 (example 8.10), it was found that then per group would be 35.9 ( =  36), soN =  143.6 ( =  144). Redoing this now as a Case 0 MRC problem in determin­ ing N, the specifications are:
```

---

## [33/50] Statistical Power Analysis for the Behavioral Sciences

| Field | Value |
|-------|-------|
| **Pages** | 283-284 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.760 |
| **Found In** | vector |
| **Chunk ID** | `6562bebf-f603-4023-bc4e-976d3a2cb637` |
| **YOUR GRADE** | ____ |

**Full Text (931 chars):**

```
Illustrative Examples
Since w =  .289  is not  tabled,  the  use  of  formula  (7  .4.1)  is required. For N.10, the  sample  size  needed  to  detect  w  =  .10  with  power= .80  for a= .05  and  u =  3,  we  use  the  third  subtable  of Table  7.4.6  (for  a= .05, u  =  3)  for  column  w  =  .10  and  row  power= .80,  and  find N .to=  1090. Substituting in formula (7.4.1),
<!-- formula-not-decoded -->
Thus,  131  repondents  will  lead  to a  .80  probability  of rejecting  the null hypothesis of  equal preference at a  =  .05, given that the population departure is  indexed by w = .289.
7.4.2 CASE  1: CoNTINGENCY TEST.  As in Case 0, one finds the necessary total  sample  size  N  in  Case  1  by  finding  the  subtable  for  the  significance criterion (a) and degrees of freedom [u = (k- I)(r- I)] which obtain, and seeking  w  and  the  power  desired.  Formula (7.4.1)  is  again  used  for  non­ tabulated w.
```

---

## [34/50] Statistical Power Analysis for the Behavioral Sciences

| Field | Value |
|-------|-------|
| **Pages** | 530-530 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.757 |
| **Found In** | vector |
| **Chunk ID** | `8137cac8-9071-475f-a51f-cc84ab43df66` |
| **YOUR GRADE** | ____ |

**Full Text (850 chars):**

```
Illustrative Examples
- 10.18 In example 10.3, our epidemiologist changed her plans. Retain­ ing the planned N of 100, she planned to reduce kv to 3 and kx to 2, in order to reduce s and thereby increase 1 2  from .0660 to .1180, and also to reduce u from 48 to 6. Thus revised, she found power to be .88. She wondered what sample size would be needed to increase power to .90 for the revised specifi­ cations:
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
From Table 9.4.1, for u  =  6 at power =  .90, for trial v =  120, A for V =  oo, A =  23.2. From (10.4.1), the implied v =  203. Interpolating with (10.4.2), A =  24.1  which, when entered into (10.4.1) yields the iterated v  = 198, which, entered into (10.4.3) with the other parameters, yields N =  104. a slight increase over the 100 she was provisionally planning.
= 24.8, and
```

---

## [35/50] Statistical Power Analysis for the Behavioral Sciences

| Field | Value |
|-------|-------|
| **Pages** | 407-408 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.756 |
| **Found In** | vector |
| **Chunk ID** | `058d77e8-ecf7-4699-b100-77cb0b67d4ae` |
| **YOUR GRADE** | ____ |

**Full Text (1212 chars):**

```
Illustrative Examples
- 8.12 To  illustrate  Case I in surveys  of  natural  populations,  return to example 8.3,  where  a  political  science class designs an  opinion  survey  of college  students  on  government  centralism. A source  of  variance  to  be studied  is  the  academic  areas  of  respondents  of which  there  are  6  ( =  k). The f for  the  anticipated  unequal n 1  is  posited  at  .15,  and  a= .05.  Now, instead  of treating  this  as  a  completed  or  committed  experiment  (where total N  was  set at 300  and  power  then  found  to  be  .48),  let  us  ask  what N is  required to attain  power of .80.  The specifications are
<!-- formula-not-decoded -->
In  the first  subtable  of Table  8.4.5 (for a= .05, u =  5)  at column f =  15 and  row  power= .80,  n = 96.  This  is  the  average  size  necessary  for  the  6 academic area samples.  The quantity we  need  is  the  total  sample size, N  = 6(96) = 576.
Example 8.3  went on  to consider the  effect  on  power of a  reduction  of k  from  6  to  3  more  broadly  defined  academic  areas.  Paralleling  this,  we
determine ~ needed for k = 3,  keeping the other specifications unchanged:
<!-- formula-not-decoded -->
```

---

## [36/50] Statistical Power Analysis for the Behavioral Sciences

| Field | Value |
|-------|-------|
| **Pages** | 73-73 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.756 |
| **Found In** | vector |
| **Chunk ID** | `0314cb89-47e0-43fc-8cc6-0e369fd2a656` |
| **YOUR GRADE** | ____ |

**Full Text (1389 chars):**

```
2.4 SAMPLE  SIZE  TABLES
It is  proposed  here  as  a  convention  that,  when  the  investigator  has  no other basis  for  setting  the  desired  power  value,  the  value  .80  be  used.  This means that b  is  set  at  .20.  This  arbitrary  but  reasonable  value  is  offered  for several  reasons (Cohen,  1965,  pp.  98-99).  The chief among them  takes into consideration  the  implicit  convention  for  a  of .05.  The  b  of .20  is  chosen with  the  idea  that  the  general  relative  seriousness  of these  two  kinds  of errors is  of the  order of .20/.05,  i.e.,  that  Type  I  errors  are  of the  order of four  times  as  serious  as  Type  II  errors.  This  .80  desired  power convention is  offered  with  the  hope  that  it  will  be  ignored  whenever  an  investigator can find  a  basis  in  his  substantive  concerns  in  his  specific  research  investi­ gation to choose a value ad hoc.
Returning to the Case 0 use  of the n tables and summarizing, the investi­ gator  finds  (a)  the  table  for  the  significance  criterion  (a)  he  is  using,  and looks  for  (b)  the  standardized  difference  between  the  population  means (d)  along  the  horizontal  stub  and  (c)  the  desired  power  along  the  vertical stub. These determine n,  the  necessary size  of each sample to detect d  at the a  significance criterion with  the desired  power.
```

---

## [37/50] Vol 2: Causal Inference

| Field | Value |
|-------|-------|
| **Pages** | 104-104 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.755 |
| **Found In** | vector |
| **Chunk ID** | `7a39e3f8-9f99-439a-a72a-409d491d6e99` |
| **YOUR GRADE** | ____ |

**Full Text (878 chars):**

```
Solution:
Approach : State the precise probabilistic definition, then provide an intuitive restatement.
One-sentence definition : Statistical power is the probability of correctly rejecting the null hypothesis when the alternative hypothesis is true, i.e., Power = P ( reject H 0 | H 1 true ) = 1 -β .
Intuitive restatement : Power is the probability of detecting a true effect-if an effect exists, power tells you how likely your study is to find it.
Key Insight : Low power means you might miss real effects (high Type II error). The standard threshold is 80%, meaning you want at least an 80% chance of detecting the effect if it exists.
Common Pitfalls : (1) Confusing power with α (significance level). (2) Forgetting that power depends on effect size-larger effects are easier to detect. (3) Computing power after the study is not meaningful (post-hoc power is a fallacy).
```

---

## [38/50] (Quantitative Methodology) Joop J. Hox, Mirjam Moerbeek, Rens Van De Schoot - Multilevel Analysis  Techniques and Applications-Routledge (2017)

| Field | Value |
|-------|-------|
| **Pages** | 248-248 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.753 |
| **Found In** | vector |
| **Chunk ID** | `51a119be-108a-4063-8095-0a982557d199` |
| **YOUR GRADE** | ____ |

**Full Text (887 chars):**

```
12.5 MeTHoDS foR MeTA-AnALySiS
size of γ = 0.01, with an associated standard error of 0.01 (the value of the standard error for N tot in Table 8.4), and the significance level set at α = 0.05. We again use Equation 12.3: (effect size) / (standard error) ≈ ( Z 1-α + Z 1-β ), which in this case becomes (0.01) / (0.01) = (1.64 + Z 1-β ). So, Z --β = 1 - 1.64 = -0.64. This leads to a post-hoc power estimate of 0.74, which appears adequate. The failure to find a significant effect for the study sample size is not likely to be the result of insufficient power of the statistical test.
Post-hoc power analysis is not only useful in evaluating one's own analysis, as just shown, but also in the planning stages of a new study. By investigating the power of earlier studies, we find which effect sizes and intraclass correlations we may expect, which should help us to design our own study.
```

---

## [39/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 624-625 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.751 |
| **Found In** | vector |
| **Chunk ID** | `a5f43848-3ba3-4843-901b-2d62e9dd01b2` |
| **YOUR GRADE** | ____ |

**Full Text (772 chars):**

```
A.13.3. Derivations for Case-Control Exposure Probabilities from the Probabilities in the Underlying Population
Suppose we wish to use a Wald test for the null hypothesis RERI = 0. The sample size required to detect a RERI of magnitude η = e κ 1 + κ 2 + κ 3 -e κ 1 -e κ 2 + 1 with significance level α and power β is
<!-- formula-not-decoded -->
where Z 1 -α/ 2 and Z β are the (1 -α/ 2)th and β th quantiles, respectively, of the standard normal distribution and where VRERI ( RR ) is the variance of RERI = e ̂ κ 1 + ̂ κ 2 + ̂ κ 3 -e ̂ κ 1 -e ̂ κ 2 + 1 under the alternative. Likewise, to calculate the power for a given sample size we could use
<!-- formula-not-decoded -->
Using an argument analogous to that in Section A.13.6 we have that
<!-- formula-not-decoded -->
```

---

## [40/50] R_in_Action,_Second_Edition (5)

| Field | Value |
|-------|-------|
| **Pages** | 282-282 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.750 |
| **Found In** | fts |
| **Chunk ID** | `12d8dafd-0a01-470e-9fb5-3bbc688dbad4` |
| **YOUR GRADE** | ____ |

**Full Text (1302 chars):**

```
10.4 Other packages
Table 10.4 Specialized power-analysis packages

asypow, Purpose = Power calculations via asymptotic likelihood ratio methods. longpower, Purpose = Sample-size calculations for longitudinal data. PwrGSD, Purpose = Power analysis for group sequential designs. pamm, Purpose = Power analysis for random effects in mixed models. powerSurvEpi, Purpose = Power and sample-size calculations for survival analysis in epidemio- logical studies. powerMediation, Purpose = Power and sample-size calculations for mediation effects in linear, logistic, Poisson, and cox regression. powerpkg, Purpose = Power analyses for the affected sib pair and the TDT (transmission disequilibrium test) design. powerGWASinteraction, Purpose = Power calculations for interactions for GWAS. pedantics, Purpose = Functions to facilitate power analyses for genetic studies of natural populations. gap, Purpose = Functions for power and sample-size calculations in case-cohort designs. ssize.fdr, Purpose = Sample-size calculations for microarray experiments
Finally, the MBESS package contains a wide range of functions that can be used for various forms of power analysis and sample size determination. The functions are particularly relevant for researchers in the behavioral, educational, and social sciences.
```

---

## [41/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 223-223 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.747 |
| **Found In** | fts, vector |
| **Chunk ID** | `a35487bc-f385-429a-93ca-8aa6eac6ef39` |
| **YOUR GRADE** | ____ |

**Full Text (1462 chars):**

```
7.9. POWER AND SAMPLE SIZE CALCULATIONS FOR MEDIATION ANALYSIS
One issue we have not discussed which is relevant to mediation, especially in designing a study, is that of power and sample size calculations for direct and indirect effects. Unfortunately, the current literature on this topic is somewhat limited and further development is still needed. Fritz and MacKinnon (2007) present some basic power and sample size requirements using simulations corresponding to small-, medium-, and large-sized effects for the exposure on the mediator and the mediator on the outcome. However, these do not allow an investigator to precisely calculate power or sample size when specifying exact effect sizes other than the scenarios they give. Kenny and Judd (2014) show that in many settings, power to detect indirect effects is greater than that for total effects, and power to detect direct effects is less than that for total effects. Vittinghoff et al. (2009) present a variety of power- and sample-size formulae when one is willing to assume that the exposure has an effect on the mediator (so that all that needs to be tested is whether the mediator has an effect on the outcome), but this approach presupposes part of the hypothesis that is to be tested in mediation (that the exposure affects the mediator). Freedman et al. (1992) and Freedman and Schatzkin (1992) present power andsamplesize formulae for the proportion mediated measure discussed in Section 2.13
```

---

## [42/50] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 549-549 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.747 |
| **Found In** | vector |
| **Chunk ID** | `050bb6fa-2f3a-452a-b9ef-f26afb65ba9b` |
| **YOUR GRADE** | ____ |

**Full Text (356 chars):**

```
[1] 0.639
Consider now a sample size calculation. What value of n would yield 80 %power? That is, we want
<!-- formula-not-decoded -->
Therefore, we want to set z 1 -α -µ a -30 σ/ √ n = z 0 . 20 and solve for n yielding
<!-- formula-not-decoded -->
```
mu0 = 30; mua = 32; sigma = 4 ceiling (sigma ^ 2 * ( qnorm (.95) -qnorm (.2)) ^ 2 / (mua -mu0) ^ 2)
```
```

---

## [43/50] google_ds_l4_l5_guide

| Field | Value |
|-------|-------|
| **Pages** | 119-120 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.745 |
| **Found In** | fts |
| **Chunk ID** | `06ffb203-973b-4b1e-9041-aac46600a89b` |
| **YOUR GRADE** | ____ |

**Full Text (495 chars):**

```
Python Solution:
{results['estimated_days']} days (assuming 1000 users/day)")
```
```
# Sensitivity analysis: What if we want 90% power? results_90 = calculate_ab_test_sample_size(baseline_rate=0.10, target_rate=0.12, power=0.90) print(f"\n90% power requires: {results_90['sample_size_per_group']:,} per group ( {results_90['total_sample_size']:,} total)") print(f"Increase: {(results_90['sample_size_per_group'] - results[ 'sample_size_per_group']) / results['sample_size_per_group']:.1%}")
```
```

---

## [44/50] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 143-144 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.740 |
| **Found In** | fts |
| **Chunk ID** | `7af8a89e-4b45-4516-bb68-0bba1ee0ec2b` |
| **YOUR GRADE** | ____ |

**Full Text (1071 chars):**

```
15.5 Student's t-Test Power Analysis
```
# calculate power curves for varying sample and effect size from numpy import array from matplotlib import pyplot from statsmodels.stats.power import TTestIndPower # parameters for power analysis effect_sizes = array([0.2, 0.5, 0.8]) sample_sizes = array(range(5, 100)) # calculate power curves from multiple power analyses analysis = TTestIndPower() analysis.plot_power(dep_var= ' nobs ' , nobs=sample_sizes, effect_size=effect_sizes) pyplot.show()
```

Listing 15.5: Example of calculating a power analysis.
Running the example creates the plot showing the impact on statistical power (y-axis) for three different effect sizes (es) as the sample size (x-axis) is increased. We can see that if we are interested in a large effect that a point of diminishing returns in terms of statistical power occurs at around 40-to-50 observations.
Figure 15.1: Power curves for Student's t-test.
Usefully, Statsmodels has classes to perform a power analysis with other statistical tests, such as the F-test, Z-test, and the Chi-Squared test.
```

---

## [45/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 387-387 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.739 |
| **Found In** | fts |
| **Chunk ID** | `87226744-3cb6-46c7-835a-5c2be522dd23` |
| **YOUR GRADE** | ____ |

**Full Text (1321 chars):**

```
13.6. DISCUSSION
In this chapter we have given sample-size and power formulae for additive and multiplicative interaction in a variety of scenarios. We saw that when the main effects were both positive, then the power to detect positive interaction on the additive scale was in general greater than on the multiplicative scale. We have also discussed how the sample-size and power calculations for the relative excess risk due to interaction can be easily modified to provide sample-size and power calculations for mechanistic interaction corresponding to notions of synergism in the sufficient cause framework and to notions of compositional epistasis in genetics.
We have focused here on cohort, case-control, and case-only data; but as discussed in the Chapter 12, other study designs, such as matched case-control studies (cf. Gauderman, 2002a,b), and other methods for testing such as the joint tests of Chapter 12, are also sometimes used. Software is also available to implement power and sample-size calculations for a number of these other settings. Windows-based QUANTO,developedbyGauderman, is available at http://hydra.usc.edu/gxe and will implement sample-size calculations for likelihood ratio-based tests of interaction using various study designs, and the reader is referred there for further information.
```

---

## [46/50] R_in_Action,_Second_Edition (5)

| Field | Value |
|-------|-------|
| **Pages** | 271-271 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.726 |
| **Found In** | fts |
| **Chunk ID** | `af31822b-98b3-4401-9796-d10de4beee74` |
| **YOUR GRADE** | ____ |

**Full Text (1006 chars):**

```
10.2 Implementing power analysis with the pwr package
The pwr package,  developed  by  Stéphane  Champely,  implements power analysis as outlined by Cohen (1988). Some of the more important functions are listed in table 10.1. For each function, you can specify three of the four quantities (sample size, significance level, power, effect size), and the fourth will be calculated.
Table 10.1 pwr package functions

pwr.2p.test, Power calculations for… = Two proportions (equal n). pwr.2p2n.test, Power calculations for… = Two proportions (unequal n). pwr.anova.test, Power calculations for… = Balanced one-way ANOVA. pwr.chisq.test, Power calculations for… = Chi-square test. pwr.f2.test, Power calculations for… = General linear model. pwr.p.test, Power calculations for… = Proportion (one sample). pwr.r.test, Power calculations for… = Correlation. pwr.t.test, Power calculations for… = t-tests (one sample, two samples, paired). pwr.t2n.test, Power calculations for… = t-test (two samples with unequal n)
```

---

## [47/50] google_ds_l4_l5_guide

| Field | Value |
|-------|-------|
| **Pages** | 119-119 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.717 |
| **Found In** | fts |
| **Chunk ID** | `03c5d175-04ed-46cf-8887-9645c39632b0` |
| **YOUR GRADE** | ____ |

**Full Text (813 chars):**

```
Python Solution:
# Assumption days_to_run = total_sample_size / users_per_day return { 'sample_size_per_group': sample_size_per_group, 'total_sample_size': int(total_sample_size), 'baseline_rate': baseline_rate, 'target_rate': target_rate, 'absolute_lift': round(absolute_lift, 4), 'relative_lift': round(relative_lift, 4), 'effect_size_cohens_h': round(effect_size, 3), 'alpha': alpha, 'power': power, 'estimated_days': int(np.ceil(days_to_run)) } # Test: 10% → 12% conversion rate results = calculate_ab_test_sample_size( baseline_rate=0.10, target_rate=0.12, alpha=0.05, power=0.80 ) print(results) print(f"\nRequired sample size: {results['sample_size_per_group']:,} per group ( {results['total_sample_size']:,} total)") print(f"Relative lift: {results['relative_lift']:.1%}") print(f"Estimated test duration:
```

---

## [48/50] google_ds_l4_l5_guide

| Field | Value |
|-------|-------|
| **Pages** | 119-119 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.697 |
| **Found In** | fts |
| **Chunk ID** | `4a46677f-a34f-4797-8055-b952114dd2c2` |
| **YOUR GRADE** | ____ |

**Full Text (623 chars):**

```
Python Solution:
# Use two-sided test (alternative='two-sided') sample_size_per_group = zt_ind_solve_power( effect_size=effect_size, alpha=alpha, power=power, ratio=ratio, alternative='two-sided' ) # Round up (can't have fractional users) sample_size_per_group = int(np.ceil(sample_size_per_group)) # Step 3: Calculate total sample size total_sample_size = sample_size_per_group * (1 + ratio) # Step 4: Calculate relative and absolute lift relative_lift = (target_rate - baseline_rate) / baseline_rate absolute_lift = target_rate - baseline_rate # Step 5: Estimate test duration (assume 1000 users/day) users_per_day = 1000
```

---

## [49/50] Linear Mixed Models: A Practical Guide Using Statistical Software: Third Edition

| Field | Value |
|-------|-------|
| **Pages** | 15-15 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.685 |
| **Found In** | fts |
| **Chunk ID** | `c24eae90-097a-450f-9f43-fd8a28e375f0` |
| **YOUR GRADE** | ____ |

**Full Text (996 chars):**

```
Contents

 = 8.8. , 2 = The Implied Marginal Covariance Matrix for the Final Model. , 3 = The Implied Marginal Covariance Matrix for the Final Model. , 4 = 415. , 1 = 8.9. , 2 = Recommended Diagnostics for the Final Model. , 3 = Recommended Diagnostics for the Final Model. , 4 = 416. , 1 = 8.10. , 2 = Software Notes and Additional Recommendations. , 3 = Software Notes and Additional Recommendations. , 4 = 417. 9, 1 = Power Analysis and Sample Size Calculations for Linear Mixed Models. 9, 2 = Power Analysis and Sample Size Calculations for Linear Mixed Models. 9, 3 = Power Analysis and Sample Size Calculations for Linear Mixed Models. 9, 4 = 419. , 1 = 9.1. , 2 = Introduction . . . . . . . .. , 3 = . . . . . . . . . . . .. , 4 = 419. , 1 = 9.2. , 2 = Direct Power Computations. , 3 = . . . . . . . . . . .. , 4 = 419. , 1 = . , 2 = 9.2.1. , 3 = Software for Direct Power Computations. , 4 = 420. , 1 = . , 2 = 9.2.2. , 3 = Examples of Direct Power Computations. , 4 = 420. , 1 = 9.3. , 2
```

---

## [50/50] Explanation in Causal Inference: Methods for Mediation and Interaction

| Field | Value |
|-------|-------|
| **Pages** | 11-11 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.639 |
| **Found In** | fts |
| **Chunk ID** | `205828db-c05f-4d72-b773-261152193ce3` |
| **YOUR GRADE** | ____ |

**Full Text (1274 chars):**

```
10. Mechanistic Interaction 286
- 10.7. Extensions to Three or More Exposures 302
- 10.8. Other Extensions 304
- 10.9. Antagonism 306
- 10.10. Limits of Inference Concerning Biology 316
- 10.11. Discussion 319
11. Bias Analysis for Interactions 320
- 11.1. Sensitivity Analysis and Robustness for Additive Interaction 320
- 11.2. Sensitivity Analysis and Robustness for Multiplicative Interaction 325
- 11.3. Sensitivity Analysis for the Relative Excess Risk Due to Interaction 327
- 11.4. Measurement Error and Additive Interaction 330
- 11.5. Measurement Error and Multiplicative Interaction 333
- 11.6. Discussion 335
12. Interaction in Genetics: Independence and Boosting Power 337
- 12.1. Case-Only Estimators of Interaction 337
- 12.2. Joint Tests for Interactions and Main Effects 340
- 12.3. Multiple Testing 343
- 12.4. Discussion 345
13. Power and Sample-Size Calculations for Interaction Analysis 346
- 13.1. Power and Sample-Size Calculations for Interaction for Continuous Outcomes 347
- 13.2. Power and Sample-Size Calculations for Binary Outcomes: Multiplicative Interaction 348
- 13.3. Power and Sample Size Calculations for Binary Outcomes: Additive Interaction 355
- 13.4. Power and Sample Size Calculations for Binary Outcomes: Mechanistic Interaction 363
```

---
