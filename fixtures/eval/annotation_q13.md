# Query 13: bootstrap confidence interval

**Domain:** statistics
**Query ID:** q_stat_005
**Candidates:** 44
**Grading:** 0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant

---

## [1/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability -Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 510-511 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.840 |
| **Found In** | hybrid |
| **Chunk ID** | `7bf69782-4d98-43a7-9b63-bf45b8a51e8b` |
| **YOUR GRADE** | ____ |

**Full Text (918 chars):**

```
29.14 Confidence Intervals
The bootstrap distribution is defined as H Boot ( x ) = P ∗ ( θ ∗ s ≤ x ). For given α , let H -1 Boot ( α ) be the α th quantile of H Boot. We consider the bootstrapt (BT) confidence bounds and intervals for θ . They are obtained as
<!-- formula-not-decoded -->
There are some remarkable results on the accuracy in coverage of the BT one-sided bounds and confidence intervals. We state one key result below.
and the intervals θ L , BT = θ ( α/ 2) BT and θ U , BT = ¯ θ ( α/ 2) BT .
<!-- formula-not-decoded -->
These results are derived in Hall (1989).
Remark. It is remarkable that one already gets third-order accuracy for the one-sided confidence bounds and fourth-order accuracy for the two-sided bounds. There seems to be no intuitive explanation for this phenomenon. It just happens that certain terms cancel in the Cornish-Fisher expansions used in the proof for the regression case.
```

---

## [2/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability-Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 492-493 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.839 |
| **Found In** | hybrid |
| **Chunk ID** | `a63e98b0-f259-4721-a858-a0235e34c4d2` |
| **YOUR GRADE** | ____ |

**Full Text (947 chars):**

```
29.14 Confidence Intervals
The bootstrap distribution is defined as H Boot ( x ) = P ∗ ( θ ∗ s ≤ x ). For given α , let H -1 Boot ( α ) be the α th quantile of H Boot. We consider the bootstrapt (BT) confidence bounds and intervals for θ . They are obtained as
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
and the intervals θ L , BT = θ ( α/ 2) BT and θ U , BT = ¯ θ ( α/ 2) BT .
There are some remarkable results on the accuracy in coverage of the BT one-sided bounds and confidence intervals. We state one key result below.
<!-- formula-not-decoded -->
These results are derived in Hall (1989).
Remark. It is remarkable that one already gets third-order accuracy for the one-sided confidence bounds and fourth-order accuracy for the two-sided bounds. There seems to be no intuitive explanation for this phenomenon. It just happens that certain terms cancel in the Cornish-Fisher expansions used in the proof for the regression case.
```

---

## [3/44] (Springer Texts in Statistics'',) Larry Wasserman - All of Statistics  A Concise Course in Statistical Inference (Springer Texts in Statistics)-Springer (2003)

| Field | Value |
|-------|-------|
| **Pages** | 123-124 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.830 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `91282621-a464-4d33-a893-8799d8466c90` |
| **YOUR GRADE** | ____ |

**Full Text (978 chars):**

```
8.3 Bootstrap Confidence Intervals
There are several ways to construct bootstrap confidence intervals. Here we discuss three methods.
Method 1: The Normal Interval. The simplest method is the Normal interval
T n ± z α/ 2 ̂ se boot (8.2) where ̂ se boot = √ v boot is the bootstrap estimate of the standard error. This interval is not accurate unless the distribution of T n is close to Normal.
Method 2: Pivotal Intervals. Let θ = T ( F ) and ̂ θ n = T ( ̂ F n ) and define the pivot R n = ̂ θ n -θ . Let ̂ θ ∗ n, 1 , . . . , ̂ θ ∗ n,B denote bootstrap replications of ̂ θ n . Let H ( r ) denote the cdf of the pivot:
<!-- formula-not-decoded -->
Define C /star n = ( a, b ) where
It follows that
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
Hence, C /star n is an exact 1 -α confidence interval for θ . Unfortunately, a and b depend on the unknown distribution H but we can form a bootstrap estimate of H :
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
```

---

## [4/44] R_in_Action,_Second_Edition (5)

| Field | Value |
|-------|-------|
| **Pages** | 325-326 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.829 |
| **Found In** | fts, vector, hybrid, citations |
| **Chunk ID** | `7350a637-4ca7-49b7-8816-cf6722ad5708` |
| **YOUR GRADE** | ____ |

**Full Text (1085 chars):**

```
12.6.2 Bootstrapping several statistics
```
> boot.ci(results, type="bca", index=2) BOOTSTRAP CONFIDENCE INTERVAL CALCULATIONS Based on 1000 bootstrap replicates CALL : boot.ci(boot.out = results, type = "bca", index = 2) Intervals : Level       BCa 95%   (-5.66, -1.19 ) Calculations and Intervals on Original Scale
```
Figure 12.3 Distribution of bootstrapping regression coefficients for car weight
```
> boot.ci(results, type="bca", index=3) BOOTSTRAP CONFIDENCE INTERVAL CALCULATIONS Based on 1000 bootstrap replicates CALL : boot.ci(boot.out = results, type = "bca", index = 3) Intervals : Level       BCa 95%   (-0.0331,  0.0010 ) Calculations and Intervals on Original Scale
```
NOTE The previous example resamples the entire sample of data each time. If  you  can  assume  that  the  predictor  variables  have  fixed  levels  (typical  in planned experiments), you'd do better to only resample residual terms. See Mooney and Duval (1993, pp. 16-17) for a simple explanation and algorithm.
Before we leave bootstrapping, it's worth addressing two questions that come up often:
```

---

## [5/44] computer age statistical inference

| Field | Value |
|-------|-------|
| **Pages** | 199-199 |
| **Suggested Grade** | 3 |
| **Similarity** | 0.829 |
| **Found In** | fts, vector, hybrid, citations |
| **Chunk ID** | `056d77cb-54e6-4fcd-9f2f-47ccac984674` |
| **YOUR GRADE** | ____ |

**Full Text (1097 chars):**

```
Bootstrap Confidence Intervals
The jackknife and the bootstrap represent a different use of modern computer power: rather than extending classical methodology-from ordinary least squares to generalized linear models, for example-they extend the reach of classical inference.
Chapter 10 focused on standard errors. Here we will take up a more ambitious inferential goal, the bootstrap automation of confidence intervals. The familiar standard intervals
<!-- formula-not-decoded -->
for approximate 95% coverage, are immensely useful in practice but often not very accurate. If we observe O D 10 from a Poisson model O Poi . / , the standard 95% interval .3:8; 16:2/ (using b se D O 1=2 ) is a mediocre approximation to the exact interval 1
<!-- formula-not-decoded -->
Standard intervals (11.1) are symmetric around O , this being their main weakness. Poisson distributions grow more variable as increases, which is why interval (11.2) extends farther to the right of O D 10 than to the left. Correctly capturing such effects in an automatic way is the goal of bootstrap confidence interval theory.
```

---

## [6/44] Introduction to Mathematical Statistics

| Field | Value |
|-------|-------|
| **Pages** | 322-322 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.828 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `80e9f47f-92e7-4ac7-a643-062c1de07001` |
| **YOUR GRADE** | ____ |

**Full Text (1208 chars):**

```
4.9.1 Percentile Bootstrap Confidence Intervals
size of the sample and provides a vector to store the ˆ θ ∗ s. In the for loop, the i th bootstrap sample is obtained by the single command sample(x,n,replace=T) , which is followed by the computation of ˆ θ ∗ i . The remainder of the code forms the bootstrap confidence interval, while the list command returns the estimate and the bootstrap confidence interval. The optional second list command returns the ˆ θ ∗ s, also. Notice that it easy to change the code for an estimator other than the mean. For example, to obtain a bootstrap confidence interval for the median just replace the two occurrences of mean with median . We illustrate this discussion in the next example.
Example 4.9.1. In this example, we sample from a known distribution, but, in practice, the distribution is usually unknown. Let X 1 , X 2 , . . . , X n be a random sample from a Γ(1 , β ) distribution. Since the mean of this distribution is β , the sample average X is an unbiased estimator of β . In this example, the X serves as our point estimator of β . The following 20 data points are the realizations (rounded) of a random sample of size n = 20 from a Γ(1 , 100) distribution:
```

---

## [7/44] Robert V. Hogg, Joeseph McKean, Allen T Craig-Introduction to Mathematical Statistics-Pearson

| Field | Value |
|-------|-------|
| **Pages** | 285-285 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.827 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `ce78045d-87f6-42d4-988e-54f1e7a06e8e` |
| **YOUR GRADE** | ____ |

**Full Text (976 chars):**

```
4.9.1 Percentile Bootstrap Confidence Intervals
We now give an algorithm which obtains a bootstrap confidence interval. For clarity, we present a formal algorithm, which can be readily coded into languages such as R. Let x ′ = ( x 1 , x 2 , . . . , x n ) be the realization of a random sample drawn from a cdf F ( x ; θ ), θ ∈ Ω. Let ̂ θ be a point estimator of θ . Let B , an integer, denote the number of bootstrap replications, i.e., the number of resamples. In practice, B is often 3000 or more.
1. Set j = 1.
2. While j ≤ B , do steps 2-5.
3. Let x ∗ j be a random sample of size n drawn from the sample x . That is, the observations x ∗ j are drawn at random from x 1 , x 2 , . . . , x n , with replacement.
4. Let ̂ θ ∗ j = ̂ θ ( x ∗ j ). 5. Replace j by j +1.
6. Let ̂ θ ∗ (1) ≤ ̂ θ ∗ (2) ≤ · · · ≤ ̂ θ ∗ ( B ) denote the ordered values of ̂ θ ∗ 1 , ̂ θ ∗ 2 , . . . , ̂ θ ∗ B . Let m = [( α/ 2) B ], where [ · ] denotes the greatest integer function. Form the interval
```

---

## [8/44] Introduction to Mathematical Statistics

| Field | Value |
|-------|-------|
| **Pages** | 321-321 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.827 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `b5440212-5b7b-4e24-afff-191202ad455d` |
| **YOUR GRADE** | ____ |

**Full Text (975 chars):**

```
4.9.1 Percentile Bootstrap Confidence Intervals
We now give an algorithm that obtains a bootstrap confidence interval. For clarity, we present a formal algorithm, which can be readily coded into languages such as R. Let x ′ = ( x 1 , x 2 , . . . , x n ) be the realization of a random sample drawn from a cdf F ( x ; θ ), θ ∈ Ω. Let ̂ θ be a point estimator of θ . Let B , an integer, denote the number of bootstrap replications, i.e., the number of resamples. In practice, B is often 3000 or more.
1. Set j = 1.
2. While j ≤ B , do steps 2-5.
3. Let x ∗ j be a random sample of size n drawn from the sample x . That is, the observations x ∗ j are drawn at random from x 1 , x 2 , . . . , x n , with replacement.
4. Let ̂ θ ∗ j = ̂ θ ( x ∗ j ). 5. Replace j by j +1.
6. Let ̂ θ ∗ (1) ≤ ̂ θ ∗ (2) ≤ · · · ≤ ̂ θ ∗ ( B ) denote the ordered values of ̂ θ ∗ 1 , ̂ θ ∗ 2 , . . . , ̂ θ ∗ B . Let m = [( α/ 2) B ], where [ · ] denotes the greatest integer function. Form the interval
```

---

## [9/44] google_ds_l4_l5_guide

| Field | Value |
|-------|-------|
| **Pages** | 115-115 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.825 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `f520959e-a7b8-465e-8f96-1128677bfc5f` |
| **YOUR GRADE** | ____ |

**Full Text (1014 chars):**

```
Python Solution:
```
import pandas as pd import numpy as np from typing import Dict, Callable def bootstrap_confidence_interval( control: np.ndarray, treatment: np.ndarray, statistic: Callable = np.mean, n_bootstrap: int = 10000, confidence_level: float = 0.95, random_state: int = 42 ) -> Dict: """ Calculate bootstrap confidence interval for difference between two groups. Time: O(n_bootstrap × (n + m)) Space: O(n_bootstrap) DS Application: A/B testing, conversion rate analysis, any metric without closedform CI Advantages: 1. No distributional assumptions 2. Works for any statistic (mean, median, ratio, percentile) 3. Asymptotically accurate """ np.random.seed(random_state) # Step 1: Calculate observed statistic observed_diff = statistic(treatment) - statistic(control) # Step 2: Bootstrap resampling bootstrap_diffs = [] for _ in range(n_bootstrap): # Resample with replacement control_resample = np.random.choice(control, size=len(control), replace=True) treatment_resample = np.random.choice(treatment,
```

---

## [10/44] Nicolas Bousquet (editor), Pietro Bernardara (editor) - Extreme Value Theory with Applications to Natural Hazards: From Statistical Theory to Industrial Practice-Springer

| Field | Value |
|-------|-------|
| **Pages** | 182-183 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.824 |
| **Found In** | hybrid |
| **Chunk ID** | `e1f1d1bd-cead-4615-b948-83c18a023a01` |
| **YOUR GRADE** | ____ |

**Full Text (1119 chars):**

```
8.1.2.3 Bootstrap and Confidence Intervals
- (b) Nonparametric approach. The boundaries of the 100 ( 1 -α) % confidence interval are estimated by the order α/ 2 and 1 -α/ 2 empirical quantiles of the bootstrap distribution. We denote ˆ ψ b the variable that has this distribution. As in the previous method, this approach can be corrected by supposing that the unknown bias ˆ ψ -ψ can be estimated by the difference ¯ ψ b - ˆ ψ . The original bootstrap distribution is thus modified by replacing ˆ ψ b with ˆ ψ b + ˆ ψ - ¯ ψ b .
- (c) Bias-corrected and accelerated (BCa) method. This method, proposed by Efron [245], allows confidence intervals obtained using the nonparametric approach (empirical quantiles) to be corrected so as to obtain the correct skewness ∗ and kurtosis ∗ of the distribution. The boundaries of the 100 ( 1 -α) % confidence interval are estimated by the q α 1 and q α 2 quantiles of the bootstrap distribution, with
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
where /Phi1 is the cumulative distribution function of the standard normal distribution, and
<!-- formula-not-decoded -->
```

---

## [11/44] (Springer Texts in Statistics'',) Larry Wasserman - All of Statistics  A Concise Course in Statistical Inference (Springer Texts in Statistics)-Springer (2003)

| Field | Value |
|-------|-------|
| **Pages** | 129-130 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.820 |
| **Found In** | fts, vector, hybrid, citations |
| **Chunk ID** | `27926d06-f2a6-44d0-9255-6136b6a25622` |
| **YOUR GRADE** | ____ |

**Full Text (1067 chars):**

```
8.6 Exercises
1. Consider the data in Example 8.6. Find the plug-in estimate of the correlation coefficient. Estimate the standard error using the bootstrap. Find a 95 percent confidence interval using the Normal, pivotal, and percentile methods.
2. ( Computer Experiment. ) Conduct a simulation to compare the various bootstrap confidence interval methods. Let n = 50 and let T ( F ) = ∫ ( x -µ ) 3 dF ( x ) /σ 3 be the skewness. Draw Y 1 , . . . , Y n ∼ N (0 , 1) and set X i = e Y i , i = 1 , . . . , n . Construct the three types of bootstrap 95 percent intervals for T ( F ) from the data X 1 , . . . , X n . Repeat this whole thing many times and estimate the true coverage of the three intervals.
3. Let
<!-- formula-not-decoded -->
where n = 25. Let θ = T ( F ) = ( q . 75 -q . 25 ) / 1 . 34 where q p denotes the p th quantile. Do a simulation to compare the coverage and length of the following confidence intervals for θ : (i) Normal interval with standard error from the bootstrap, (ii) bootstrap percentile interval, and (iii) pivotal bootstrap interval.
```

---

## [12/44] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 354-355 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.819 |
| **Found In** | fts, hybrid, citations |
| **Chunk ID** | `44428ab5-002d-47c6-b4bf-b3d2b8c93290` |
| **YOUR GRADE** | ____ |

**Full Text (1043 chars):**

```
10.4.2 Example: sleep data
which is a bit shorter. It is a good idea to check and see how the bootstrap confidence intervals compare.
```
set.seed (236791) n.boot<-100000 bootstraps<rep (NA,n.boot) for (i in 1 : n.boot) { #Bootstrap the data bootstraps[i]<mean (diff_T2_minus_T1[ sample (1 : 10,replace=TRUE)]) } quantile (bootstraps,probs= c (0.025,0.975))
```
```
2.5% 97.5% 0.95 2.38
```
Interestingly, the bootstrap-based confidence interval is 6 % shorter than the CLT, which, in turn, is 14 % shorter than the t confidence interval. This is likely due to the fact that both the bootstrap and the CLT confidence intervals have a lower than nominal probability ( 95 %) of covering the true value of the parameter. This is a recognized problem of the bootstrap in small sample sizes and many corrections have been proposed. Probably one of the best solutions is to estimate the mean and standard deviation of the bootstrap means and then use the quantiles of the t n -1 distribution to construct the confidence interval. This is done below:
```

---

## [13/44] Introduction to Mathematical Statistics

| Field | Value |
|-------|-------|
| **Pages** | 322-322 |
| **Suggested Grade** | 2 |
| **Similarity** | 0.819 |
| **Found In** | fts, vector, hybrid, citations |
| **Chunk ID** | `3d7bc4e9-f806-43b1-be56-ef38040eca65` |
| **YOUR GRADE** | ____ |

**Full Text (1102 chars):**

```
4.9.1 Percentile Bootstrap Confidence Intervals
The sample mean of this particular bootstrap sample is x ∗ = 78 . 725. To obtain our bootstrap confidence interval for β , we need to compute many more resamples. For this computation, we used the R function percentciboot discussed above. Let x denote the R vector of the original sample of observations. We selected 3000 as the number of bootstraps and chose α = 0 . 10. We used the code percentciboot(x,3000,.10) to compute our bootstrap confidence interval. Figure 4.9.1 displays a histogram of the 3000 sample means x ∗ s computed by the code. The sample mean of these 3000 values is 90.13, close to x = 90 . 59. Our program also obtained a 90% (bootstrap percentile) confidence interval given by (61 . 655 , 120 . 48), which the reader can locate on the figure. It does trap the true value µ = 100. Exercise 4.9.3 shows that if we are sampling from a Γ(1 , β ) distribution, then the interval (2 nx/ [ χ 2 2 n ] (1 -( α/ 2)) , 2 nx/ [ χ 2 2 n ] ( α/ 2) ) is an exact (1 -α )100% confidence interval for β . Note that, in keeping with our superscript
```

---

## [14/44] computer age statistical inference

| Field | Value |
|-------|-------|
| **Pages** | 203-204 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.818 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `3ab1c907-800f-409e-9300-1286fca199a6` |
| **YOUR GRADE** | ____ |

**Full Text (1231 chars):**

```
11.2 The Percentile Method
Our goal is to automate the calculation of confidence intervals: given the bootstrap distribution of a statistical estimator O , we want to automatically produce an appropriate confidence interval for the unseen parameter . To this end, a series of four increasingly accurate bootstrap confidence interval algorithms will be described.
The first and simplest method is to use the standard interval (11.1), O 1:96 b se for 95% coverage, with b se taken to be the bootstrap standard error b seboot (10.16). The limitations of this approach become obvious in Figure 11.3, where the histogram shows B D 2000 nonparametric bootstrap replications O of the sample correlation coefficient for the student
3 This is an anachronism. Fisher hated the term 'confidence interval' after it was later coined by Neyman for his comprehensive theory. He thought of (11.13) as an example of the logic of inductive inference .
Figure 11.3 Histogram of B D 2000 nonparametric bootstrap replications O for the student score sample correlation; the solid curve is the ideal parametric bootstrap distribution f O .r/ as in Figure 11.1. Observed correlation O D 0:498 . Small triangles show histogram's 0.025 and 0.975 quantiles.
```

---

## [15/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability-Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 21-21 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.815 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `7502d9fe-4668-4355-a3cb-eca4581817d3` |
| **YOUR GRADE** | ____ |

**Full Text (801 chars):**

```
Contents

 ,  = 29.10. ,  = Some Numerical Examples . . . . . . . . . . . . . . . . . . . .. , Contents = . . . 482. ,  = 29.11. ,  = Bootstrap Confidence Intervals for Quantiles . . . . .. , Contents = . . . 483. ,  = 29.12. ,  = Bootstrap in Regression . . . . . . . . . . . . . . . . . . . . . .. , Contents = . . . 483. ,  = 29.13. ,  = Residual Bootstrap . . . . . . . . . . . . . . . . . . . . . . . . . . .. , Contents = . . . 484. ,  = 29.14. ,  = Confidence Intervals . . . . . . . . . . . . . . . . . . . . . . . . .. , Contents = . . . 485. ,  = 29.15. ,  = Distribution Estimates in Regression . . . . . . . . . . . .. , Contents = . . . 486. ,  = 29.16. ,  = Bootstrap for Dependent Data . . . . . . . . . . . . . . . . .. , Contents = . . . 487. ,  = 29.17. ,  = Consistent Bootstrap for
```

---

## [16/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability -Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 22-22 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.815 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `224849a0-03ad-4992-8949-a13a689d04f6` |
| **YOUR GRADE** | ____ |

**Full Text (801 chars):**

```
Contents

 ,  = 29.10. ,  = Some Numerical Examples . . . . . . . . . . . . . . . . . . . .. , Contents = . . . 482. ,  = 29.11. ,  = Bootstrap Confidence Intervals for Quantiles . . . . .. , Contents = . . . 483. ,  = 29.12. ,  = Bootstrap in Regression . . . . . . . . . . . . . . . . . . . . . .. , Contents = . . . 483. ,  = 29.13. ,  = Residual Bootstrap . . . . . . . . . . . . . . . . . . . . . . . . . . .. , Contents = . . . 484. ,  = 29.14. ,  = Confidence Intervals . . . . . . . . . . . . . . . . . . . . . . . . .. , Contents = . . . 485. ,  = 29.15. ,  = Distribution Estimates in Regression . . . . . . . . . . . .. , Contents = . . . 486. ,  = 29.16. ,  = Bootstrap for Dependent Data . . . . . . . . . . . . . . . . .. , Contents = . . . 487. ,  = 29.17. ,  = Consistent Bootstrap for
```

---

## [17/44] Robert V. Hogg, Joeseph McKean, Allen T Craig-Introduction to Mathematical Statistics-Pearson

| Field | Value |
|-------|-------|
| **Pages** | 286-286 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.815 |
| **Found In** | fts, vector, hybrid, citations |
| **Chunk ID** | `1aabc0f2-73b6-4134-bd97-78cc1b44de74` |
| **YOUR GRADE** | ____ |

**Full Text (1030 chars):**

```
4.9.1 Percentile Bootstrap Confidence Intervals
The value of X for this sample is x = 90 . 59, which is our point estimate of β . For illustration, we generated one bootstrap sample of these data. This ordered bootstrap sample is
<!-- formula-not-decoded -->
As Exercise 4.9.1 shows, in general, the sample mean of a bootstrap sample is an unbiased estimator of original sample mean x . The sample mean of this particular bootstrap sample is x ∗ = 78 . 725. We wrote an R function to generate bootstrap samples and the percentile confidence interval above; see the program percentciboot.s of Appendix B. Figure 4.9.1 displays a histogram of 3000 x ∗ s for the above sample. The sample mean of these 3000 values is 90.13, close to x = 90 . 59. Our program also obtained a 90% (bootstrap percentile) confidence interval given by (61 . 655 , 120 . 48), which the reader can locate on the figure. It did trap µ = 100.
Figure 4.9.1: Histogram of the 3000 bootstrap x ∗ s. The 90% bootstrap confidence interval is (61 . 655 , 120 . 48).
```

---

## [18/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability-Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 492-492 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.814 |
| **Found In** | vector, hybrid |
| **Chunk ID** | `b493eefc-1074-418d-80bf-c3fb9600ba30` |
| **YOUR GRADE** | ____ |

**Full Text (745 chars):**

```
29.14 Confidence Intervals
We present some results on bootstrap confidence intervals for a linear combination θ = c ′ β 1, where β ′ = ( β 0 , β ′ 1 ); i.e., there is an intercept term in the model. Correspondingly, x ′ i = (1 , t ′ i ). The confidence interval for θ or confidence bounds (lower or upper) are going to be in terms of the studentized version of the LSE of θ , namely ˆ θ = c ′ ˆ β 1 . In fact, ˆ β 1 = S -1 t t Sty , where Stt = ∑ i ( t i -¯ t )( t i -¯ t ) ′ and Sty = ∑ i ( t i -¯ t )( yi -¯ y ) ′ . The bootstrapped version of ˆ θ is θ ∗ = c ′ β ∗ 1 , where β ∗ ′ = ( β ∗ 0 , β ∗ ′ 1 ) as before. Since the variance of ˆ θ is σ 2 c ′ S -1 t t c , the bootstrapped version of the studentized ˆ θ is
<!-- formula-not-decoded -->
```

---

## [19/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability -Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 510-510 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.814 |
| **Found In** | vector, hybrid |
| **Chunk ID** | `bcc0325a-0f25-4f13-9109-86522bb4c224` |
| **YOUR GRADE** | ____ |

**Full Text (745 chars):**

```
29.14 Confidence Intervals
We present some results on bootstrap confidence intervals for a linear combination θ = c ′ β 1, where β ′ = ( β 0 , β ′ 1 ); i.e., there is an intercept term in the model. Correspondingly, x ′ i = (1 , t ′ i ). The confidence interval for θ or confidence bounds (lower or upper) are going to be in terms of the studentized version of the LSE of θ , namely ˆ θ = c ′ ˆ β 1 . In fact, ˆ β 1 = S -1 t t Sty , where Stt = ∑ i ( t i -¯ t )( t i -¯ t ) ′ and Sty = ∑ i ( t i -¯ t )( yi -¯ y ) ′ . The bootstrapped version of ˆ θ is θ ∗ = c ′ β ∗ 1 , where β ∗ ′ = ( β ∗ 0 , β ∗ ′ 1 ) as before. Since the variance of ˆ θ is σ 2 c ′ S -1 t t c , the bootstrapped version of the studentized ˆ θ is
<!-- formula-not-decoded -->
```

---

## [20/44] Introduction to Mathematical Statistics

| Field | Value |
|-------|-------|
| **Pages** | 329-329 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.813 |
| **Found In** | hybrid, citations |
| **Chunk ID** | `7684259f-cfda-4fba-8317-e354a7ed40ca` |
| **YOUR GRADE** | ____ |

**Full Text (1074 chars):**

```
EXERCISES
- 4.9.5. Suppose X 1 , X 2 , . . . , X n is a random sample drawn from a N ( µ, σ 2 ) distribution. As discussed in Example 4.2.1, the pivot random variable for a confidence interval is
<!-- formula-not-decoded -->
where X and S are the sample mean and standard deviation, respectively. Recall by Theorem 3.6.1 that t has a Student t -distribution with n -1 degrees of freedom; hence, its distribution is free of all parameters for this normal situation. In the notation of this section, t ( γ ) n -1 denotes the γ 100% percentile of a t -distribution with n -1 degrees of freedom. Using this notation, show that a (1 -α )100% confidence interval for µ is
<!-- formula-not-decoded -->
- 4.9.6. Frequently, the bootstrap percentile confidence interval can be improved if the estimator ̂ θ is standardized by an estimate of scale. To illustrate this, consider a bootstrap for a confidence interval for the mean. Let x ∗ 1 , x ∗ 2 , . . . , x ∗ n be a bootstrap sample drawn from the sample x 1 , x 2 , . . . , x n . Consider the bootstrap pivot [analog of (4.9.13)]:
```

---

## [21/44] computer age statistical inference

| Field | Value |
|-------|-------|
| **Pages** | 220-220 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.810 |
| **Found In** | fts, vector, citations |
| **Chunk ID** | `7459012e-5fef-49d0-90ef-61933ca86135` |
| **YOUR GRADE** | ____ |

**Full Text (661 chars):**

```
11.6 Objective Bayes Intervals and the Confidence Distribution
Bootstrap confidence intervals provide easily computable confidence densities. Let O G. / be the bootstrap cdf and O g. / its density function (obtained by differentiating a smoothed version of O G. / when O G is based on B bootstrap replications). The percentile confidence limits O D O G 1 . / (11.17) have D O G. / , giving
<!-- formula-not-decoded -->
(It is helpful to picture this in Figure 11.4.) For the percentile method, the bootstrap density is the confidence density.
For the BCa intervals (11.39), the confidence density is obtained by reweighting O g. / ,
<!-- formula-not-decoded -->
```

---

## [22/44] R_in_Action,_Second_Edition (5)

| Field | Value |
|-------|-------|
| **Pages** | 322-322 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.810 |
| **Found In** | vector |
| **Chunk ID** | `9c8ae71e-f34b-4de0-9545-9d5eee7d1f3a` |
| **YOUR GRADE** | ____ |

**Full Text (991 chars):**

```
Now to the specifics.
Table 12.4 Elements of the object returned by the boot() function

t0, Description = The observed values of k statistics applied to the original data. t, Description = An R × k matrix, where each row is a bootstrap replicate of the k statistics
You can access these elements as bootobject$t0 and bootobject$t .
Once  you  generate  the  bootstrap  samples,  you  can  use print() and plot() to examine the results. If the results look reasonable, you can use the boot.ci() function to obtain confidence intervals for the statistic(s). The format is boot.ci( bootobject , conf=, type= )
The parameters are given in table 12.5.
Table 12.5 Parameters of the boot.ci() function

bootobject, Description = The object returned by the boot() function.. conf, Description = The desired confidence interval (default: conf =0.95).. type, Description = The type of confidence interval returned. Possible values are norm , basic , stud , perc , bca , and all (default: type="all" )
```

---

## [23/44] Robert V. Hogg, Joeseph McKean, Allen T Craig-Introduction to Mathematical Statistics-Pearson

| Field | Value |
|-------|-------|
| **Pages** | 294-294 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.809 |
| **Found In** | vector |
| **Chunk ID** | `53093fa8-8efd-4ed1-8257-95b3a622e866` |
| **YOUR GRADE** | ____ |

**Full Text (865 chars):**

```
EXERCISES
- (b) Rewrite the R program percentciboot.s of Appendix B and use it to find a 90% confidence interval for µ for the data in Example 4.9.3. Use 3000 bootstraps.
- (c) Compare your confidence interval in the last part with the nonstandardized bootstrap confidence interval based on the program percentciboot.s of Appendix B.
- 4.9.6. Consider the algorithm for a two-sample bootstrap test given in Section 4.9.2.
- (a) Rewrite the algorithm for the bootstrap test based on the difference in medians.
- (b) Consider the data in Example 4.9.2. By substituting the difference in medians for the difference in means in the R program boottesttwo.s of Appendix B, obtain the bootstrap test for the algorithm of part (a).
- (c) Obtain the estimated p -value of your test for B = 3000 and compare it to the estimated p -value of 0 . 063 which the authors obtained.
```

---

## [24/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability-Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 485-485 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.808 |
| **Found In** | fts, citations |
| **Chunk ID** | `0a354eda-6384-4414-998f-c3b8d9021a5b` |
| **YOUR GRADE** | ____ |

**Full Text (1412 chars):**

```
29.9 Bootstrap Confidence Intervals
This agenda requires the use of a standard deviation estimate ˆ σ n for the standard deviation of ˆ θ n and the knowledge of the function G ( x ). Furthermore, in many cases, the limiting CDF G may depend on some unknown parameters, too, that will have to be estimated in turn to construct the confidence interval. The bootstrap methodology offers an omnibus, sometimes easy to implement, and often more accurate method of constructing confidence intervals. Bootstrap confidence intervals and lower and upper one-sided confidence limits of various types have been proposed in great generality. Although, as a matter of methodology, they can be used in an automatic manner, a theoretical evaluation of their performance requires specific structural assumptions. The theoretical evaluation involves an Edgeworth expansion for the relevant statistic and an expansion for their quantiles, called Cornish-Fisher expansions. Necessarily, we are limited to the cases where the underlying statistic admits a known Edgeworth and Cornish-Fisher expansion. The main reference is Hall (1988), but see also G¨ oetze (1989), Hall and Martin (1989), Bickel (1992), Konishi (1991), DiCiccio and Efron (1996), and Lee (1999), of which the article by DiCiccio and Efron is a survey article and Lee (1999) discusses m / n bootstrap confidence intervals. There are also confidence intervals based
```

---

## [25/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability -Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 503-503 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.808 |
| **Found In** | fts, citations |
| **Chunk ID** | `89869a41-b9d8-4e82-acc3-0dd5a3e6bdc8` |
| **YOUR GRADE** | ____ |

**Full Text (1412 chars):**

```
29.9 Bootstrap Confidence Intervals
This agenda requires the use of a standard deviation estimate ˆ σ n for the standard deviation of ˆ θ n and the knowledge of the function G ( x ). Furthermore, in many cases, the limiting CDF G may depend on some unknown parameters, too, that will have to be estimated in turn to construct the confidence interval. The bootstrap methodology offers an omnibus, sometimes easy to implement, and often more accurate method of constructing confidence intervals. Bootstrap confidence intervals and lower and upper one-sided confidence limits of various types have been proposed in great generality. Although, as a matter of methodology, they can be used in an automatic manner, a theoretical evaluation of their performance requires specific structural assumptions. The theoretical evaluation involves an Edgeworth expansion for the relevant statistic and an expansion for their quantiles, called Cornish-Fisher expansions. Necessarily, we are limited to the cases where the underlying statistic admits a known Edgeworth and Cornish-Fisher expansion. The main reference is Hall (1988), but see also G¨ oetze (1989), Hall and Martin (1989), Bickel (1992), Konishi (1991), DiCiccio and Efron (1996), and Lee (1999), of which the article by DiCiccio and Efron is a survey article and Lee (1999) discusses m / n bootstrap confidence intervals. There are also confidence intervals based
```

---

## [26/44] Robert V. Hogg, Joeseph McKean, Allen T Craig-Introduction to Mathematical Statistics-Pearson

| Field | Value |
|-------|-------|
| **Pages** | 287-287 |
| **Suggested Grade** | 1 |
| **Similarity** | 0.807 |
| **Found In** | fts, citations |
| **Chunk ID** | `f95f747b-4149-44bf-8376-9a5eea11e80e` |
| **YOUR GRADE** | ____ |

**Full Text (715 chars):**

```
4.9.1 Percentile Bootstrap Confidence Intervals
What about the validity of a bootstrap confidence interval? Davison and Hinkley (1997) discuss the theory behind the bootstrap in Chapter 2 of their book. Under some general conditions, they show that the bootstrap confidence interval is asymptotically valid.
One way of improving the bootstrap is to use a pivot random variable, a variable whose distribution is free of other parameters. For instance, in the last example, instead of using X , use X/ ˆ σ X , where ˆ σ X = S/ √ n and S = [ ∑ ( X i -X ) 2 / ( n -1)] 1 / 2 ; that is, adjust X by its standard error. This is discussed in Exercise 4.9.5. Other improvements are discussed in the two books cited earlier.
```

---

## [27/44] Introduction to Mathematical Statistics

| Field | Value |
|-------|-------|
| **Pages** | 330-330 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.807 |
| **Found In** | vector |
| **Chunk ID** | `950a3d73-0646-475f-8a50-7081022a8770` |
| **YOUR GRADE** | ____ |

**Full Text (829 chars):**

```
EXERCISES
- (b) Rewrite the R program percentciboot.s and then use it to find a 90% confidence interval for µ for the data in Example 4.9.3. Use 3000 bootstraps.
- (c) Compare your confidence interval in the last part with the nonstandardized bootstrap confidence interval based on the program percentciboot.s .
- 4.9.7. Consider the algorithm for a two-sample bootstrap test given in Section 4.9.2.
- (a) Rewrite the algorithm for the bootstrap test based on the difference in medians.
- (b) Consider the data in Example 4.9.2. By substituting the difference in medians for the difference in means in the R program boottesttwo.s , obtain the bootstrap test for the algorithm of part (a).
- (c) Obtain the estimated p -value of your test for B = 3000 and compare it to the estimated p -value of 0 . 063 that the authors obtained.
```

---

## [28/44] Introduction to Mathematical Statistics

| Field | Value |
|-------|-------|
| **Pages** | 322-323 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.806 |
| **Found In** | fts, vector, citations |
| **Chunk ID** | `33fa1d9e-f69f-45ad-b8c9-2ee77078bfb0` |
| **YOUR GRADE** | ____ |

**Full Text (1036 chars):**

```
4.9.1 Percentile Bootstrap Confidence Intervals
notation for critical values, [ χ 2 2 n ] ( γ ) denotes the γ 100% percentile of a χ 2 distribution with 2 n degrees of freedom. This exact 90% confidence interval for our sample is (64 . 99 , 136 . 69).
What about the validity of a bootstrap confidence interval? Davison and Hinkley (1997) discuss the theory behind the bootstrap in Chapter 2 of their book. Under some general conditions, they show that the bootstrap confidence interval is asymptotically valid.
Figure 4.9.1: Histogram of the 3000 bootstrap x ∗ s. The 90% bootstrap confidence interval is (61 . 655 , 120 . 48).
One way of improving the bootstrap is to use a pivot random variable, a variable whose distribution is free of other parameters. For instance, in the last example, instead of using X , use X/ ˆ σ X , where ˆ σ X = S/ √ n and S = [ ∑ ( X i -X ) 2 / ( n -1)] 1 / 2 ; that is, adjust X by its standard error. This is discussed in Exercise 4.9.6. Other improvements are discussed in the two books cited earlier.
```

---

## [29/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability -Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 519-519 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.805 |
| **Found In** | vector |
| **Chunk ID** | `036e0e26-a28e-4a2f-8175-5ece47a40d3e` |
| **YOUR GRADE** | ____ |

**Full Text (988 chars):**

```
29.20 Exercises
- (a) the usual 95% confidence interval;
- (b) the interval based on the variance stabilizing transformation (Fisher's z ) (see Chapter 4);
- (c) the bootstrap percentile interval;
- (d) the bootstrap hybrid percentile interval;
- (e) the bootstrapt interval with ˆ σ n as the usual estimate;
- (f) the accelerated bias-corrected bootstrap interval using ϕ as Fisher's z , z 0 = r 2 √ n (the choice coming from theory), and three different values of a near zero.
Discuss your findings.
Exercise 29.15 * In which of the following cases are the results in Hall (1988) not applicable and why?
- (a) estimating the 80th percentile of a density on R ;
- (b) estimating the variance of a Gamma density with known scale and unknown shape parameter;
- (c) estimating θ in the U [0 , θ ] density;
- (d) estimating P ( X > 0) in a location-parameter Cauchy density;
- (e) estimating the variance of the t -statistic for Weibull data;
- (f) estimating a binomial success probability.
```

---

## [30/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability-Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 501-501 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.805 |
| **Found In** | vector |
| **Chunk ID** | `bcf9cc95-5910-4bb0-a4c5-9e93fdcc23e3` |
| **YOUR GRADE** | ____ |

**Full Text (988 chars):**

```
29.20 Exercises
- (a) the usual 95% confidence interval;
- (b) the interval based on the variance stabilizing transformation (Fisher's z ) (see Chapter 4);
- (c) the bootstrap percentile interval;
- (d) the bootstrap hybrid percentile interval;
- (e) the bootstrapt interval with ˆ σ n as the usual estimate;
- (f) the accelerated bias-corrected bootstrap interval using ϕ as Fisher's z , z 0 = r 2 √ n (the choice coming from theory), and three different values of a near zero.
Discuss your findings.
Exercise 29.15 * In which of the following cases are the results in Hall (1988) not applicable and why?
- (a) estimating the 80th percentile of a density on R ;
- (b) estimating the variance of a Gamma density with known scale and unknown shape parameter;
- (c) estimating θ in the U [0 , θ ] density;
- (d) estimating P ( X > 0) in a location-parameter Cauchy density;
- (e) estimating the variance of the t -statistic for Weibull data;
- (f) estimating a binomial success probability.
```

---

## [31/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability -Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 504-504 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.805 |
| **Found In** | vector |
| **Chunk ID** | `fc2e9a83-24e1-4adc-a942-f13b81c9f90b` |
| **YOUR GRADE** | ____ |

**Full Text (863 chars):**

```
29.9 Bootstrap Confidence Intervals
- (c) Bootstrap-t ( BT ). Let t n = ˆ θ n -θ ˆ σ n , where ˆ σ n is an estimate of the standard error of ˆ θ n , and let t ∗ n = ˆ θ ∗ n -ˆ θ n ˆ σ ∗ n be its bootstrap counterpart. As usual, let H Boot ( x ) = P ∗ { t ∗ n ≤ x } . The bootstrapt lower bound is ˆ θ n -H -1 Boot (1 -α ) ˆ σ n , and the two-sided BT confidence limits are ˆ θ n -H -1 Boot (1 -α 1 ) ˆ σ n and ˆ θ n -H -1 Boot ( α 2 ) ˆ σ n , where α 1 + α 2 = α , the nominal confidence level.
- (d) Bias-corrected bootstrap percentile bound ( BC ). The derivation of the BC bound involves quite a lot of calculation; see Efron (1981) and Shao and Tu (1995). The BC lower confidence bound is given by θ BC = ˆ G -1 [ ψ ( z α + 2 ψ -1 ( ˆ G ( ˆ θ n )))] , where ˆ G is the bootstrap distribution of ˆ θ ∗ n , ψ is as above, and z α = ψ -1 ( α ).
- BH = -cn Boot -
```

---

## [32/44] Credit-Risk Modelling : Theoretical Foundations, Diagnostic -- Bolder, David Jamieson -- 1st ed: 2018, Cham, 2018 -- Springer International Publishing -- 9783319946870 -- 1170ed4252a1f324789001b8a7d0729f -- Anna’s Archive

| Field | Value |
|-------|-------|
| **Pages** | 559-560 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.803 |
| **Found In** | vector |
| **Chunk ID** | `1f49a00c-65b5-4f6f-9f3d-6e707458195d` |
| **YOUR GRADE** | ____ |

**Full Text (1118 chars):**

```
9.2.8.2 The Bootstrap Technique
Fig. 9.11 Binomial bootstrap distribution : This figure summarizes the bootstrap distribution of ˆ p ∗ associated with 10,000 simulated samples of 10 observations. For perspective, the maximumlikelihood estimator and normal approximation are also illustrated. There is strong agreement among the results.
which agrees almost perfectly with the resulting 95% confidence interval from equation 9.74. 24 The alternative is to use the quantiles of the bootstrap distribution summarized in Fig. 9.11. That is,
<!-- formula-not-decoded -->
24 ¯ p ∗ and SE (p ∗ ) are computed precisely as one would expect. Given M simulations, they are defined as,
<!-- formula-not-decoded -->
<!-- formula-not-decoded -->
where p ∗ (m) denotes the maximum-likelihood estimate from the m th simulated dataset in our bootstrap computation.
Again, the agreement is very close, but the quantiles are, in this case, round figures. The reason is simple: the empirical distribution is constructed of a collection of simulated samples, each possessing only 10 observations. More granular estimates are not possible.
```

---

## [33/44] Introduction to Mathematical Statistics

| Field | Value |
|-------|-------|
| **Pages** | 411-411 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.803 |
| **Found In** | vector |
| **Chunk ID** | `5af94a7d-0081-4c0a-a745-9bba81ecb4bd` |
| **YOUR GRADE** | ____ |

**Full Text (983 chars):**

```
EXERCISES
- (b) Edit the R function bootstrapcis64.R to compute a bootstrap confidence interval for b . Then run your R function on the data of Exercise 6.4.7 to compute a 95% confidence interval for b .
- 6.4.9. Consider two Bernoulli distributions with unknown parameters p 1 and p 2 . If Y and Z equal the numbers of successes in two independent random samples, each of size n , from the respective distributions, determine the mles of p 1 and p 2 if we know that 0 ≤ p 1 ≤ p 2 ≤ 1.
- 6.4.10. Show that if X i follows the model (6.4.14), then its pdf is b -1 f (( x -a ) /b ).
- 6.4.11. Verify the partial derivatives and the entries of the information matrix for the location and scale family as given in Example 6.4.4.
- 6.4.12. Suppose the pdf of X is of a location and scale family as defined in Example 6.4.4. Show that if f ( z ) = f ( -z ), then the entry I 12 of the information matrix is 0. Then argue that in this case the mles of a and b are asymptotically independent.
```

---

## [34/44] Credit-Risk Modelling : Theoretical Foundations, Diagnostic -- Bolder, David Jamieson -- 1st ed: 2018, Cham, 2018 -- Springer International Publishing -- 9783319946870 -- 1170ed4252a1f324789001b8a7d0729f -- Anna’s Archive

| Field | Value |
|-------|-------|
| **Pages** | 557-558 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.803 |
| **Found In** | vector |
| **Chunk ID** | `b2cc48d9-b889-4bb2-8558-1f3df7b38e39` |
| **YOUR GRADE** | ____ |

**Full Text (1171 chars):**

```
9.2.8.2 The Bootstrap Technique
An additional advantage of this example is that it permits us to review the likelihood framework in some detail. Given the dataset in equation 9.65 along with our hardearned analytic formulae, our maximum-likelihood estimator is now simply
<!-- formula-not-decoded -->
Using the maximum-likelihood framework, assuming the Gaussianity of the error distribution, and equation 9.72, a 95% confidence interval for ˆ p is given as,
<!-- formula-not-decoded -->
Our interval estimate thus suggests that, with a 95% level of confidence, the true (and unknown) parameter value, p , lies between 0.10 and 0.70. While this is a surprisingly wide interval and may prove disappointing to the analyst-or her boss-it is nonetheless a realistic and prudent estimate.
At this point, we would like to employ the bootstrapping technique to this problem. The idea is relatively simple. We use ˆ p to generate a sequence of random samples-each of size N . For each simulated sample, we estimate an alternative population parameter estimate, p ∗ . 22 This yields an empirical distribution for ˆ p , which we employ to construct an associated interval estimate.
```

---

## [35/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability-Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 485-486 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.801 |
| **Found In** | vector |
| **Chunk ID** | `50297ef9-552e-436d-8c88-8f6d89c4d730` |
| **YOUR GRADE** | ____ |

**Full Text (1173 chars):**

```
29.9 Bootstrap Confidence Intervals
on more general subsampling methods, which work asymptotically under the mildest conditions. These intervals and their extensions to higher dimensions are discussed in Politis, Romano, and Wolf (1999).
Over time, various bootstrap confidence limits have been proposed. Generally, the evolution is from the algebraically simplest to progressively more complicated and computer-intensive formulas for the limits. Many of these limits have, however, now been incorporated into standard statistical software. We present below a selection of these different bootstrap confidence
limits and bounds. Let ˆ θ n = ˆ θ n ( X 1 , . . . , Xn ) be a specific estimate of the underlying parameter of interest θ .
- (a) The bootstrap percentile lower bound ( BP ). Let G ( x ) = Gn ( x ) = PF { ˆ θ n ≤ x } be the exact distribution and let ˆ G ( x ) = P ∗ { ˆ θ ∗ n ≤ x } be the bootstrap distribution. The lower 1 -α bootstrap percentile confidence bound would be ˆ G -1 ( α ), so the reported interval would be [ ˆ G -1 ( α ) , ∞ ). This was present in Efron (1979) itself, but it is seldom used because it tends to have a significant coverage bias.
```

---

## [36/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability -Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 503-504 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.801 |
| **Found In** | vector |
| **Chunk ID** | `51df602b-7cdf-47d8-a885-050ad599df5a` |
| **YOUR GRADE** | ____ |

**Full Text (1173 chars):**

```
29.9 Bootstrap Confidence Intervals
on more general subsampling methods, which work asymptotically under the mildest conditions. These intervals and their extensions to higher dimensions are discussed in Politis, Romano, and Wolf (1999).
Over time, various bootstrap confidence limits have been proposed. Generally, the evolution is from the algebraically simplest to progressively more complicated and computer-intensive formulas for the limits. Many of these limits have, however, now been incorporated into standard statistical software. We present below a selection of these different bootstrap confidence
limits and bounds. Let ˆ θ n = ˆ θ n ( X 1 , . . . , Xn ) be a specific estimate of the underlying parameter of interest θ .
- (a) The bootstrap percentile lower bound ( BP ). Let G ( x ) = Gn ( x ) = PF { ˆ θ n ≤ x } be the exact distribution and let ˆ G ( x ) = P ∗ { ˆ θ ∗ n ≤ x } be the bootstrap distribution. The lower 1 -α bootstrap percentile confidence bound would be ˆ G -1 ( α ), so the reported interval would be [ ˆ G -1 ( α ) , ∞ ). This was present in Efron (1979) itself, but it is seldom used because it tends to have a significant coverage bias.
```

---

## [37/44] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 188-188 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.789 |
| **Found In** | fts |
| **Chunk ID** | `ac4dde15-521a-4b34-9e62-8f500a05556d` |
| **YOUR GRADE** | ____ |

**Full Text (1303 chars):**

```
21.4 Nonparametric Confidence Interval
Often we do not know the distribution for a chosen performance measure. Alternately, we may not know the analytical way to calculate a confidence interval for a skill score.
The assumptions that underlie parametric confidence intervals are often violated. The predicted variable sometimes isn't normally distributed, and even when it is, the variance of the normal distribution might not be equal at all levels of the predictor variable.
- Page 326, Empirical Methods for Artificial Intelligence , 1995.
In these cases, the bootstrap resampling method can be used as a nonparametric method for calculating confidence intervals, nominally called bootstrap confidence intervals. The bootstrap is a simulated Monte Carlo method where samples are drawn from a fixed finite dataset with replacement and a parameter is estimated on each sample. This procedure leads to a robust estimate of the true population parameter via sampling. The bootstrap method was covered in detail in Chapter 17. We can demonstrate this with the following pseudocode.
```
statistics = [] for i in bootstraps: sample = select_sample_with_replacement(data) stat = calculate_statistic(sample) statistics.append(stat)
```

Listing 21.7: Pseudocode for estimating a statistic using the bootstrap.
```

---

## [38/44] biostatmethods

| Field | Value |
|-------|-------|
| **Pages** | 407-407 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.780 |
| **Found In** | fts |
| **Chunk ID** | `d3b420e1-498c-42c5-8742-0aa626ad5cb1` |
| **YOUR GRADE** | ____ |

**Full Text (640 chars):**

```
12.2.1 Bootstrap confidence intervals
Such bootstrap percentile intervals have the correct coverage asymptotically, under assumptions. However, a better interval can be constructed using a bias corrected confidence interval in the package boot . Let us try a bootstrapped interval for the standard deviation, since the correction does not help for the median we have been discussing so far. The bcanon function will perform nonparametric bias-corrected confiedence intervals:
sd (x)
```
[1] 12.43283 out = bcanon (x = x, nboot = 1000, theta = sd, alpha = c (.025, out $ confpoints alpha bca point [1,] 0.025 11.84043 [2,] 0.975 13.15082
```
```

---

## [39/44] Introduction to Causal Inference

| Field | Value |
|-------|-------|
| **Pages** | 79-79 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.770 |
| **Found In** | fts |
| **Chunk ID** | `cd7ebd5b-a4ec-43e7-a97f-b913cdf24b62` |
| **YOUR GRADE** | ____ |

**Full Text (1114 chars):**

```
7.9.1 Confidence Intervals
So far, in this chapter, we have only discussed point estimates for causal effects. We haven't discussed how we can gauge our uncertainty due to data sampling. We haven't discussed how to calculate confidence intervals on these estimates. This is a machine learning perspective, after all; who cares about confidence intervals... Jokes aside, because we are allowing for arbitrary machine learning models in all of the estimators we discuss, it is actually quite difficult to get valid confidence intervals.
Bootstrapping One way to get confidence intervals is to use bootstrapping. With bootstrapping, we repeat the causal effect estimation process many times, each time with a different sample (with replacement) from our data. This allows us to build an empirical distribution for the estimate. We can then compute whatever confidence interval we like from that empirical distribution. Unfortunately, bootstrapped confidence intervals are not always valid. For example, if we take a bootstrapped 95% confidence interval, it might not contain the true value (estimand) 95% of the time.
```

---

## [40/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability-Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 697-697 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.766 |
| **Found In** | fts |
| **Chunk ID** | `159c6440-16bc-4743-8259-0da72144f5a0` |
| **YOUR GRADE** | ____ |

**Full Text (939 chars):**

```
B
M´ ori-Sz´ ekely inequality, 633
Boos, D., 22, 451, 578, 579
Bootstrap, 461-492, 547
block bootstrap methods, 489-491
bootstrap Monte Carlo, 462
bootstrap-t, 479-480
circular block bootstrap (CBB), 490
confidence intervals, 426-429
See also Confidence intervals confidence intervals, 485-486
consistent bootstrap for stationary autoregression, 488-489
delta theorem for, 468
for dependent data, 487-488
autoregressive processes, 487 moving average processes, 488
distribution and the meaning of consistency, 462-464
failure of, 475-476
Glivenko-Cantelli lemma, 467
higher-order accuracy for functions of means, 472
higher-order accuracy for the t-statistic, 472
Kolmogorov and Wasserstein metrics consistency in, 464-467
m out of n bootstrap, 476-478
moving block bootstrap (MBB), 490
nonoverlapping block bootstrap (NBB), 490
optimal block length, 491-492
percentile, 471, 479-480
in regression, 483-484
residual bootstrap (RB), 484-485
```

---

## [41/44] Anirban DasGupta (auth.) - Asymptotic Theory of Statistics and Probability -Springer-Verlag New York

| Field | Value |
|-------|-------|
| **Pages** | 720-720 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.766 |
| **Found In** | fts |
| **Chunk ID** | `21af29c0-e437-4d1d-bf19-57a233027cc1` |
| **YOUR GRADE** | ____ |

**Full Text (939 chars):**

```
B
M´ ori-Sz´ ekely inequality, 633
Boos, D., 22, 451, 578, 579
Bootstrap, 461-492, 547
block bootstrap methods, 489-491
bootstrap Monte Carlo, 462
bootstrap-t, 479-480
circular block bootstrap (CBB), 490
confidence intervals, 426-429
See also Confidence intervals confidence intervals, 485-486
consistent bootstrap for stationary autoregression, 488-489
delta theorem for, 468
for dependent data, 487-488
autoregressive processes, 487 moving average processes, 488
distribution and the meaning of consistency, 462-464
failure of, 475-476
Glivenko-Cantelli lemma, 467
higher-order accuracy for functions of means, 472
higher-order accuracy for the t-statistic, 472
Kolmogorov and Wasserstein metrics consistency in, 464-467
m out of n bootstrap, 476-478
moving block bootstrap (MBB), 490
nonoverlapping block bootstrap (NBB), 490
optimal block length, 491-492
percentile, 471, 479-480
in regression, 483-484
residual bootstrap (RB), 484-485
```

---

## [42/44] statistical methods for machine learning

| Field | Value |
|-------|-------|
| **Pages** | 189-190 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.741 |
| **Found In** | fts |
| **Chunk ID** | `2d8e7f18-6df7-4f4e-aca5-884de69c789b` |
| **YOUR GRADE** | ____ |

**Full Text (1103 chars):**

```
21.4 Nonparametric Confidence Interval
```
... # calculate 95% confidence intervals (100 -alpha) alpha = 5.0
```

Listing 21.11: Define the level of confidence.
First, the desired lower percentile is calculated based on the chosen confidence interval. Then the observation at this percentile is retrieved from the sample of bootstrap statistics.
```
... # calculate lower percentile (e.g. 2.5) lower_p = alpha / 2.0 # retrieve observation at lower percentile lower = max(0.0, percentile(scores, lower_p))
```

Listing 21.12: Example of calculating the lower-bound on the confidence interval.
We do the same thing for the upper boundary of the confidence interval.
```
... # calculate upper percentile (e.g. 97.5) upper_p = (100 -alpha) + (alpha / 2.0) # retrieve observation at upper percentile upper = min(1.0, percentile(scores, upper_p))
```

Listing 21.13: Example of calculating the upper-bound on the confidence interval.
The complete example is listed below.
```
# bootstrap confidence intervals from numpy.random import seed from numpy.random import rand from numpy.random import randint
```
```
```

---

## [43/44] (Springer Texts in Statistics'',) Larry Wasserman - All of Statistics  A Concise Course in Statistical Inference (Springer Texts in Statistics)-Springer (2003)

| Field | Value |
|-------|-------|
| **Pages** | 448-448 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.722 |
| **Found In** | fts |
| **Chunk ID** | `ff26769b-e738-43f2-aac3-6325e89e8455` |
| **YOUR GRADE** | ____ |

**Full Text (1333 chars):**

```
Index
Bibliographic Remarks, 13
Binomial distribution, 26
bins, 303, 306
binwidth, 306
bivariate distribution, 31
Bonferroni method, 166
boosting, 375
bootstrap, 107
parametric, 134
Bootstrap Confidence Intervals, 110
bootstrap percentile interval, 111
bootstrap pivotal confidence, 111
Bootstrap variance estimation, 109
branching process, 398
candidate, 411
Cauchy distribution, 30
Cauchy-Schwartz inequality, 4.8 , 66 causal odds ratio, 252
causal regression function, 256
causal relative risk, 253
Central Limit Theorem (CLT), 5.8 ,
77
Chapman-Kolmogorov equations, 23.9 ,
385
Index
Chebyshev's inequality, 4.2 , 64
checking assumptions, 135
child, 265
classes, 387
classification, 349
classification rule, 349
classification trees, 360
classifier assessing error rate, 362
clique, 285
closed, 388
CLT, 77
collider, 265
comparing risk functions, 194
complete, 281, 328
composite hypothesis, 151
Computer Experiment, 16, 17
concave, 66
conditional causal effect, 255
conditional distribution, 36
conditional expectation, 54
conditional independence, 264
minimal, 287
conditional likelihood, 213
Conditional Probability, 10
conditional probability, 10, 10
conditional probability density func- tion, 37
conditional probability mass function, 36
conditioning by intervention, 274
conditioning by observation, 274
confidence band, 99
```

---

## [44/44] Causal Inference and Discovery in Python

| Field | Value |
|-------|-------|
| **Pages** | 293-293 |
| **Suggested Grade** | 0 |
| **Similarity** | 0.689 |
| **Found In** | fts |
| **Chunk ID** | `c9966f1d-b12d-4bda-a46e-33b792ced903` |
| **YOUR GRADE** | ____ |

**Full Text (1205 chars):**

```
Model selection - a simplified guide
Table 10.3 - Comparison of selected EconML estimators
SLearner, Treatment type = Categorical (can be adapted to continuous). SLearner, Confidence intervals = Only by bootstrapping. SLearner, Linear treatment assumed = No. SLearner, Multiple outcomes = Yes. SLearner, Training speed (relative to S-Learner) = 1x. TLearner, Treatment type = Categorical. TLearner, Confidence intervals = Only by bootstrapping. TLearner, Linear treatment assumed = No. TLearner, Multiple outcomes = Yes. TLearner, Training speed (relative to S-Learner) = 2x. XLearner, Treatment type = Categorical. XLearner, Confidence intervals = Only by bootstrapping. XLearner, Linear treatment assumed = No. XLearner, Multiple outcomes = Yes. XLearner, Training speed (relative to S-Learner) = 5x. DRLearner, Treatment type = Categorical. DRLearner, Confidence intervals = Only by bootstrapping. DRLearner, Linear treatment assumed = No. DRLearner, Multiple outcomes = No. DRLearner, Training speed (relative to S-Learner) = 13x. LinearDML, Treatment type = Categorical, continuous. LinearDML, Confidence intervals = Natively. LinearDML, Linear treatment assumed = Yes. LinearDML, Multiple outcomes =
```

---
