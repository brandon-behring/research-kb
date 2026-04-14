
## Double/Debiased Machine Learning for Treatment and Structural Parameters

Victor Chernozhukov † , Denis Chetverikov ‡ , Mert Demirer † Esther Duflo † , Christian Hansen § , Whitney Newey † , James Robins ⋆

† Massachusetts Institute of Technology, 50 Memorial Drive, Cambridge, MA, 02139, USA

vchern@mit.edu, mdemirer@mit.edu, duflo@mit.edu, wnewey@mit.edu

E-mail: † University of California Los Angeles, 315 Portola Plaza,

Los Angeles, CA 90095

chetverikov@econ.ucla.edu

E-mail:

§ University of Chicago, 5807 S. Woodlawn Ave., Chicago, IL 60637

E-mail:

chansen1@chicagobooth.edu

⋆ Harvard University, 677 Huntington Avenue Boston, Massachusetts 02115

E-mail:

robins@hsph.harvard.edu

Received: June 2014

Summary We revisit the classic semiparametric problem of inference on a low dimensional parameter θ 0 in the presence of high-dimensional nuisance parameters η 0 . We depart from the classical setting by allowing for η 0 to be so high-dimensional that the traditional assumptions, such as Donsker properties, that limit complexity of the parameter space for this object break down. To estimate η 0 , we consider the use of statistical or machine learning (ML) methods which are particularly well-suited to estimation in modern, very high-dimensional cases. ML methods perform well by employing regularization to reduce variance and trading off regularization bias with overfitting in practice. However, both regularization bias and overfitting in estimating η 0 cause a heavy bias in estimators of θ 0 that are obtained by naively plugging ML estimators of η 0 into estimating equations for θ 0 . This bias results in the naive estimator failing to be N -1 / 2 consistent, where N is the sample size. We show that the impact of regularization bias and overfitting on estimation of the parameter of interest θ 0 can be removed by using two simple, yet critical, ingredients: (1) using Neyman-orthogonal moments/scores that have reduced sensitivity with respect to nuisance parameters to estimate θ 0 , and (2) making use of cross-fitting which provides an efficient form of data-splitting. We call the resulting set of methods double or debiased ML (DML). We verify that DML delivers point estimators that concentrate in a N -1 / 2 -neighborhood of the true parameter values and are approximately unbiased and normally distributed, which allows construction of valid confidence statements. The generic statistical theory of DML is elementary and simultaneously relies on only weak theoretical requirements which will admit the use of a broad array of modern ML methods for estimating the nuisance parameters such as random forests, lasso, ridge, deep neural nets, boosted trees, and various hybrids and ensembles of these methods. We illustrate the general theory by applying it to provide theoretical properties of DML applied to learn the main regression parameter in a partially linear regression model, DML applied to learn the coefficient on an endogenous variable in a partially linear instrumental variables model, DML applied to learn the average treatment effect and the average treatment effect on the treated under unconfoundedness, and DML applied to learn the local average treatment effect in an instrumental variables setting. In addition to these theoretical applications, we also illustrate the use of DML in three empirical examples.

## 1. INTRODUCTION AND MOTIVATION

## 1.1. Motivation

We develop a series of simple results for obtaining rootN consistent estimation, where N is the sample size, and valid inferential statements about a low-dimensional parameter of interest, θ 0 , in the presence of a high-dimensional or 'highly complex' nuisance parameter, η 0 . The parameter of interest will typically be a causal parameter or treatment effect parameter, and we consider settings in which the nuisance parameter will be estimated using machine learning (ML) methods such as random forests, lasso or post-lasso, neural nets, boosted regression trees, and various hybrids and ensembles of these methods. These ML methods are able to handle many covariates and provide natural estimators of nuisance parameters when these parameters are highly complex. Here, highly complex formally means that the entropy of the parameter space for the nuisance parameter is increasing with the sample size in a way that moves us outside of the traditional framework considered in the classical semi-parametric literature where the complexity of the nuisance parameter space is taken to be sufficiently small. Offering a general and simple procedure for estimating and doing inference on θ 0 that is formally valid in these highly complex settings is the main contribution of this paper.

Example 1.1. (Partially Linear Regression) As a lead example, consider the following partially linear regression (PLR) model as in Robinson (1988):

$$Y = Dθ0 + g0(X) + U,
E[U | X, D] = 0,
(1.1)$$

$$D = m0(X) + V,
E[V | X] = 0,
(1.2)$$

where Y is the outcome variable, D is the policy/treatment variable of interest, vector

$$X = (X1, ..., Xp)$$

consists of other controls, and U and V are disturbances. 1 The first equation is the main equation, and θ 0 is the main regression coefficient that we would like to infer. If D is exogenous conditional on controls X , θ 0 has the interpretation of the treatment effect (TE) parameter or 'lift' parameter in business applications. The second equation keeps track of confounding, namely the dependence of the treatment variable on controls. This equation is not of interest per se but is important for characterizing and removing regularization bias. The confounding factors X affect the policy variable D via the function m 0 ( X ) and the outcome variable via the function g 0 ( X ). In many applications, the dimension p of vector X is large relative to N . To capture the feature that p is not vanishingly small relative to the sample size, modern analyses then model p as increasing with the sample size, which causes traditional assumptions that limit the complexity of the parameter space for the nuisance parameters η 0 = ( m 0 , g 0 ) to fail. ■

Regularization Bias. A naive approach to estimation of θ 0 using ML methods would be, for example, to construct a sophisticated ML estimator D ̂ θ 0 + ̂ g 0 ( X ) for learning the

1 We consider the case where D is a scalar for simplicity. Extension to the case where D is a vector of fixed, finite dimension is accomplished by introducing an equation like (1.2) for each element of the vector.

regression function Dθ 0 + g 0 ( X ). 2 Suppose, for the sake of clarity, that we randomly split the sample into two parts: a main part of size n , with observation numbers indexed by i ∈ I , and an auxiliary part of size N -n , with observations indexed by i ∈ I c . For simplicity, we take n = N/ 2 for the moment and turn to more general cases which cover unequal split-sizes, using more than one split, and achieving the same efficiency as if the full sample were used for estimating θ 0 in the formal development in Section 3. Suppose ̂ g 0 is obtained using the auxiliary sample and that, given this ̂ g 0 , the final estimate of θ 0 is obtained using the main sample:

$$bθ0 =
 1
n
X
i∈I
D2
i
−1 1
n
X
i∈I
Di(Yi −bg0(Xi)).
(1.3)$$

The estimator ̂ θ 0 will generally have a slower than 1 / √ n rate of convergence, namely,

$$|√n(bθ0 −θ0)| →P ∞.
(1.4)$$

As detailed below, the driving force behind this 'inferior' behavior is the bias in learning g 0 . Figure 1 provides a numerical illustration of this phenomenon for a naive ML estimator based on a random forest in a simple computational experiment.

To heuristically illustrate the impact of the bias in learning g 0 , we can decompose the scaled estimation error in ̂ θ 0 as

$$√n(bθ0 −θ0) =
 1
n
X
i∈I
D2
i
−1 1
√n
X
i∈I
DiUi
|
{z
}
:=a
+
 1
n
X
i∈I
D2
i
−1 1
√n
X
i∈I
Di(g0(Xi) −bg0(Xi))
|
{z
}
:=b
.$$

The first term is well-behaved under mild conditions, obeying a ❀ N (0 , ¯ Σ) for some ¯ Σ. Term b is the regularization bias term, which is not centered and diverges in general. Indeed, we have

$$b = (EDi
2)−1 1
√n
X
i∈I
m0(Xi)(g0(Xi) −bg0(Xi)) + oP (1)$$

̸

to the first order. Heuristically, b is the sum of n terms that do not have mean zero, m 0 ( X i )( g 0 ( X i ) -̂ g 0 ( X i )), divided by √ n . These terms have non-zero mean because, in high dimensional or otherwise highly complex settings, we must employ regularized estimators -such as lasso, ridge, boosting, or penalized neural nets -for informative learning to be feasible. The regularization in these estimators keeps the variance of the estimator from exploding but also necessarily induces substantive biases in the estimator ̂ g 0 of g 0 . Specifically, the rate of convergence of (the bias of) ̂ g 0 to g 0 in the root mean squared error sense will typically be n -φ g with φ g &lt; 1 / 2. Hence, we expect b to be of stochastic order √ nn -φ g → ∞ since D i is centered at m 0 ( X i ) = 0, which then implies (1.4).

Overcoming Regularization Biases using Orthogonalization. Now consider a second construction that employs an 'orthogonalized' formulation obtained by directly partialling out the effect of X from D to obtain the orthogonalized regressor V = D -m 0 ( X ). Specifically, we obtain ̂ V = D -̂ m 0 ( X ), where ̂ m 0 is an ML estimator of m 0

2 For instance, we could use lasso if we believe g 0 is well-approximated by a sparse linear combination of prespecified functions of X . In other settings, we could, for example, use iterative methods that alternate between random forests, for estimating g 0 , and least squares, for estimating θ 0 .

Figure 1 . Left Panel: Behavior of a conventional (non-orthogonal) ML estimator, ̂ θ 0 , in the partially linear model in a simple simulation experiment where we learn g 0 using a random forest. The g 0 in this experiment is a very smooth function of a small number of variables, so the experiment is seemingly favorable to the use of random forests a priori. The histogram shows the simulated distribution of the centered estimator, ̂ θ 0 -θ 0 . The estimator is badly biased, shifted much to the right relative to the true value θ 0 . The distribution of the estimator (approximated by the blue histogram) is substantively different from a normal approximation (shown by the red curve) derived under the assumption that the bias is negligible. Right Panel: Behavior of the orthogonal, DML estimator, ˇ θ 0 , in the partially linear model in a simple experiment where we learn nuisance functions using random forests. Note that the simulated data are exactly the same as those underlying left panel. The simulated distribution of the centered estimator, ˇ θ 0 -θ 0 , (given by the blue histogram) illustrates that the estimator is approximately unbiased, concentrates around θ 0 , and is well-approximated by the normal approximation obtained in Section 3 (shown by the red curve).

<!-- image -->

obtained using the auxiliary sample of observations. We are now solving an auxiliary prediction problem to estimate the conditional mean of D given X , so we are doing 'double prediction' or 'double machine learning'.

1 After partialling the effect of X out from D and obtaining a preliminary estimate of g 0 from the auxiliary sample as before, we may formulate the following 'debiased' machine learning estimator for θ 0 using the main sample of observations:

$$ˇθ0 =
 1
n
X
i∈I
bViDi
−1 1
n
X
i∈I
bVi(Yi −bg0(Xi)).3
(1.5)$$

By approximately orthogonalizing D with respect to X and approximately removing the direct effect of confounding by subtracting an estimate of g 0 , ˇ θ 0 removes the effect of regularization bias that contaminates (1.3). The formulation of ˇ θ 0 also provides direct links to both the classical econometric literature, as the estimator can clearly be interpreted as a linear instrumental variable (IV) estimator, and to the more recent literature on debiased lasso in the context where g 0 is taken to be well-approximated by a sparse

3 In Section 4, we also consider another debiased estimator, based on the partialling-out approach of Robinson (1988):

$$ˇθ0 =
 1
n
X
i∈I
bVi bVi
−1 1
n
X
i∈I
bVi(Yi −bℓ0(Xi)),
ℓ0(X) = E[Y |X].$$

linear combination of prespecified functions of X ; see, e.g., Belloni et al. (2013); Zhang and Zhang (2014); Javanmard and Montanari (2014b); van de Geer et al. (2014); Belloni et al. (2014); and Belloni et al. (2014). 4

To illustrate the benefits of the auxiliary prediction step and estimating θ 0 with ˇ θ 0 , we sketch the properties of ˇ θ 0 here. We can decompose the scaled estimation error of ˇ θ 0 into three components:

$$√n(ˇθ0 −θ0) = a∗+ b∗+ c∗.$$

The leading term, a ∗ , will satisfy

$$a∗= (EV 2)−1 1
√n
X
i∈I
ViUi ; N(0, Σ)$$

under mild conditions. The second term, b ∗ , captures the impact of regularization bias in estimating g 0 and m 0 . Specifically, we will have

$$b∗= (EV 2)−1 1
√n
X
i∈I
( bm0(Xi) −m0(Xi))(bg0(Xi) −g0(Xi)),$$

which now depends on the product of the estimation errors in ̂ m 0 and ̂ g 0 . Because this term depends only on the product of the estimation errors, it can vanish under a broad range of data-generating processes. Indeed, this term is upper-bounded by √ nn -( φ m + φ g ) , where n -φ m and n -φ g are respectively the rates of convergence of ̂ m 0 to m 0 and ̂ g 0 to g 0 ; and this upper bound can clearly vanish even though both m 0 and g 0 are estimated at relatively slow rates. Verifying that ˇ θ 0 has good properties then requires that the remainder term, c ∗ , is sufficiently well-behaved. Sample-splitting will play a key role in allowing us to guarantee that c ∗ = o P (1) under weak conditions as outlined below and discussed in detail in Section 3.

The Role of Sample Splitting in Removing Bias Induced by Overfitting. Our analysis makes use of sample-splitting which plays a key role in establishing that remainder terms, like c ∗ , vanish in probability. In the partially linear model, we have that the remainder c ∗ contains terms like

$$1
√n
X
i∈I
Vi(bg0(Xi) −g0(Xi))
(1.6)$$

that involve 1 / √ n normalized sums of products of structural unobservables from model (1.1)-(1.2) with estimation errors in learning the nuisance functions g 0 and m 0 and need to be shown to vanish in probability. The use of sample splitting allows simple and tight control of such terms. To see this, assume that observations are independent and recall that ̂ g 0 is estimated using only observations in the auxiliary sample. Then, conditioning on the auxiliary sample and recalling that E[ V i | X i ] = 0, it is easy to verify that term (1.6) has mean zero and variance of order

$$1
n
X
i∈I
(bg0(Xi) −g0(Xi))2 →P 0.$$

Thus, the term (1.6) vanishes in probability by Chebyshev's inequality.

While sample splitting allows us to deal with remainder terms such as c ∗ , its direct

4 Each of these works differs in terms of detail but can be viewed through the lens of either 'debiasing' or 'orthogonalization' to alleviate the impact of regularization bias on subsequent estimation and inference.

application does have the drawback that the estimator of the parameter of interest only makes use of the main sample which may result in a substantial loss of efficiency as we are only making use of a subset of the available data. However, we can flip the role of the main and auxiliary samples to obtain a second version of the estimator of the parameter of interest. By averaging the two resulting estimators, we may regain full efficiency. Indeed, the two estimators will be approximately independent, so simply averaging them offers an efficient procedure. We call this sample splitting procedure where we swap the roles of main and auxiliary samples to obtain multiple estimates and then average the results cross-fitting . We formally define this procedure and discuss a K -fold version of cross-fitting in Section 3.

Without sample splitting, terms such as (1.6) may not vanish and can lead to poor performance of estimators of θ 0 . The difficulty arises because model errors, such as V i , and estimation errors, such as ̂ g 0 ( X i ) -g 0 ( X i ), are generally related because the data for observation i is used in forming the estimator ̂ g 0 . The association may then lead to poor performance of an estimator of θ 0 that makes use of ̂ g 0 as a plug-in estimator for g 0 even when this estimator converges at a very favorable rate, say N -1 / 2+ ϵ .

As an artificial but illustrative example of the problems that may result from overfitting, let ̂ g 0 ( X i ) = g 0 ( X i ) + ( Y i -g 0 ( X i )) /N 1 / 2 -ϵ for any i in the sample used to form estimator ̂ g 0 , and note that the second term provides a simple model that captures overfitting of the outcome variable within the estimation sample. This estimator is excellent in terms of rates: If the U i 's and D i 's are bounded, ̂ g 0 converges uniformly to g 0 at the nearly parametric rate N -1 / 2+ ϵ . Despite this fast rate of convergence, term c ∗ now explodes if we do not use sample splitting. For example, suppose that the full sample is used to estimate both ̂ g 0 and ˇ θ 0 . A simple calculation then reveals that term c ∗ becomes

$$1
√
N
N
X
i=1
Vi(bg0(Xi) −g0(Xi)) ∝N ϵ →∞.$$

This bias due to overfitting is illustrated in the left panel of Figure 2. The histogram in the figure gives a simulated distribution for the studentized ˇ θ resulting from using the full sample and the contrived estimator ̂ g ( X i ) given above. We can see that the histogram is shifted markedly to the left demonstrating substantial bias resulting from overfitting. The right panel of Figure 2 also illustrates that this bias is completely removed by sample splitting. The results the right panel of Figure 2 make use of the two-fold crossfitting procedure discussed above using the estimator ˇ θ and the contrived estimator ̂ g ( X i ) exactly as in the left panel. The difference is that ̂ g ( X i ) is formed in one half of the sample and then ˇ θ is estimated using the other half of the sample. This procedure is then repeated swapping the roles of the two samples and the results are averaged. We can see that the substantial bias from the full sample estimator has been removed and that the spread of the histogram corresponding to the cross-fit estimator is roughly the same as that of the full sample estimator clearly illustrating the bias-reduction property and efficiency of the cross-fitting procedure.

A less contrived example that highlights the improvements brought by sample-splitting is the sparse high-dimensional instrumental variable (IV) model analyzed in Belloni et al. (2012). Specifically, they consider the IV model

$$Y = Dθ0 + ϵ$$

where E[ ϵ | D ] = 0 but instruments Z exist such that E[ D | Z ] is not a constant and

̸

Figure 2 . This figure illustrates how the bias resulting from overfitting in the estimation of nuisance functions can cause the main estimator ˇ θ 0 to be biased and how sample splitting completely eliminates this problem. Left Panel: The histogram shows the finite-sample distribution of ˇ θ 0 in the partially linear model where nuisance parameters are estimated with overfitting using the full sample, i.e. without sample splitting. The finite-sample distribution is clearly shifted to the left of the true parameter value demonstrating the substantial bias. Right Panel: The histogram shows the finite-sample distribution of ˇ θ 0 in the partially linear model where nuisance parameters are estimated with overfitting using the cross-fitting sample-splitting estimator. Here, we see that the use of sample-splitting has completely eliminated the bias induced by overfitting.

<!-- image -->

E[ ϵ | Z ] = 0. Within this model, Belloni et al. (2012) focus on the problem of estimating the optimal instrument, η 0 ( Z ) = E[ D | Z ] using lasso-type methods. If η 0 ( Z ) is approximately sparse in the sense that only s terms of the dictionary of series transformations B ( Z ) = ( B 1 ( Z ) , . . . , B p ( Z )) are needed to approximate the function accurately, Belloni et al. (2012) require that s 2 ≪ n to establish their asymptotic results when sample splitting is not used but show that these results continue to hold under the much weaker requirement that s ≪ n if one employs sample splitting. We note that this example provides a prototypical example where Neyman orthogonality holds and ML methods can usefully be adopted to aid in learning structural parameters of interest. We also note that the weaker conditions required when using sample sample-splitting would also carry over to sparsity-based estimators in the partially linear model cited above. We discuss this in more detail in Section 4.

While we find substantial appeal in using sample-splitting, one may also use empirical process methods to verify that biases introduced due to overfitting are negligible. For example, consider the problematic term in the partially linear model described previously, 1 √ n ∑ i ∈ I V i ( ̂ g 0 ( X i ) -g 0 ( X i )). This term is clearly bounded by

$$sup
g∈GN
 1
√n
X
i∈I
Vi(g(Xi) −g0(Xi))
,
(1.7)$$

where G N is the smallest class of functions that contains estimators of g 0 , ̂ g , with high probability. In conventional semiparametric statistical and econometric analysis, the complexity of G N is controlled by invoking Donsker conditions which allow verification that terms such as (1.7) vanish asymptotically. Importantly, Donsker conditions require that G N has bounded complexity, specifically a bounded entropy integral. Because of the latter property, Donsker conditions are inappropriate in settings using ML methods where the dimension of X is modeled as increasing with the sample size and estimators necessarily

live in highly complex spaces. For example, Donsker conditions rule out even the simplest linear parametric model with high-dimensional regressors with parameter space given by the Euclidean ball with the unit radius:

$$GN = {x 7→g(x) = x′θ;
θ ∈RpN : ∥θ∥⩽1}.$$

The entropy of this model, as measured by the logarithm of the covering number, grows at the rate p N . Without invoking Donsker conditions, one may still show that terms such as (1.7) vanish as long as G N 's entropy does not increase with N too rapidly. A fairly general treatment is given in Belloni et al. (2017) who provide a set of conditions under which terms like c ∗ can vanish making use of the full sample. However, these conditions on the growth of entropy could result in unnecessarily strong restrictions on model complexity, such as very strict requirements on sparsity in the context of lasso estimation as demonstrated in IV example mentioned above. Sample splitting allows one to obtain good results under very weak conditions.

Neyman Orthogonality and Moment Conditions. Now we turn to a generalization of the orthogonalization principle above. The first 'conventional' estimator ̂ θ 0 given in (1.3) can be viewed as a solution to estimating equations

$$1
n
X
i∈I
φ(W; bθ0, bg0) = 0,$$

where φ is a known 'score' function and ̂ g 0 is the estimator of the nuisance parameter g 0 . For example, in the partially linear model above, the score function is φ ( W ; θ, g ) = ( Y -θD -g ( X )) D. It is easy to see that this score function φ is sensitive to biased estimation of g . Specifically, the Gateaux derivative operator with respect to g does not vanish:

̸

$$∂gEφ(W; θ0, g0)[g −g0] ̸= 0.5$$

The proofs of the general results in Section 3 show that this term's vanishing is a key to establishing good behavior of an estimator for θ 0 .

By contrast the orthogonalized or double/debiased ML estimator ˇ θ 0 given in (1.5) solves

$$1
n
X
i∈I
ψ(W; ˇθ0, bη0) = 0,$$

where ̂ η 0 is the estimator of the nuisance parameter η 0 and ψ is an orthogonalized or debiased 'score' function that satisfies the property that the Gateaux derivative operator with respect to η vanishes when evaluated at the true parameter values:

$$∂ηEψ(W; θ0, η0)[η −η0] = 0.
(1.8)$$

We refer to property (1.8) as 'Neyman orthogonality' and to ψ as the Neyman orthogonal score function due to the fundamental contributions in Neyman (1959) and Neyman (1979), where this notion was introduced. Intuitively, the Neyman orthogonality condition means that the moment conditions used to identify θ 0 are locally insensitive to the value of the nuisance parameter which allows one to plug-in noisy estimates of these parameters without strongly violating the moment condition. In the partially linear model (1.1)-(1.2), the estimator ˇ θ 0 uses the score function ψ ( W ; θ, η ) = ( Y -Dα -g ( X ))( D -m ( X )) , with

5 See Section 2 for the definition of the Gateaux derivative operator.

the nuisance parameter being η = ( m,g ). It is easy to see that these score functions ψ are not sensitive to biased estimation of η 0 in the sense that (1.8) holds. The proofs of the general results in Section 3 show that this property and sample splitting are two generic keys that allow establishing good behavior of an estimator for θ 0 .

## 1.2. Literature Overview

Our paper builds upon two important bodies of research within the semiparametric literature. The first is the literature on obtaining √ N -consistent and asymptotically normal estimates of low-dimensional objects in the presence of high-dimensional or nonparametric nuisance functions. The second is the literature on the use of sample-splitting to relax entropy conditions. We provide links to each of these literatures in turn.

The problem we study is obviously related to the classical semiparametric estimation framework which focuses on obtaining √ N -consistent and asymptotically normal estimates for low-dimensional components with nuisance parameters estimated by conventional nonparametric estimators such as kernels or series. See, for example, the work by Levit (1975), Ibragimov and Hasminskii (1981), Bickel (1982), Robinson (1988), Newey (1990), van der Vaart (1991), Andrews (1994a), Newey (1994), Newey et al. (1998), Robins and Rotnitzky (1995), Linton (1996), Bickel et al. (1998), Chen et al. (2003), Newey et al. (2004), van der Laan and Rose (2011), and Ai and Chen (2012). Neyman orthogonality (1.8), introduced by Neyman (1959), plays a key role in optimal testing theory and adaptive estimation, semiparametric learning theory and econometrics, and, more recently, targeted learning theory. For example, Andrews (1994a), Newey (1994) and van der Vaart (1998) provide a general set of results on estimation of a low-dimensional parameter θ 0 in the presence of nuisance parameters η 0 . Andrews (1994a) uses Neyman orthogonality (1.8) and Donsker conditions to demonstrate the key equicontinuity condition

$$1
√n
X
i∈I

ψ(Wi; θ0, bη) −
Z
ψ(w; θ0, bη)dP(w) −ψ(Wi; θ0, η0)

→P 0,$$

which reduces to (1.6) in the partially linear regression model. Newey (1994) gives conditions on estimating equations and nuisance function estimators so that nuisance function estimators do not affect the limiting distribution of parameters of interest, providing a semiparametric version of Neyman orthogonality. van der Vaart (1998) discusses use of semiparametrically efficient scores to define estimators that solve estimating equations setting averages of efficient scores to zero. He also uses efficient scores to define k-step estimators, where a preliminary estimator is used to estimate the efficient score and then updating is done to further improve estimation; see also comments below on the use of sample-splitting.

There is also a related targeted maximum likelihood learning approach, introduced in Scharfstein et al. (1999) in the context of treatments effects analysis and substantially generalized by van der Laan and Rubin (2006). van der Laan and Rubin (2006) use maximum likelihood in a least favorable direction and then perform 'one-step' or 'k-step' updates using the estimated scores in an effort to better estimate the target parameter. 6 This procedure is like the least favorable direction approach in semiparametrics; see, for

6 Targeted minimum loss estimation, which shares similar properties, is also discussed in e.g. van der Laan and Rose (2011) and van der Laan (2015).

example, Severini and Wong (1992). The introduction of the likelihood introduces major benefits such as allowing simple and natural imposition of constraints inherent in the data, such as support restrictions when the outcome is binary or censored, and permitting the use of likelihood cross-validation to choose the nuisance parameter estimator. This data adaptive choice of the nuisance parameter has been dubbed the 'super learner' by van der Laan et al. (2007). In subsequent work, van der Laan and Rose (2011) emphasize the use of ML methods to estimate the nuisance parameters for use with the super learner. Much of this work, including recent work such as Luedtke and van der Laan (2016), Toth and van der Laan (2016), and Zheng et al. (2016), focuses on formal results under a Donsker condition, though the use of sample splitting to relax these conditions has also been advocated in the targeted maximum likelihood setting as discussed below.

The Donsker condition is a powerful classical condition that allows rich structures for fixed function classes G , but it is unfortunately unsuitable for high-dimensional settings. Examples of function classes where a Donsker condition holds include functions of a single variable that have total variation bounded by 1 and functions x ↦→ f ( x ) that have r &gt; dim( x ) / 2 uniformly bounded derivatives. As a further example, functions composed from function classes with VC dimensions bounded by p through a fixed number of algebraic and monotone transforms are Donsker. However, this property will no longer hold if we let dim( x ) grow to infinity with the sample size as this increase in dimension would require that the VC dimension also increases with n . More generally, Donsker conditions are easily violated once dimensions get large. A major point of departure of the present work from the classical literature on semiparametric estimation is its explicit focus on high-complexity/entropy cases. One way to analyze the problem of estimation in high-entropy cases is to see to what degree equicontinuity results continue to hold while allowing moderate growth of the complexity/entropy of G N . Examples of papers taking this approach in an approximately sparse settings are Belloni et al. (2017), Belloni et al. (2014), Belloni et al. (2016), Chernozhukov et al. (2015b), Javanmard and Montanari (2014a), van de Geer et al. (2014), and Zhang and Zhang (2014). In all of these examples, entropy growth must be limited in what may be very restrictive ways. The entropy conditions rule out the contrived overfitting example mentioned above, which does approximate realistic examples, and may otherwise place severe restrictions on the model. For example, in Belloni et al. (2010) and Belloni et al. (2012), the optimal instrument needs to be sparse of order s ≪ √ n .

A key device that we use to avoid strong entropy conditions is cross-fitting via sample splitting. Cross-fitting is a practical, efficient form of data splitting. Importantly, its use here is not simply as a device to make proofs elementary (which it does), but as a practical method to allow us to overcome the overfitting/high-complexity phenomena that commonly arise in data analysis based on highly adaptive ML methods. Our treatment builds upon the sample-splitting ideas employed in Belloni et al. (2010) and Belloni et al. (2012) who considered sample-splitting in a high-dimensional sparse optimal IV model to weaken the sparsity condition mentioned in the previous paragraph to s ≪ n . This work in turn was inspired by Angrist and Krueger (1995). We also build on Ayyagari (2010) and Robins et al. (2013), where ML methods and sample splitting were used in the estimation of a partially linear model of the effects of pollution while controlling for several covariates. We use the term 'cross-fitting' to characterize our recommended procedure, partly borrowing the jargon from Fan et al. (2012) which employed a slightly different form of sample-splitting to estimate the scale parameter in a high-dimensional sparse regression. Of course, the use of sample-splitting to relax entropy conditions has a long

history in semiparametric estimation problems. For example, Bickel (1982) considered estimating nuisance functions using a vanishing fraction of the sample, and these results were extended to sample splitting into two equal halves and discretization of the parameter space by Schick (1986). Similarly, van der Vaart (1998) uses 2-way sample splitting and discretization of the parameter space to give weak conditions for k-step estimators using the efficient scores where sample splitting is used to estimate the 'updates'; see also Hubbard et al. (2016). Robins et al. (2008) and Robins et al. (2017) use sample splitting in the construction of higher-order influence function corrections in semiparametric estimation. Some recent work in the targeted maximum likelihood literature, for example Zheng and van der Laan (2011), also notes the utility of sample splitting in the context of k-step updating, though this sample splitting approach is different from the cross-fitting approach we pursue.

Plan of the Paper. We organize the rest of the paper as follows. In Section 2, we formally define Neyman orthogonality and provide a brief discussion that synthesizes various models and frameworks that may be used to produce estimating equations satisfying this key condition. In Section 3, we carefully define DML estimators and develop their general theory. We then illustrate this general theory by applying it to provide theoretical results for using DML to estimate and do inference for key parameters in the partially linear regression model and for using DML to estimate and do inference for coefficients on endogenous variables in a partially linear instrumental variables model in Section 4. In Section 5, we provide a further illustration of the general theory by applying it to develop theoretical results for DML estimation and inference for average treatment effects and average treatment effects on the treated under unconfoundedness and for DML estimation of local average treatment effects in an IV context within the potential outcomes framework; see Imbens and Rubin (2015). Finally, we apply DML in three empirical illustrations in Section 6. In an appendix, we define additional notation and present proofs.

Notation. The symbols P and E denote probability and expectation operators with respect to a generic probability measure that describes the law of the data. If we need to signify the dependence on a probability measure P , we use P as a subscript in P P and E P . We use capital letters, such as W , to denote random elements and use the corresponding lower case letters, such as w , to denote fixed values that these random elements can take. In what follows, we use ∥ · ∥ P,q to denote the L q ( P ) norm; for example, we denote ∥ f ∥ P,q := ∥ f ( W ) ∥ P,q := (∫ | f ( w ) | q dP ( w ) ) 1 /q , where ∥ f ∥ P, ∞ stands for the essential supremum. We use x ′ to denote the transpose of a column vector x . For a differentiable map x ↦→ f ( x ), mapping R d to R k , we use ∂ x ′ f to abbreviate the partial derivatives ( ∂/∂x ′ ) f , and we correspondingly use the expression ∂ x ′ f ( x 0 ) to mean ∂ x ′ f ( x ) | x = x 0 , etc.

## 2. CONSTRUCTION OF NEYMAN ORTHOGONAL SCORE/MOMENT FUNCTIONS

Here we formally introduce the model and discuss several methods for generating orthogonal scores in a wide variety of settings, including the classical Neyman's construction. We also use this as an opportunity to synthesize some recent developments in the literature.

We are interested in the true value θ 0 of the low-dimensional target parameter θ ∈ Θ, where Θ is a non-empty measurable subset of R d θ . We assume that θ 0 satisfies the moment conditions

$$EP [ψ(W; θ0, η0)] = 0,
(2.1)$$

where ψ = ( ψ 1 , . . . , ψ d θ ) ′ is a vector of known score functions, W is a random element taking values in a measurable space ( W , A W ) with law determined by a probability measure P ∈ P N , and η 0 is the true value of the nuisance parameter η ∈ T , where T is a convex subset of some normed vector space with the norm denoted by ∥ · ∥ T . We assume that the score functions ψ j : W× Θ × T → R are measurable once we equip Θ and T with their Borel σ -fields, and we assume that a random sample ( W i ) N i =1 from the distribution of W is available for estimation and inference.

As discussed in the Introduction, we require the Neyman orthogonality condition for the score ψ . To introduce the condition, for ˜ T = { η -η 0 : η ∈ T } we define the pathwise (or the Gateaux) derivative map D r : ˜ T → R d θ ,

$$Dr[η −η0] := ∂r

EP
h
ψ(W; θ0, η0 + r(η −η0)
i
,
η ∈T,$$

for all r ∈ [0 , 1), which we assume to exist. For convenience, we also denote

$$∂ηEP ψ(W; θ0, η0)[η −η0] := D0[η −η0],
η ∈T.
(2.2)$$

Note that ψ ( W ; θ 0 , η 0 + r ( η -η 0 )) here is well-defined because for all r ∈ [0 , 1) and η ∈ T ,

$$η0 + r(η −η0) = (1 −r)η0 + rη ∈T$$

since T is a convex set. In addition, let T N ⊂ T be a nuisance realization set such that the estimators ̂ η 0 of η 0 specified below take values in this set with high probability. In practice, we typically assume that T N is a properly shrinking neighborhood of η 0 . Note that T N -η 0 is the nuisance deviation set , which contains deviations of ̂ η 0 from η 0 , ̂ η 0 -η 0 , with high probability. The Neyman orthogonality condition requires that the derivative in (2.2) vanishes for all η ∈ T N .

Definition 2.1. ( Neyman orthogonality ) The score ψ = ( ψ 1 , . . . , ψ d θ ) ′ obeys the orthogonality condition at ( θ 0 , η 0 ) with respect to the nuisance realization set T N ⊂ T if (2.1) holds and the pathwise derivative map D r [ η -η 0 ] exists for all r ∈ [0 , 1) and η ∈ T N and vanishes at r = 0 ; namely,

$$∂ηEP ψ(W; θ0, η0)[η −η0] = 0,
for all η ∈TN.
(2.3)$$

We remark here that condition (2.3) holds with T N = T when η is a finite-dimensional vector as long as ∂ η E P [ ψ j ( W ; θ 0 , η 0 )] = 0 for all j = 1 , . . . , d θ , where ∂ η E P [ ψ j ( W ; θ 0 , η 0 )] denotes the vector of partial derivatives of the function η ↦→ E P [ ψ j ( W ; θ 0 , η )] for η = η 0 .

Sometimes it will also be helpful to use an approximate Neyman orthogonality condition as opposed to the exact one given in Definition 2.1:

Definition 2.2. ( Neyman Near-Orthogonality ) The score ψ = ( ψ 1 , . . . , ψ d θ ) ′ obeys the λ N near-orthogonality condition at ( θ 0 , η 0 ) with respect to the nuisance realization set

T N ⊂ T if (2.1) holds and the pathwise derivative map D r [ η -η 0 ] exists for all r ∈ [0 , 1) and η ∈ T N and is small at r = 0 ; namely,

$$∂ηEP ψ(W; θ0, η0)[η −η0]



 ⩽λN,
for all η ∈TN,
(2.4)$$

where { λ N } N ⩾ 1 is a sequence of positive constants such that λ N = o ( N -1 / 2 ) .

## 2.2. Construction of Neyman Orthogonal Scores

If we start with a score φ that does not satisfy the orthogonality condition above, we first transform it into a score ψ that does. Here we outline several methods for doing so.

## 2.2.1. Neyman Orthogonal Scores for Likelihood and Other M-Estimation Problems with Finite-Dimensional Nuisance Parameters

First, we describe the construction used by Neyman (1959) to derive his celebrated orthogonal score and C ( α )-statistic in a maximum likelihood setting. 7 Such construction also underlies the concept of local unbiasedness in construction of optimal tests in e.g. Ferguson (1967) and was extended to non-likelihood settings by Wooldridge (1991). The discussion of Neyman's construction here draws on Chernozhukov et al. (2015a).

To describe the construction, let θ ∈ Θ ⊂ R d θ and β ∈ B ⊂ R d β , where B is a convex set, be the target and the nuisance parameters, respectively. Further, suppose that the true parameter values θ 0 and β 0 solve the optimization problem

$$max
θ∈Θ, β∈B EP [ℓ(W; θ, β)],
(2.5)$$

where ℓ ( W ; θ, β ) is a known criterion function. For example, ℓ ( W ; θ, β ) can be the loglikelihood function associated to observation W . More generally, we refer to ℓ ( W ; θ, β ) as the quasi-log-likelihood function. Then, under mild regularity conditions, θ 0 and β 0 satisfy

$$EP [∂θℓ(W; θ0, β0)] = 0,
EP [∂βℓ(W; θ0, β0)] = 0.
(2.6)$$

Note that the original score function φ ( W ; θ, β ) = ∂ θ ℓ ( W ; θ, β ) for estimating θ 0 will not generally satisfy the orthogonality condition. Now consider the new score function, which we refer to as the Neyman orthogonal score,

$$ψ(W; θ, η) = ∂θℓ(W; θ, β) −µ∂βℓ(W; θ, β),
(2.7)$$

where the nuisance parameter is

$$η = (β′, vec(µ)′)′ ∈T = B × Rdθdβ ⊂Rp,
p = dβ + dθdβ,$$

and µ is the d θ × d β orthogonalization parameter matrix whose true value µ 0 solves the equation

$$Jθβ −µJββ = 0
(2.8)$$

for

$$J =
 Jθθ
Jθβ
Jβθ
Jββ

= ∂(θ′,β′)EP
h
∂(θ′,β′)′ℓ(W; θ, β)
i
θ=θ0; β=β0.$$

7 The C ( α )-statistic, or the orthogonal score statistic, has been explicitly used for testing and estimation in high-dimensional sparse models in Belloni et al. (2015).

The true value of the nuisance parameter η is

$$η0 = (β′
0, vec(µ0)′)′;
(2.9)$$

and when J ββ is invertible, (2.8) has the unique solution,

$$µ0 = JθβJ−1
ββ .
(2.10)$$

The following lemma shows that the score ψ in (2.7) satisfies the Neyman orthogonality condition.

Lemma 2.1. ( Neyman Orthogonal Scores for Quasi-Likelihood Settings ) If (2.6) holds, J exists, and J ββ is invertible, the score ψ in (2.7) is Neyman orthogonal at ( θ 0 , η 0 ) with respect to the nuisance realization set T N = T .

Remark 2.1. (Additional nuisance parameters) Note that the orthogonal score ψ in (2.7) has nuisance parameters consisting of the elements of µ in addition to the elements of β , and Lemma 2.1 shows that Neyman orthogonality holds both with respect to β and with respect to µ . We will find that Neyman orthogonal scores in other settings, including infinite-dimensional ones, have a similar property.

Remark 2.2. (Efficiency) Note that in this example, µ 0 not only creates the necessary orthogonality but also creates the efficient score for inference on the target parameter θ when the quasi-log-likelihood function is the true (possibly conditional) log-likelihood, as demonstrated by Neyman (1959).

Example 2.1. (High-Dimensional Linear Regression) As an application of the construction above, consider the following linear predictive model:

$$Y = Dθ0 + X′β0 + U,
EP [U(X′, D)′] = 0,
(2.11)$$

$$D = X′γ0 + V,
EP [V X] = 0,
(2.12)$$

where for simplicity we assume that θ 0 is a scalar. The first equation here is the main predictive model, and the second equation only plays a role in the construction of the Neyman orthogonal scores. It is well-known that θ 0 and β 0 in this model solve the optimization problem (2.5) with

$$ℓ(W; θ, β) = −(Y −Dθ −X′β)2
2
,
θ ∈Θ = R, β ∈B = Rdβ,$$

where we denoted W = ( Y, D, X ′ ) ′ . Hence, equations (2.6) hold with

$$∂ℓθ(W; θ, β) = (Y −Dθ −X′β)D,
∂ℓβ(W; θ, β) = (Y −Dθ −X′β)X,$$

and the matrix J satisfies

$$Jθβ = −EP [DX′],
Jββ = −EP [XX′].$$

The Neyman orthogonal score is then given by

$$ψ(W; θ, η) = (Y −Dθ −X′β)(D −µX);
η = (β′, vec(µ)′)′;
ψ(W; θ0, η0) = U(D −µ0X);
µ0 = EP [DX′](EP [XX′])−1 = γ′
0.$$

$$(2.13)$$

If the vector of covariates X here is high-dimensional but the vectors of parameters β 0

and γ 0 are approximately sparse, we can use ℓ 1 -penalized least squares, ℓ 2 -boosting, or forward selection methods to estimate β 0 and γ 0 = µ ′ 0 , and hence µ 0 = ( β ′ 0 , vec( µ 0 ) ′ ) ′ ; see references cited in the Introduction. ■

If J ββ is not invertible, equation (2.8) typically has multiple solutions. In this case, it is convenient to focus on a minimal norm solution,

$$µ0 = arg min ∥µ∥such that ∥Jθβ −µJββ∥q = 0$$

for a suitably chosen norm ∥·∥ q on the space of d θ × d β matrices. With an eye on solving the empirical version of this problem, we may also consider the relaxed version of this problem,

$$µ0 = arg min ∥µ∥such that ∥Jθβ −µJββ∥q ⩽rN
(2.14)$$

for some r N &gt; 0 such that r N → 0 as N →∞ . This relaxation is also helpful when J ββ is invertible but ill-conditioned. The following lemma shows that using µ 0 in (2.14) leads to Neyman near-orthogonal scores. The proof of this lemma can be found in the Appendix.

Lemma 2.2. ( Neyman Near-Orthogonal Scores for Quasi-Likelihood Settings ) If (2.6) holds, J exists, the solution of the optimization problem (2.14) exists, and µ 0 is taken to be this solution, the score ψ defined in (2.7) is Neyman λ N near-orthogonal at ( θ 0 , η 0 ) with respect to the nuisance realization set T N = { β ∈ B : ∥ β -β 0 ∥ ∗ q ⩽ λ N /r N }× R d θ d β , where the norm ∥ · ∥ ∗ q on R d β is defined by ∥ β ∥ ∗ q = sup A ∥ Aβ ∥ with the supremum being taken over all d θ × d β matrices A such that ∥ A ∥ q ⩽ 1 .

Example 2.1. (High-Dimensional Linear Regression, Continued) In the highdimensional linear regression example above, the relaxation (2.14) is helpful when J ββ = E P [ XX ′ ] is ill-conditioned. Specifically, if one suspects that E P [ XX ′ ] is ill-conditioned, one can define µ 0 as the solution to the following optimization problem:

$$min ∥µ∥such that ∥EP [DX′] −µEP [XX′]∥∞⩽rN.
(2.15)$$

Lemma 2.2 above then shows that using this µ 0 leads to a score ψ that obeys the Neyman near-orthogonality condition. Alternatively, one can define µ 0 as the solution of the following closely related optimization problem,

$$min
µ

µEP [XX′]µ′ −µEP [DX] + rN∥µ∥1

,$$

whose solution also obeys ∥ E P [ DX ] -µ E P [ XX ′ ] ∥ ∞ ⩽ r N which follows from the first order conditions. An empirical version of either problem leads to a Lasso-type estimator of the regularized solution µ 0 ; see Javanmard and Montanari (2014a). ■

Remark 2.3. (Giving up Efficiency) Note that the regularized µ 0 in (2.14) creates the necessary near-orthogonality at the cost of giving up somewhat on efficiency of the score ψ . At the same time, regularization may generate additional robustness gains since achieving full efficiency by estimating µ 0 in (2.10) may require stronger conditions.

Remark 2.4. (Concentrating-out Approach) The approach for constructing Neyman orthogonal scores described above is closely related to the following concentrating-out approach which has been used, for example, in Newey (1994), to show Neyman orthogonality when β is infinite dimensional. For all θ ∈ Θ, let β θ be the solution of the following

optimization problem:

$$max
β∈B EP [ℓ(W; θ, β)].$$

Under mild regularity conditions, β θ satisfies

$$∂βEP [ℓ(W; θ, βθ)] = 0,
for all θ ∈Θ.
(2.16)$$

Differentiating (2.16) with respect to θ and interchanging the order of differentiation gives

$$0 = ∂θ∂βEP
h
ℓ(W; θ, βθ)
i
= ∂β∂θEP
h
ℓ(W; θ, βθ)
i
= ∂βEP
h
∂θℓ(W; θ, βθ) + [∂θβθ]
′∂βℓ(W; θ, βθ)
i
= ∂βEP
h
ψ(W; θ, β, ∂θβθ)
i
β=βθ
,$$

where we denoted

$$ψ(W; θ, β, ∂θβθ) := ∂θℓ(W; θ, β) + [∂θβθ]′ ∂βℓ(W; θ, β).$$

This vector of functions is a score with nuisance parameters η = ( β ′ , vec( ∂ θ β θ )) ′ . As before, additional nuisance parameters, ∂ θ β θ in this case, are introduced when the orthogonal score is formed. Evaluating these equations at θ 0 and β 0 , it follows from the previous equation that ψ ( W ; θ, β, ∂ θ β θ ) is orthogonal with respect to β and from E P [ ∂ β ℓ ( W ; θ 0 , β 0 )] = 0 that we have orthogonality with respect to ∂ θ β θ . Thus, maximizing the expected objective function with respect to the nuisance parameters, plugging that maximum back in, and differentiating with respect to the parameters of interest produces an orthogonal moment condition. See also Section 2.2.3.

## 2.2.2. Neyman Orthogonal Scores in GMM Problems

The construction in the previous section gives a Neyman orthogonal score whenever the moment conditions (2.6) hold, and, as discussed in Remark 2.2, the resulting score is efficient as long as ℓ ( W ; θ, β ) is the log-likelihood function. The question, however, remains about constructing the efficient score when ℓ ( W ; θ, β ) is not necessarily a loglikelihood function. In this section, we answer this question and describe a GMM-based method of constructing an efficient and Neyman orthogonal score in this more general case. The discussion here is related to Lee (2005), Bera et al. (2010), and Chernozhukov et al. (2015b).

Since GMM does not require that the equations (2.6) are obtained from the first-order conditions of the optimization problem (2.5), we use a different notation for the moment conditions. Specifically, we consider parameters θ ∈ Θ ⊂ R d θ and β ∈ B ⊂ R d β , where B is a convex set, whose true values, θ 0 and β 0 , solve the moment conditions

$$EP [m(W; θ0, β0)] = 0,
(2.17)$$

where m : W× Θ ×B → R d m is a known vector-valued function, and d m ⩾ d θ + d β is the number of moment conditions. In this case, a Neyman orthogonal score function is

$$ψ(W; θ, η) = µm(W; θ, β),
(2.18)$$

where the nuisance parameter is

$$η = (β′, vec(µ)′)′ ∈T = B × Rdθdm ⊂Rp,
p = dβ + dθdm,$$

and µ is the d θ × d m orthogonalization parameter matrix whose true value is

$$µ0 =

A′Ω−1 −A′Ω−1Gβ(G′
βΩ−1Gβ)−1G′
βΩ−1
,$$

where

$$Gγ = ∂γ′EP [m(W; θ, β)]

γ=γ0
=
h
∂θ′EP [m(W; θ, β)], ∂β′EP [m(W; θ, β)]
i
γ=γ0 =:
h
Gθ, Gβ
i
,$$

for γ = ( θ ′ , β ′ ) ′ and γ 0 = ( θ ′ 0 , β ′ 0 ) ′ , A is a d m × d θ moment selection matrix, Ω is a d m × d m positive definite weighting matrix, and both A and Ω can be chosen arbitrarily. Note that setting

$$A = Gθ and Ω= VarP (m(W; θ0, β0)]) = EP
h
m(W; θ0, β0)m(W; θ0, β0)′i$$

leads to the efficient score in the sense of yielding an estimator of θ 0 having the smallest variance in the class of GMM estimators (Hansen, 1982), and, in fact, to the semiparametrically efficient score; see Levit (1975), Nevelson (1977), and Chamberlain (1987). Let η 0 = ( β ′ 0 , vec( µ 0 ) ′ ) ′ be the true value of the nuisance parameter η = ( β ′ , vec( µ ) ′ ) ′ . The following lemma shows that the score ψ in (2.18) satisfies the Neyman orthogonality condition.

Lemma 2.3. ( Neyman Orthogonal Scores for GMM Settings ) If (2.17) holds, G γ exists, and Ω is invertible, the score ψ in (2.18) is Neyman orthogonal at ( θ 0 , η 0 ) with respect to the nuisance realization set T N = T .

As in the quasi-likelihood case, we can also consider near-orthogonal scores. Specifically, note that one of the orthogonality conditions that the score ψ in (2.18) has to satisfy is that µ 0 G β = 0, which can be rewritten as

$$A′Ω−1/2(I −L(L′L)−1L′)L = 0,
where L = Ω−1/2Gβ$$

Here, the part A ′ Ω -1 / 2 L ( L ′ L ) -1 L ′ can be expressed as γ 0 L ′ , where γ 0 = A ′ Ω -1 / 2 L ( L ′ L ) -1 solves the optimization problem

$$min ∥γ∥o such that ∥A′Ω−1/2L −γL′L∥∞= 0,$$

for a suitably chosen norm ∥ · ∥ o . When L ′ L is close to being singular, this problem can be relaxed:

$$min ∥γ∥o such that ∥A′Ω−1/2L −γL′L∥∞⩽rN.
(2.19)$$

This relaxation leads to Neyman near-orthogonal scores:

Lemma 2.4. ( Neyman Near-Orthogonal Scores for GMM settings ) In the set-up above, with γ 0 denoting the solution of (2.19), we have for µ 0 := A ′ Ω -1 -γ 0 L ′ Ω -1 / 2 and η 0 = ( β ′ 0 , vec ( µ 0 ) ′ ) ′ that ψ defined in (2.18) is the Neyman λ N near-orthogonal score at ( θ 0 , η 0 ) with respect to the nuisance realization set T N = { β ∈ B : ∥ β -β 0 ∥ 1 ⩽ λ N /r N } × R d θ d m .

2.2.3. Neyman Orthogonal Scores for Likelihood and Other M-Estimation Problems with Infinite-Dimensional Nuisance Parameters

Here we show that the concentrating-out approach described in Remark 2.4 for the case of finite-dimensional nuisance parameters can be extended to the case of infinitedimensional nuisance parameters. Let ℓ ( W ; θ, β ) be a known criterion function, where θ and β are the target and the nuisance parameters taking values in Θ and B , respectively and assume that the true values of these parameters, θ 0 and β 0 , solve the optimization problem (2.5). The function ℓ ( W ; θ, β ) is analogous to that discussed above but now, instead of assuming that B is a (convex) subset of a finite-dimensional space, we assume that B is some (convex) set of functions, so that β is the functional nuisance parameter. For example, ℓ ( W ; θ, β ) could be a semiparametric log-likelihood where β is the nonparametric part of the model. More generally, ℓ ( W ; θ, β ) could be some other criterion function such as the negative of a squared residual. Also let

$$βθ = arg max
β∈B EP [ℓ(W; θ, β)]
(2.20)$$

be the 'concentrated-out' nonparametric part of the model. Note that β θ is a functionvalued function. Now consider the score function

$$ψ(W; θ, η) = dℓ(W; θ, η(θ))
dθ
,
(2.21)$$

where the nuisance parameter is η : Θ →B , and its true value η 0 is given by

$$η0(θ) = βθ,
for all θ ∈Θ.$$

Here, the symbol d/dθ denotes the full derivative with respect to θ , so that we differentiate with respect to both θ arguments in ℓ ( W ; θ, η ( θ )). The following lemma shows that the score ψ in (2.21) satisfies the Neyman orthogonality condition.

Lemma 2.5. ( Neyman Orthogonal Scores via Concentrating-Out Approach ) Suppose that (2.5) holds, and let T be a convex set of functions mapping Θ into B such that η 0 ∈ T . Also, suppose that for each η ∈ T , the function θ ↦→ ℓ ( W ; θ, η ( θ )) is continuously differentiable almost surely. Then, under mild regularity conditions, the score ψ in (2.21) is Neyman orthogonal at ( θ 0 , η 0 ) with respect to the nuisance realization set T N = T .

As an example, consider the partially linear model from the Introduction. Let

$$ℓ(W; θ, β) = −1
2(Y −Dθ −β(X))2,$$

and let B be the set of functions of X with finite mean square. Then

$$(θ0, β0) = arg
max
θ∈Θ,β∈B EP [ℓ(W; θ, β)]$$

$$βθ(X) = EP [Y −Dθ|X],
θ ∈Θ.$$

Hence, (2.21) gives the following Neyman orthogonal score:

$$ψ(W; θ, βθ) = −1
2
d{Y −Dθ −EP [Y −Dθ|X]}2
dθ
= (D −EP [D|X]) × (Y −EP [Y |X] −(D −EP [D|X])θ)
= (D −m0(X)) × (Y −Dθ −g0(X)),$$

and

which corresponds to the estimator θ 0 described in the Introduction in (1.5).

It is important to note that the concentrating-out approach described here gives a Neyman orthogonal score without requiring that ℓ ( W ; θ, β ) is the log-likelihood function. Except for the technical conditions needed to ensure the existence of derivatives and their interchangeability, the only condition that is required is that θ 0 and β 0 solve the optimization problem (2.5). If ℓ ( W ; θ, β ) is the log-likelihood function, however, it follows from Newey (1994), p. 1359, that the concentrating-out approach actually yields the efficient score. An alternative, but closely related, approach to derive the efficient score in the likelihood setting would be to apply Neyman's construction described above for a one-dimensional least favorable parametric sub-model; see Severini and Wong (1992) and Chap. 25 of van der Vaart (1998).

Remark 2.5. (Generating Orthogonal Scores by Varying B ) When we calculate the 'concentrated-out' nonparametric part β θ , we can use some other set of functions Υ instead of B on the right-hand side of (2.20):

$$βθ = arg max
β∈Υ EP [ℓ(W; θ, β)].$$

By replacing B by Υ we can generate a different Neyman orthogonal score. Of course, this replacement may also change the true value θ 0 of the parameter of interest, which is an important consideration for the selection of Υ. For example, consider the partially linear model and assume that X has two components, X 1 and X 2 . Now, consider what would happen if we replaced B , which is the set of functions of X with finite mean square, by the set of functions Υ that is the mean square closure of functions that are additive in X 1 and X 2 :

$$Υ = {h(X1) + h(X2)}.$$

Let ¯ E P denote the least squares projection on Υ. Then, applying the previous calculation with ¯ E P replacing E P gives

$$ψ(W; θ, βθ) = (D −¯EP [D|X]) × (Y −¯EP [Y |X] + (D −¯EP [D|X])θ),$$

which provides an orthogonal score based on additive function of X 1 and X 2 . Here, it is important to note that the solution to E P [ ψ ( W,θ,β θ )] = 0 will be the true θ 0 only when the true function of X in the partially linear model is additive. More generally, the solution of the moment condition would be the coefficient of D in the least squares projection of Y on functions of the form Dθ + h 1 ( X 1 ) + h 1 ( X 2 ) . Note though that the corresponding score is orthogonal by virtue of additivity being imposed in the estimation of ¯ E P [ Y | X ] and ¯ E P [ D | X ] .

2.2.4. Neyman Orthogonal Scores for Conditional Moment Restriction Problems with Infinite-Dimensional Nuisance Parameters

Next we consider the conditional moment restrictions framework studied in Chamberlain (1992). To define the framework, let W , R , and Z be random vectors taking values in W ⊂ R d w , R ⊂ R d r , and Z ⊂ R d z , respectively. Assume that Z is a sub-vector of R and R is a sub-vector of W , so that d z ⩽ d r ⩽ d w . Also, let θ ∈ Θ ⊂ R d θ be a finitedimensional parameter whose true value θ 0 is of interest, and let h be a vector-valued functional nuisance parameter taking values in a convex set of functions H mapping Z to R d h , with the true value of h being h 0 . The conditional moment restrictions framework

assumes that θ 0 and h 0 satisfy the moment conditions

$$EP [m(W; θ0, h0(Z)) | R] = 0,
(2.22)$$

where m : W × Θ × R d h → R d m is a known vector-valued function. This framework is of interest because it covers a rich variety of models without having to explicitly rely on the likelihood formulation.

To build a Neyman orthogonal score ψ ( W ; θ, η ) for estimating θ 0 , consider the matrixvalued functional parameter µ : R→ R d θ × d m whose true value is given by

$$µ0(R) = A(R)′Ω(R)−1 −G(Z)Γ(R)′Ω(R)−1,
(2.23)$$

where the moment selection matrix-valued function A : R → R d m × d θ and the weighting positive definite matrix-valued function Ω: R → R d m × d m can be chosen arbitrarily, and the matrix-valued functions Γ: R→ R d m × d θ and G : Z → R d θ × d m are given by

$$Γ(R) = ∂v′EP
h
m(W; θ0, v) | R
i
v=h0(Z), and
(2.24)$$

$$G(Z) = EP
h
A(R)′Ω(R)−1Γ(R) | Z
i
×

EP [Γ(R)′Ω(R)−1Γ(R) | Z]
−1
.
(2.25)$$

Note that µ 0 in (2.23) is well-defined even though the right-hand side of (2.23) contains both R and Z since Z is a sub-vector of R . Then a Neyman orthogonal score is

$$ψ(W; θ, η) = µ(R)m(W; θ, h(Z)),
(2.26)$$

where the nuisance parameter is

$$η = (µ, h) ∈T = L1(R; Rdθ×dm) × H.$$

Here, L 1 ( R ; R d θ × d m ) is the vector space of matrix-valued functions f : R → R d θ × d m satisfying E P [ ∥ f ( R ) ∥ ] &lt; ∞ . Also, note that even though the matrix-valued functions A and Ω can be chosen arbitrarily, setting

$$A(R) = ∂θ′EP
h
m(W; θ, h0(Z)) | R
i
θ=θ0 and
(2.27)$$

$$Ω(R) = EP
h
m(W; θ0, h0(Z))m(W; θ0, h0(Z))′ | R
i
(2.28)$$

leads to an asymptotic variance equal to the semiparametric bound of Chamberlain (1992). Let η 0 = ( µ 0 , h 0 ) be the true value of the nuisance parameter η = ( µ, h ). The following lemma shows that the score ψ in (2.26) satisfies the Neyman orthogonality condition.

Lemma 2.6. ( Neyman Orthogonal Scores for Conditional Moment Settings ) Suppose that (a) (2.22) holds, (b) the matrices E P [ ∥ Γ( R ) ∥ 4 ] , E P [ ∥ G ( Z ) ∥ 4 ] , E P [ ∥ A ( R ) ∥ 2 ] , and E P [ ∥ Ω( R ) ∥ -2 ] are finite, and (c) for all h ∈ H , there exists a constant C h &gt; 0 such that P P (E P [ ∥ m ( W ; θ 0 , h ( Z )) ∥ | R ] ⩽ C h ) = 1 . Then the score ψ in (2.26) is Neyman orthogonal at ( θ 0 , η 0 ) with respect to the nuisance realization set T N = T .

As an application of the conditional moment restrictions framework, let us derive Neyman orthogonal scores in the partially linear regression example using this framework. The partially linear regression model (1.1) is equivalent to

$$EP [Y −Dθ0 −g0(X) | X, D] = 0,$$

which can be written in the form of the conditional moment restrictions framework (2.22) with W = ( Y, D, X ′ ) ′ , R = ( D,X ′ ) ′ , Z = X , h ( Z ) = g ( X ), and m ( W ; θ, v ) = Y -Dθ -v . Hence, using (2.27) and (2.28) and denoting σ ( D,X ) 2 = E P [ U 2 | D,X ] for U = Y -Dθ 0 -g 0 ( X ), we can take

$$A(R) = −D,
Ω(R) = EP [U 2 | D, X] = σ(D, X)2.$$

With this choice of A ( R ) and Ω( R ), we have

$$Γ(R) = −1,
G(Z) =

EP
h
D
σ(D, X)2 | X
i
×

EP
h
1
σ(D, X)2 | X
i−1
,$$

and so (2.23) and (2.26) give

$$ψ(W; θ, η0)
=
1
σ(D, X)2

D −EP
h
D
σ(D, X)2 | X
i.
EP
h
1
σ(D, X)2 | X
i
×

Y −Dθ −g0(X)

.$$

By construction, the score ψ above is efficient and Neyman orthogonal. Note, however, that using this score would require estimating the heteroscedasticity function σ ( D,X ) 2 which would requires the imposition of some additional smoothness assumptions over this conditional variance function. Instead, if are willing to give up on efficiency to gain some robustness, we can take

$$A(R) = −D,
Ω(R) = 1;$$

$$Γ(R) = −1,
G(Z) = EP [D | X].$$

in which case we have

(2.23) and (2.26) then give

$$ψ(W; θ, η0) = (D −EP [D | X]) × (Y −Dθ −g0(X))
= (D −m0(X)) × (Y −Dθ −g0(X)).$$

This score ψ is Neyman orthogonal and corresponds to the estimator of θ 0 described in the Introduction in (1.5). Note, however, that this score ψ is efficient only if σ ( X,D ) is a constant.

## 2.2.5. Neyman Orthogonal Scores and Influence Functions

Neyman orthogonality is a joint property of the score ψ ( W ; θ, η ), the true parameter value η 0 , the parameter set T , and the distribution of W . It is not determined by any particular model for the parameter θ . Nevertheless, it is possible to use semiparametric efficiency calculations to construct the orthogonal score from the original score as in Chernozhukov et al. (2016). Specifically, an orthogonal score can be constructed by adding to the original score the influence function adjustment for estimation of the nuisance functions that is analyzed in Newey (1994). The resulting orthogonal score will be the influence function of the limit of the average of the original score.

To explain, consider the original score φ ( W ; θ, β ), where β is some function, and let ̂ β 0 be a nonparametric estimator of β 0 , the true value of β . Here, β is implicitly allowed to depend on θ , though we suppress that dependence for notational convenience. The

corresponding orthogonal score can be formed when there is ϕ ( W ; θ, η ) such that

$$Z
φ(w; θ0, bβ0)dP(w) = 1
n
n
X
i=1
ϕ(Wi; θ0, η0) + oP (n−1/2),
(2.29)$$

where η is a vector of nuisance functions that includes β . ϕ ( W ; θ, η ) is an adjustment for the presence of the estimated function ̂ β 0 in the original score φ ( W ; θ, β ). The decomposition (2.29) typically holds when ̂ β is either a kernel or a series estimator with a suitably chosen tuning parameter. The Neyman orthogonal score is given by

$$ψ(W; θ, η) = φ(W; θ, β) + ϕ(W; θ, η).
(2.30)$$

Here ψ ( W ; θ 0 , η 0 ) is the influence function of the limit of n -1 ∑ n i =1 φ ( W i ; θ 0 , ̂ β 0 ), as analyzed in Newey (1994), with the restriction E P [ ψ ( W ; θ 0 , η 0 )] = 0 identifying θ 0 .

The form of the adjustment term ϕ ( W ; θ, η ) depends on the estimator ̂ β 0 and, of course, on the form of φ ( W ; θ, β ) . Such adjustment terms have been derived for various ̂ β 0 by Newey (1994). Also Ichimura and Newey (2015) show how the adjustment term can be computed from the limit of a certain derivative. Any of these results can be applied to a particular starting score φ ( W ; θ, β ) and estimator ̂ β 0 to obtain an orthogonal score.

For example, consider again the partially linear model with the original score

$$φ(W; θ, β) = D(Y −Dθ −g0(X)).$$

Here ̂ β 0 = ̂ g 0 is a nonparametric regression estimator. From Newey (1994), we know that we obtain the influence function adjustment by taking the conditional expectation of the derivative of the score with respect to g 0 ( x ) (obtaining -m 0 ( X ) = -E P [ D | X ]) and multiplying the result by the nonparametric residual to obtain

$$ϕ(W, θ, η) = −m0(X){Y −Dθ −β(X, θ)}.$$

The corresponding orthogonal score is then simply

$$ψ(W; θ, η) = {D −m0(X)}{Y −Dθ −β(X, θ)},
β0(X, θ) = EP [Y −Dθ|X],
m0(X) = EP [D|X],$$

illustrating that an orthogonal score for the partially linear model can be derived from an influence function adjustment.

Influence functions have been used to estimate functionals of nonparametric estimators by Hasminskii and Ibragimov (1978) and Bickel and Ritov (1988). Newey et al. (1998, 2004) showed that n -1 / 2 ∑ n i =1 ψ ( W i ; θ 0 , ̂ η 0 ) from equation (2.30) will have a second order remainder in ̂ η 0 , which is the key asymptotic property of orthogonal scores. Orthogonality of influence functions in semiparametric models follows from van der Vaart (1991), as shown for higher order counterparts in Robins et al. (2008, 2017). Chernozhukov et al. (2016) point out that in general an orthogonal score can be constructed from an original score and nonparametric estimator ̂ β 0 by adding to the original score the adjustment term for estimation of β 0 as described above. This construction provides a way of obtaining an orthogonal score from any initial score φ ( W ; θ, β ) and nonparametric estimator ̂ β 0 .

## 3. DML: POST-REGULARIZED INFERENCE BASED ON NEYMAN-ORTHOGONAL ESTIMATING EQUATIONS

## 3.1. Definition of DML and Its Basic Properties

We assume that we have a sample ( W i ) N i =1 , modeled as i.i.d. copies of W , whose law is determined by the probability measure P on W . Estimation will be carried out using the finite-sample analog of the estimating equations (2.1).

We assume that the true value η 0 of the nuisance parameter η can be estimated by ̂ η 0 using a part of the data ( W i ) N i =1 . Different structured assumptions on η 0 allow us to use different machine-learning tools for estimating η 0 . For instance,

1. approximate sparsity for η 0 with respect to some dictionary calls for the use of forward selection, lasso, post-lasso, ℓ 2 -boosting, or some other sparsity-based technique;
2. well-approximability of η 0 by trees calls for the use of regression trees and random forests;
3. well-approximability of η 0 by sparse neural and deep neural nets calls for the use of ℓ 1 -penalized neural and deep neural networks;
4. well-approximability of η 0 by at least one model mentioned in 1)-3) above calls for the use of an ensemble/aggregated method over the estimation methods mentioned in 1)-3).

There are performance guarantees for most of these ML methods that make it possible to satisfy the conditions stated below. Ensemble and aggregation methods ensure that the performance guarantee is approximately no worse than the performance of the best method.

We assume that N is divisible by K in order to simplify the notation. The following algorithm defines the simple cross-fitted DML as outlined in the Introduction.

Definition 3.1. ( DML1 ) 1) Take a K-fold random partition ( I k ) K k =1 of observation indices [ N ] = { 1 , ..., N } such that the size of each fold I k is n = N/K . Also, for each k ∈ [ K ] = { 1 , . . . , K } , define I c k := { 1 , ..., N } \ I k . 2) For each k ∈ [ K ] , construct a ML estimator

$$bη0,k = bη0((Wi)i∈Ic
k)$$

of η 0 , where ̂ η 0 ,k is a random element in T , and where randomness depends only on the subset of data indexed by I c k . 3) For each k ∈ [ K ] , construct the estimator ˇ θ 0 ,k as the solution of the following equation:

$$En,k[ψ(W; ˇθ0,k, bη0,k] = 0,
(3.1)$$

where ψ is the Neyman orthogonal score, and E n,k is the empirical expectation over the k -th fold of the data; that is, E n,k [ ψ ( W )] = n -1 ∑ i ∈ I k ψ ( W i ) . If achievement of exact 0 is not possible, define the estimator ˇ θ 0 ,k of θ 0 as an approximate ϵ N -solution:

$$En,k[ψ(W; ˇθ0,k, bη0,k)]



 ⩽inf
θ∈Θ



En,k[ψ(W; θ, bη0,k)]



 + ϵN,
ϵN = o(δNN −1/2),
(3.2)$$

where ( δ N ) N ⩾ 1 is some sequence of positive constants converging to zero. 4) Aggregate

the estimators:

$$˜θ0 = 1
K
K
X
k=1
ˇθ0,k.
(3.3)$$

This approach generalizes the 50-50 cross-fitting method mentioned in the Introduction. We now define a variation of this basic cross-fitting approach that may behave better in small samples.

Definition 3.2. ( DML2 ) 1) Take a K-fold random partition ( I k ) K k =1 of observation indices [ N ] = { 1 , ..., N } such that the size of each fold I k is n = N/K . Also, for each k ∈ [ K ] = { 1 , . . . , K } , define I c k := { 1 , ..., N } \ I k . 2) For each k ∈ [ K ] , construct a ML estimator

$$bη0,k = bη0((Wi)i∈Ic
k)$$

of η 0 , where ̂ η 0 ,k is a random element in T , and where randomness depends only on the subset of data indexed by I c k . 3) Construct the estimator ˜ θ 0 as the solution to the following equation:

$$1
K
K
X
k=1
En,k[ψ(W; ˜θ0, bη0,k)] = 0,
(3.4)$$

where ψ is the Neyman orthogonal score, and E n,k is the empirical expectation over the k -th fold of the data; that is, E n,k [ ψ ( W )] = n -1 ∑ i ∈ I k ψ ( W i ) . If achievement of exact 0 is not possible define the estimator ˜ θ 0 of θ 0 as an approximate ϵ N -solution:

$$1
K
K
X
k=1
En,k[ψ(W; ˜θ0, bη0,k)]]



 ⩽inf
θ∈Θ



 1
K
K
X
k=1
En,k[ψ(W; θ0, bη0,k)]]



 + ϵN,
(3.5)$$

for ϵ N = o ( δ N N -1 / 2 ) , where ( δ N ) N ⩾ 1 is some sequence of positive constants converging to zero.

Remark 3.1. (Recommendations) The choice of K has no asymptotic impact under our conditions but, of course, the choice of K may matter in small samples. Intuitively, larger values of K provide more observations in I c k from which to estimate the high-dimensional nuisance functions, which seems to be the more difficult part of the problem. We have found moderate values of K , such as 4 or 5, to work better than K = 2 in a variety of empirical examples and in simulations. Moreover, we generally recommend DML2 over DML1 though in some problems like estimation of ATE in the interactive model, which we discuss later, there is no difference between the two approaches. In most other problems, DML2 is better behaved since the pooled empirical Jacobian for the equation in (3.4) exhibits more stable behavior than the separate empirical Jacobians for the equation in (3.1).

## 3.2. Moment Condition Models with Linear Scores

We first consider the case of linear scores, where

$$ψ(w; θ, η) = ψa(w; η)θ + ψb(w; η),
for all w ∈W, θ ∈Θ, η ∈T.
(3.6)$$

Let c 0 &gt; 0, c 1 &gt; 0, s &gt; 0, q &gt; 2 be some finite constants such that c 0 ⩽ c 1 ; and let { δ N } N ⩾ 1 and { ∆ N } N ⩾ 1 be some sequences of positive constants converging to zero such that δ N ⩾ N -1 / 2 . Also, let K ⩾ 2 be some fixed integer, and let {P N } N ⩾ 1 be some sequence of sets of probability distributions P of W on W .

Assumption 3.1. (Linear Scores with Approximate Neyman Orthogonality) For all N ⩾ 3 and P ∈ P N , the following conditions hold. (a) The true parameter value θ 0 obeys (2.1). (b) The score ψ is linear in the sense of (3.6). (c) The map η ↦→ E P [ ψ ( W ; θ, η )] is twice continuously Gateaux-differentiable on T . (d) The score ψ obeys the Neyman orthogonality or, more generally, the Neyman λ N near-orthogonality condition at ( θ 0 , η 0 ) with respect to the nuisance realization set T N ⊂ T for

$$λN := sup
η∈TN



∂ηEP ψ(W; θ0, η0)[η −η0]



 ⩽δNN −1/2.$$

(e) The identification condition holds; namely, the singular values of the matrix

$$J0 := EP [ψa(W; η0)]$$

are between c 0 and c 1 .

Assumption 3.1 requires scores to be Neyman orthogonal or near-orthogonal and imposes mild smoothness requirements as well as the canonical identification condition.

Assumption 3.2. (Score Regularity and Quality of Nuisance Parameter Estimators) For all N ⩾ 3 and P ∈ P N , the following conditions hold. (a) Given a random subset I of [ N ] of size n = N/K , the nuisance parameter estimator ̂ η 0 = ̂ η 0 (( W i ) i ∈ I c ) belongs to the realization set T N with probability at least 1 -∆ N , where T N contains η 0 and is constrained by the next conditions. (b) The moment conditions hold:

$$mN := sup
η∈TN
(EP [∥ψ(W; θ0, η)∥q])1/q ⩽c1$$

$$m′
N := sup
η∈TN
(EP [∥ψa(W; η)∥q])1/q ⩽c1$$

$$p
TN
(EP [∥ψ(W; θ0, η)∥q])1/q ⩽c1,
p
TN
(EP [∥ψa(W; η)∥q])1/q ⩽c1.$$

(c) The following conditions on the statistical rates r N , r ′ N , and λ ′ N hold:

$$a(W; η)]
EP [ψa$$

$$rN := sup
η∈TN
∥EP [ψa(W; η)] −EP [ψa(W; η0)]∥⩽δN,
r′
N := sup
η∈TN
(EP [∥ψ(W; θ0, η) −ψ(W; θ0, η0)∥2])1/2 ⩽δN,
λ′
N :=
sup
r∈(0,1),η∈TN
∥∂2
rEP [ψ(W; θ0, η0 + r(η −η0))]∥⩽δN/
√$$

$$N.$$

(d) The variance of the score ψ is non-degenerate: All eigenvalues of the matrix

$$EP [ψ(W; θ0, η0)ψ(W; θ0, η0)′]$$

are bounded from below by c 0 .

Assumptions 3.2(a)-(c) state that the estimator of the nuisance parameter belongs to the realization set T N ⊂ T , which is a shrinking neighborhood of η 0 , which contracts around η 0 with the rate determined by the 'statistical' rates r N , r ′ N , and λ ′ N . These rates are not given in terms of the norm ∥ · ∥ T on T , but rather are the intrinsic rates

that are most connected to the statistical problem at hand. However, in smooth problems, as discussed below this translates, in the worst cases, to the crude requirement that the nuisance parameters are estimated at the rate o ( N -1 / 4 ).

The conditions in Assumption 3.2 embody refined requirements on the quality of nuisance parameter estimators. In many applications, where ( θ, η ) ↦→ ψ ( W ; θ, η ) is smooth, we can bound

$$rN ≲εN,
r′
N ≲εN,
λ′
N ≲ε2
N,
(3.7)$$

where ε N is the upper bound on the rate of convergence of ̂ η 0 to η 0 with respect to the norm ∥ · ∥ T = ∥ · ∥ P, 2 :

$$∥bη0 −η∥T ≲εN.$$

Note that T N can be chosen as the set of η that is within a neighborhood of size ε N of η 0 , possibly with other restrictions, in this case. If only (3.7) holds, Assumption 3.2, particularly λ ′ N = o ( N -1 / 2 ), imposes the (crude) rate requirement

$$εN = o(N −1/4).
(3.8)$$

This rate is achievable for many ML methods under structured assumptions on the nuisance parameters. See, among many others, Bickel et al. (2009), B¨ uhlmann and van de Geer (2011), Belloni et al. (2011), Belloni and Chernozhukov (2011), Belloni et al. (2012), and Belloni and Chernozhukov (2013) for ℓ 1 -penalized and related methods in a variety of sparse models; Kozbur (2016) for forward selection in sparse models; Luo and Spindler (2016) for L 2 -boosting in sparse linear models; Wager and Walther (2016) for concentration results for a class of regression trees and random forests; and Chen and White (1999) for a class of neural nets.

However, the presented conditions allow for more refined statements than (3.8). We note that many important structured problems - such as estimation of parameters in partially linear regression models, estimation of parameters in partially linear structural equation models, and estimation of average treatment effects under unconfoundedness - are such that some cross-derivatives vanish, allowing more refined requirements than (3.8). This feature allows us to require much finer conditions on the quality of the nuisance parameter estimators than the crude bound (3.8). For example, in many problems

$$λ′
N = 0,
(3.9)$$

because the second derivatives vanish,

$$∂2
rEP [ψ(W; θ0, η0 + r(η −η0))] = 0.$$

This occurs in the following important examples:

1. the optimal instrument problem; see Belloni et al. (2012).
2. the partially linear regression model when m 0 ( X ) = 0 or is otherwise known; see Section 4.
3. the treatment effect examples when the propensity score is known, which includes randomized control trials as an important special case; see Section 5.

If both (3.7) and (3.9) hold, Assumption 3.2, particularly r N = o (1) and r ′ N = o (1), imposes the weakest possible rate requirement:

$$εN = o(1).$$

We note that similar refined rates have appeared in the context of estimation of treatment effects in high-dimensional settings under sparsity; see Farrell (2015) and Athey et al. (2016) and related discussion in Remark 5.2. Our refined rate results complement this work by applying to a broad class of estimation contexts, including estimation of average treatment effects, and to a broad set of ML estimators.

Theorem 3.1. ( Properties of the DML ) Suppose that Assumptions 3.1 and 3.2 hold. In addition, suppose that δ N ⩾ N -1 / 2 for all N ⩾ 1 . Then the DML1 and DML2 estimators ˜ θ 0 concentrate in a 1 / √ N neighborhood of θ 0 and are approximately linear and centered Gaussian:

$$√
Nσ−1(˜θ0 −θ0) =
1
√
N
N
X
i=1
¯ψ(Wi) + OP (ρN) ; N(0, Id),
(3.10)$$

uniformly over P ∈ P N , where the size of the remainder term obeys

$$ρN := N −1/2 + rN + r′
N + N 1/2λN + N 1/2λ′
N ≲δN,
(3.11)$$

¯ ψ ( · ) := -σ -1 J -1 0 ψ ( · , θ 0 , η 0 ) is the influence function, and the approximate variance is

$$σ2 := J−1
0 EP [ψ(W; θ0, η0)ψ(W; θ0, η0)′](J−1
0 )′.$$

The result establishes that the estimator based on the orthogonal scores achieves the rootN rate of convergence and is approximately normally distributed. It is noteworthy that this convergence result, both the rate of concentration and the distributional approximation, holds uniformly with respect to P varying over an expanding class of probability measures P N . This means that the convergence holds under any sequence of probability distributions ( P N ) N ⩾ 1 with P N ∈ P N for each N , which in turn implies that the results are robust with respect to perturbations of a given P along such sequences. The same property can be shown to fail for methods not based on orthogonal scores.

Theorem 3.2. ( Variance Estimator for DML ) Suppose that Assumptions 3.1 and 3.2 hold. In addition, suppose that δ N ⩾ N -[(1 -2 /q ) ∧ 1 / 2] for all N ⩾ 1 . Consider the following estimator of the asymptotic variance matrix of √ N ( ˜ θ 0 -θ 0 ) :

$$bσ2 = bJ−1
0
1
K
K
X
k=1
En,k[ψ(W; ˜θ0, bη0,k)ψ(W; ˜θ0, bη0,k)′]( bJ−1
0 )′,$$

where

$$bJ0 = 1
K
K
X
k=1
En,k[ψa(W; bη0,k)],$$

and ˜ θ 0 is either the DML1 or the DML2 estimator. This estimator concentrates around the true variance matrix σ 2 ,

$$bσ2 = σ2 + OP (ϱN),
ϱN := N −[(1−2/q)∧1/2] + rN + r′
N ≲δN.$$

Moreover, ̂ σ 2 can replace σ 2 in the statement of Theorem 3.1 with the size of the remainder term updated as ρ N = N -[(1 -2 /q ) ∧ 1 / 2] + r N + r ′ N + N 1 / 2 λ N + N 1 / 2 λ ′ N .

Theorems 3.1 and 3.2 can be used for standard construction of confidence regions which are uniformly valid over a large, interesting class of models:

Corollary 3.1. ( Uniformly Valid Confidence Bands ) Under the conditions of Theorem 3.2, suppose we are interested in the scalar parameter ℓ ′ θ 0 for some d θ × 1 vector ℓ . Then the confidence interval

$$CI :=
h
ℓ′˜θ0 ± Φ−1(1 −α/2)
p
ℓ′bσ2ℓ/N
i$$

$$sup
P ∈PN
PP (ℓ′θ0 ∈CI) −(1 −α)
 →0.$$

Indeed, the above theorem implies that CI obeys P P N ( ℓ ′ θ 0 ∈ CI) → (1 -α ) under any sequence { P N } ∈ P N , which implies that these claims hold uniformly in P ∈ P N . For example, one may choose { P N } such that, for some ϵ N → 0

$$sup
P ∈PN
|PP (ℓ′θ0 ∈CI) −(1 −α)| ⩽|PPN (ℓ′θ0 ∈CI) −(1 −α)| + ϵN →0.$$

Next we note that the estimators need not be semi-parametrically efficient, but under some conditions they can be.

Corollary 3.2. ( Cases with Semi-parametric Efficiency ) Under the conditions of Theorem 3.1, if the score ψ is efficient for estimating θ 0 at a given P ∈ P ⊂ P N , in the semi-parametric sense as defined in van der Vaart (1998), then the large sample variance σ 2 0 of ˜ θ 0 reaches the semi-parametric efficiency bound at this P relative to the model P .

## 3.3. Models with Nonlinear Scores

Let c 0 &gt; 0, c 1 &gt; 0, a &gt; 1, v &gt; 0, s &gt; 0, and q &gt; 2 be some finite constants, and let { δ N } N ⩾ 1 , { ∆ N } N ⩾ 1 , and { τ N } N ⩾ 1 be some sequences of positive constants converging to zero. To derive the properties of the DML estimator, we will use the following assumptions.

Assumption 3.3. (Nonlinear Moment Condition Problem with Approximate Neyman Orthogonality) For all N ⩾ 3 and P ∈ P N , the following conditions hold. (a) The true parameter value θ 0 obeys (2.1), and Θ contains a ball of radius c 1 N -1 / 2 log N centered at θ 0 . (b) The map ( θ, η ) ↦→ E P [ ψ ( W ; θ, η )] is twice continuously Gateaux-differentiable on Θ × T . (c) For all θ ∈ Θ , the identification relation

$$2∥EP [ψ(W; θ, η0)]∥⩾∥J0(θ −θ0)∥∧c0$$

is satisfied, for the Jacobian matrix

$$J0 := ∂θ′
n
EP [ψ(W; θ, η0)]
o
θ=θ0$$

having singular values between c 0 and c 1 . (d) The score ψ obeys the Neyman orthogonality or, more generally the Neyman near-orthogonality with λ N = δ N N -1 / 2 for the set T N ⊂ T .

Assumption 3.3 is mild and rather standard in moment condition problems. Assumption 3.3(a) requires θ 0 to be sufficiently separated from the boundary of Θ. Assumption 3.3(b) only requires differentiability of the function ( θ, η ) ↦→ E P [ ψ ( W ; θ, η )] and does not obeys

require differentiability of the function ( θ, η ) ↦→ ψ ( W ; θ, η ). Assumption 3.3(c) implies sufficient identifiability of θ 0 . Assumption 3.3(d) is the orthogonality condition that has already been extensively discussed.

Assumption 3.4. (Score Regularity and Requirements on the Quality of Estimation of Nuisance Parameters) Let K be a fixed integer. For all N ⩾ 3 and P ∈ P N , the following conditions hold. (a) Given a random subset I of { 1 , . . . , N } of size n = N/K , we have that the nuisance parameter estimator ̂ η 0 = ̂ η 0 (( W i ) i ∈ I c ) belongs to the realization set T N with probability at least 1 -∆ N , where T N contains η 0 and is constrained by conditions given below. (b) The parameter space Θ is bounded and for each η ∈ T N , the function class F 1 ,η = { ψ j ( · , θ, η ): j = 1 , ..., d θ , θ ∈ Θ } is suitably measurable and its uniform covering entropy obeys

$$sup
Q
log N(ϵ∥F1,η∥Q,2, F1,η, ∥· ∥Q,2) ⩽v log(a/ϵ),
for all 0 < ϵ ⩽1,
(3.12)$$

where F 1 ,η is a measurable envelope for F 1 ,η that satisfies ∥ F 1 ,η ∥ P,q ⩽ c 1 . (c) The following conditions on the statistical rates r N , r ′ N , and λ ′ N hold:

$$rN :=
sup
η∈TN,θ∈Θ
∥EP [ψ(W; θ, η) −EP [ψ(W; θ, η0)]∥⩽δNτN,
r′
N :=
sup
η∈TN,∥θ−θ0∥⩽τN
(EP [∥ψ(W; θ, η) −ψ(W; θ0, η0)∥2])1/2 and r′
N log1/2(1/r′
N) ⩽δN,
λ′
N :=
sup
r∈(0,1),η∈TN,∥θ−θ0∥⩽τN
∥∂2
rEP [ψ(W; θ0, η0 + r(θ −θ0) + r(η −η0))]∥⩽δNN −1/2.$$

(d) The variance of the score is non-degenerate: All eigenvalues of the matrix

$$EP [ψ(W; θ0, η0)ψ(W; θ0, η0)′]$$

are bounded from below by c 0 .

Assumptions 3.3(a)-(c) state that the estimator of the nuisance parameter belongs to the realization set T N ⊂ T , which is a shrinking neighborhood of η 0 that contracts at the 'statistical' rates r N and r ′ N and λ ′ N . These rates are not given in terms of the norm ∥·∥ T on T , but rather are intrinsic rates that are connected to the statistical problem at hand. In smooth problems, these conditions translate to the crude requirement that nuisance parameters are estimated at the o ( N -1 / 4 ) rate as discussed previously in the case with linear scores. However, these conditions can be refined as, for example, when λ ′ N = 0 or when some cross-derivatives vanish in λ ′ N ; see the linear case in the previous subsection for further discussion. Suitable measurability and pointwise entropy conditions, required in Assumption 3.4(b), are mild regularity conditions that are satisfied in all practical cases. The assumption of a bounded parameter space Θ in Assumption 3.4(b) is embedded in the entropy condition, but we state it separately for clarity. This assumption was not needed in the linear case, and it can be removed in the nonlinear case with the imposition of more complicated Huber-like regularity conditions. Assumption 3.4(c) is a set of mild growth conditions.

Remark 3.2. (Rate Requirements on Nuisance Parameter Estimators) Similar to the discussion in the linear case, the conditions in Assumption 3.4 are very flexible and embody refined requirements on the quality of the nuisance parameter estimators. The

conditions essentially reduce to the previous conditions in the linear case, with the exception of compactness, which is imposed to make the conditions easy to verify in non-linear cases.

Theorem 3.3. ( Properties of the DML for Nonlinear Scores ) Suppose that Assumptions 3.3 and 3.4 hold. In addition, suppose that δ N ⩾ N -1 / 2+1 /q log N and that N -1 / 2 log N ⩽ τ N ⩽ δ N for all N ⩾ 1 . Then the DML1 and DML2 estimators ˜ θ 0 concentrate in a 1 / √ N neighborhood of θ 0 , and are approximately linear and centered Gaussian:

$$√
Nσ−1(˜θ0 −θ0) =
1
√
N
N
X
i=1
¯ψ(Wi) + OP (ρN) ; N(0, I),$$

uniformly over P ∈ P N , where the size of the remainder term obeys

$$ρN := N −1/2+1/q log N + r′
N log1/2(1/r′
N) + N 1/2λN + N 1/2λ′
N ≲δN,$$

¯ ψ ( · ) := -σ -1 0 J -1 0 ψ ( · , θ 0 , η 0 ) is the influence function, and the approximate variance is

$$σ2 := J−1
0 EP [ψ(W; θ0, η0)ψ(W; θ0, η0)′](J−1
0 )′.$$

Moreover, in the statement above σ 2 can be replaced by a consistent estimator ̂ σ 2 , obeying ̂ σ 2 = σ 2 + o P ( ϱ N ) uniformly in P ∈ P N , with the size of the remainder term updated as ρ N = ρ N + ϱ N . Furthermore, Corollaries 3.1 and 3.2 continue to hold under the conditions of this theorem.

## 3.4. Finite-Sample Adjustments to Incorporate Uncertainty Induced by Sample Splitting

The estimation technique developed in this paper relies on subsamples obtained by randomly partitioning the sample: an auxiliary sample for estimating the nuisance functions and a main sample for estimating the parameter of interest. Although the specific sample partition has no impact on estimation results asymptotically, the effect of the particular random split on the estimate can be important in finite samples. To make the results more robust with respect to the partitioning, we propose to repeat the DML estimator S times, obtaining the estimates

$$˜θs
0,
s = 1, . . . , S.$$

Features of these estimates may then provide insight into the sensitivity of results to the sample splitting, and we can report results that incorporate features of this set of estimates that should be less driven by any particular sample-splitting realization.

Definition 3.3. ( Incorporating the Impact of Sample Splitting using Mean and Median Methods ) For point estimation, we define

$$˜θmean
0
= 1
S
S
X
s=1
˜θs
0
or
˜θmedian
0
= median{ ˜θs
0}S
s=1,$$

where the median operation is applied coordinatewise. To quantify and incorporate the variation introduced by sample splitting, we consider variance estimators:

$$bσ2,mean = 1
S
S
X
s=1

bσ2
s + (bθs −˜θmean)(bθs −˜θmean)′
,
(3.13)$$

and a more robust version,

$$bσ2,median = median{bσ2
s + ((bθs −˜θmedian)(bθs −˜θmedian)′}S
s=1,
(3.14)$$

where the median picks out the matrix with median operator norm, which preserve nonnegative definiteness.

We recommend using medians, reporting ˜ θ median 0 and ̂ σ 2Median , as these quantities are more robust to outliers.

Corollary 3.3. If S is fixed, as N →∞ and maintaining either Assumptions 3.1 and 3.2 or Assumptions 3.3 and 3.4 as appropriate, ˜ θ mean 0 and ˜ θ median 0 are first-order equivalent to ˜ θ 0 and obey the conclusions of Theorems 3.1 and 3.2 or of Theorem 3.3. Moreover, ̂ σ 2 , median and ̂ σ 2 , mean can replace ̂ σ in the statement of the appropriate theorems.

It would be interesting to investigate the behavior under the regime where S →∞ as N →∞ .

## 4. INFERENCE IN PARTIALLY LINEAR MODELS

## 4.1. Inference in Partially Linear Regression Models

Here we revisit the partially linear regression model

$$Y = Dθ0 + g0(X) + U,
EP [U | X, D] = 0,
(4.1)$$

$$D = m0(X) + V,
EP [V | X] = 0.
(4.2)$$

The parameter of interest is the regression coefficient θ 0 . If D is conditionally exogenous (as good as randomly assigned conditional on covariates), then θ 0 measures the average causal/treatment effect of D on potential outcomes.

The first approach to inference on θ 0 , which we described in the Introduction, is to employ the DML method using the score function

$$ψ(W; θ, η) := {Y −Dθ −g(X)}(D −m(X)),
η = (g, m),
(4.3)$$

where W = ( Y, D, X ) and g and m are P -square-integrable functions mapping the support of X to R . It is easy to see that θ 0 satisfies the moment condition E P ψ ( W ; θ 0 , η 0 ) = 0 , and also the orthogonality condition ∂ η E P ψ ( W ; θ 0 , η 0 )[ η -η 0 ] = 0 where η 0 = ( g 0 , m 0 ) . A second approach employs the Robinson-style 'partialling-out' score function

$$ψ(W; θ, η) := {Y −ℓ(X) −θ(D −m(X))}(D −m(X)),
η = (ℓ, m),
(4.4)$$

where W = ( Y, D, X ) and ℓ and m are P -square-integrable functions mapping the support of X to R . This gives an alternative parameterization of the score function above, and using this score is first-order equivalent to using the previous score. It is easy to see that θ 0 satisfies the moment condition E P ψ ( W ; θ 0 , η 0 ) = 0 , and also the orthogonality condition ∂ η E P ψ ( W ; θ 0 , η 0 )[ η -η 0 ] = 0 , for η 0 = ( ℓ 0 , m 0 ) , where ℓ 0 ( X ) = E P [ Y | X ].

In the partially linear model, the DML approach complements Belloni et al. (2013), Zhang and Zhang (2014), van de Geer et al. (2014), Javanmard and Montanari (2014b), and Belloni et al. (2014), Belloni et al. (2014), and Belloni et al. (2015), all of which consider estimation and inference for parameters within the partially linear model using lasso-type methods without cross-fitting. By relying upon cross-fitting, the DML approach allows for the use of a much broader collection of ML methods for estimating

the nuisance functions and also allows relaxation of sparsity conditions in the case where lasso or other sparsity-based estimators are used. Both the DML approach and the approaches taken in the aforementioned papers can be seen as heuristically 'debiasing' the score function ( Y -Dθ -g ( X )) D , which does not possess the orthogonality property unless m 0 ( X ) = 0.

Let ( δ N ) ∞ n =1 and (∆ N ) ∞ n =1 be sequences of positive constants approaching 0 as before. Also, let c , C , and q be fixed strictly positive constants such that q &gt; 4, and let K ⩾ 2 be a fixed integer. Moreover, for any η = ( ℓ 1 , ℓ 2 ), where ℓ 1 and ℓ 2 are functions mapping the support of X to R , denote ∥ η ∥ P,q = ∥ ℓ 1 ∥ P,q ∨ ∥ ℓ 2 ∥ P,q . For simplicity, assume that N/K is an integer.

Assumption 4.1. (Regularity Conditions for Partially Linear Regression Model) Let P be the collection of probability laws P for the triple W = ( Y, D, X ) such that (a) equations (4.1)-(4.2) hold; (b) ∥ Y ∥ P,q + ∥ D ∥ P,q ⩽ C ; (c) ∥ UV ∥ P, 2 ⩾ c 2 and E P [ V 2 ] ⩾ c ; (d) ∥ E P [ U 2 | X ] ∥ P, ∞ ⩽ C and ∥ E P [ V 2 | X ] ∥ P, ∞ ⩽ C ; and (e) given a random subset I of [ N ] of size n = N/K , the nuisance parameter estimator ̂ η 0 = ̂ η 0 (( W i ) i ∈ I c ) obeys the following conditions for all n ⩾ 1 : With P -probability no less than 1 -∆ N ,

$$∥bη0 −η0∥P,∞⩽C,
∥bη0 −η0∥P,2 ⩽δN,
and8$$

- (i) for the score ψ in (4.3), where η = ( g , m )
- (ii) for the score ψ in (4.4), where η = ( ℓ , m )

$$∥η0
η0∥P,∞⩽C,
∥η0
η0∥P,2 ⩽δN,
and
re ψ in (4.3), where bη0 = (bg0, bm0),
∥bm0 −m0∥P,2 × ∥bg0 −g0∥P,2 ⩽δNN −1/2,
re ψ in (4.4), where bη0 = (bℓ0, bm0),
∥bm0 −m0∥P,2 ×

∥bm0 −m0∥P,2 + ∥bℓ0 −ℓ0∥P,2

⩽δNN −1/2.$$

Remark 4.1. (Rate Conditions for Estimators of Nuisance Parameters) The only nonprimitive condition here is the assumption on the rate of estimating the nuisance parameters. These rates of convergence are available for most often used ML methods and are case-specific, so we do not restate conditions that are needed to reach these rates.

The following theorem follows as a corollary to the results in Section 3 by verifying Assumptions 3.1 and 3.2 and will be proven as a special case of Theorem 4.2 below.

Theorem 4.1. (DML Inference on Regression Coefficients in the Partially Linear Regression Model) Suppose that Assumption 4.1 holds. Then the DML1 and DML2 estimators ˜ θ 0 constructed in Definitions 3.1 and 3.2 above using the score in either (4.3) or (4.4) are first-order equivalent and obey

$$σ−1√
N(˜θ0 −θ0) ; N(0, 1),$$

uniformly over P ∈ P , where σ 2 = [E P V 2 ] -1 E P [ V 2 U 2 ][E P V 2 ] -1 . Moreover, the result continues to hold if σ 2 is replaced by ̂ σ 2 defined in Theorem 3.2. Consequently, confidence regions based upon the DML estimators ˜ θ 0 have uniform asymptotic validity:

$$lim
N→∞sup
P ∈P
PP

θ0 ∈[˜θ0 ± Φ−1(1 −α/2)bσ/
√
N]

−(1 −α)
 = 0.$$

8 We thank Rui Wang from the University of Washington for pointing out a mistake in the published version of the paper: In Assumptions 4.1 and 4.2, the correct condition is ∥ ̂ η 0 -η 0 ∥ P, ∞ ⩽ C rather than ∥ ̂ η 0 -η 0 ∥ P,q ⩽ C appearing in the published version.

Remark 4.2. (Asymptotic Efficiency under Homoscedasticity) Under conditional homoscedasticity, i.e. E[ U 2 | Z ] = E[ U 2 ], the asymptotic variance σ 2 reduces to E[ V 2 ] -1 E[ U 2 ], which is the semi-parametric efficiency bound for θ .

Remark 4.3. (Tightness of Conditions under Cross-Fitting) The conditions in Theorem 4.1 are fairly sharp, though they are somewhat simplified for ease of presentation. The sharpness can be understood by examining the case where the regression function g 0 and the propensity function m 0 are sparse with sparsity indices s g ≪ N and s m ≪ N and are estimated by ℓ 1 -penalized estimators ̂ g 0 and ̂ m 0 that have sparsity indices of orders s g and s m and converge to g 0 and m 0 at the rates √ s g /N and √ s m /N (ignoring logs). The rate conditions in Assumption 4.1 then require (ignoring logs) that

$$p
sg/N
p
sm/N ≪N −1/2 ⇔sgsm ≪N,$$

which is much weaker than the condition

$$(sg)2 + (sm)2 ≪N$$

(ignoring logs) required without sample splitting. For example, if the propensity function m 0 is very sparse (low s m ), then the regression function is allowed to be quite dense (high s g ), and vice versa. If the propensity function is known ( s m = 0) or can be estimated at the N -1 / 2 rate, then only consistency for ̂ g 0 is needed. Such comparisons also extend to approximately sparse models.

## 4.2. Inference in Partially Linear IV Models

Here we extend the partially linear regression model studied in Section 4.1 to allow for instrumental variable (IV) identification. Specifically, we consider the model

$$Y = Dθ0 + g0(X) + U,
EP [U | X, Z] = 0,
(4.5)$$

$$Z = m0(X) + V,
EP [V | X] = 0,
(4.6)$$

where Z is the instrumental variable. As before, the parameter of interest is θ and its true value is θ 0 . If Z = D , the model (4.5)-(4.6) coincides with (4.1)-(4.2) but is otherwise different.

To estimate θ 0 and to perform inference on it, we will use the score

$$ψ(W; θ, η) := (Y −Dθ −g(X))(Z −m(X)),
η = (g, m),
(4.7)$$

where W = ( Y, D, X, Z ) and g and m are P -square-integrable functions mapping the support of X to R . Alternatively, we can use the Robinson-style score

$$ψ(W; θ, η) := (Y −ℓ(X) −θ(D −r(X)))(Z −m(X)),
η = (ℓ, m, r),
(4.8)$$

where W = ( Y, D, X, Z ) and ℓ , m , and r are P -square-integrable functions mapping the support of X to R . It is straightforward to verify that both scores satisfy the moment condition E P ψ ( W ; θ 0 , η 0 ) = 0 and also the orthogonality condition ∂ η E P ψ ( W ; θ 0 , η 0 )[ η -η 0 ] = 0, for η 0 = ( g 0 , m 0 ) in the former case and η 0 = ( ℓ 0 , m 0 , r 0 ) for ℓ 0 and r 0 defined by ℓ 0 ( X ) = E P [ Y | X ] and r 0 ( X ) = E P [ D | X ], respectively, in the latter case. 9

9 It is interesting to note that the methods for constructing Neyman orthogonal scores described in Section 2 may give scores that are different from those in (4.7) and (4.8). For example, applying the method for conditional moment restriction problems in Section 2.2.4 with Ω( R ) = 1 gives the score

Note that the score in (4.8) has a minor advantage over the score in (4.7) because all of its nuisance parameters are conditional mean functions, which can be directly estimated by the ML methods. If one prefers to use the score in (4.7), one has to construct an estimator of g 0 first. To do so, one can first obtain a DML estimator of θ 0 based on the score in (4.8), say ˜ θ 0 . Then, using the fact that g 0 ( X ) = E P [ Y -Dθ 0 | X ], one can construct an estimator ̂ g 0 by applying an ML method to regress Y -D ˜ θ 0 on X . Alternatively, one can use assumption-specific methods to directly estimate g 0 , without using the score (4.8) first. For example, if g 0 can be approximated by a sparse linear combination of a large set of transformations of X , one can use the methods of Gautier and Tsybakov (2014) to obtain an estimator of g 0 .

Let ( δ N ) ∞ n =1 and (∆ N ) ∞ n =1 be sequences of positive constants approaching 0 as before. Also, let c , C , and q be fixed strictly positive constants such that q &gt; 4, and let K ⩾ 2 be a fixed integer. Moreover, for any η = ( ℓ j ) l j =1 mapping the support of X to R l , denote ∥ η ∥ P,q = max 1 ⩽ j ⩽ l ∥ ℓ j ∥ P,q . For simplicity, assume that N/K is an integer.

Assumption 4.2. (Regularity Conditions for Partially Linear IV Model) For all probability laws P ∈ P for the quadruple W = ( Y, D, X, Z ) the following conditions hold: (a) equations (4.5)-(4.6) hold; (b) ∥ Y ∥ P,q + ∥ D ∥ P,q + ∥ Z ∥ P,q ⩽ C ; (c) ∥ UV ∥ P, 2 ⩾ c 2 and | E P [ DV ] | ⩾ c ; (d) ∥ E P [ U 2 | X ] ∥ P, ∞ ⩽ C and ∥ E P [ V 2 | X ] ∥ P, ∞ ⩽ C ; and (e) given a random subset I of [ N ] of size n = N/K , the nuisance parameter estimator ̂ η 0 = ̂ η 0 (( W i ) i ∈ I c ) obeys the following conditions: With P -probability no less than 1 -∆ N ,

$$∥bη0 −η0∥P,∞⩽C,
∥bη0 −η0∥P,2 ⩽δN,
and$$

- (i) for the score ψ in (4.7), where ̂ η 0 = ( ̂ g 0 , ̂ m 0 ) ,

$$∥bm0 −m0∥P,2 × ∥bg0 −g0∥P,2 ⩽δNN −1/2,$$

- (ii) for the score ψ in (4.8), where ̂ η 0 = ( ̂ ℓ 0 , ̂ m 0 , ̂ r 0 ) ,

$$∥bm0 −m0∥P,2 ×

∥br0 −r0∥P,2 + ∥bℓ0 −ℓ0∥P,2

⩽δNN −1/2.$$

The following theorem follows as a corollary to the results in Section 3 by verifying Assumptions 3.1 and 3.2.

Theorem 4.2. (DML Inference on Regression Coefficients in the Partially Linear IV Model) Suppose that Assumption 4.2 holds. Then the DML1 and DML2 estimators ˜ θ 0 constructed in Definitions 3.1 and 3.2 above using the score in either (4.7) or (4.8) are first-order equivalent and obey

$$σ−1√
N(˜θ0 −θ0) ; N(0, 1),$$

uniformly over P ∈ P , where σ 2 = [E P DV ] -1 E P [ V 2 U 2 ][E P DV ] -1 . Moreover, the result continues to hold if σ 2 is replaced by ̂ σ 2 defined in Theorem 3.2. Consequently, confidence regions based upon the DML estimators ˜ θ 0 have uniform asymptotic validity:

$$lim
N→∞sup
P ∈P
PP

θ0 ∈[˜θ0 ± Φ−1(1 −α/2)bσ/
√
N]

−(1 −α)
 = 0.$$

ψ ( W ; θ, η ) = ( Y -Dθ -g ( X ))( r ( Z, X ) -f ( X )), where the true values of r ( Z, X ) and f ( X ) are r 0 ( Z, X ) = E P [ D | Z, X ] and f 0 ( X ) = E P [ D | X ], respectively. It may be interesting to compare properties of the DML estimators ˜ θ 0 based on this score with those based on (4.7) and (4.8) in future work.

## 5. INFERENCE ON TREATMENT EFFECTS IN THE INTERACTIVE MODEL

## 5.1. Inference on ATE and ATTE

In this section, we specialize the results of Section 3 to estimating treatment effects under the unconfoundedness assumption of Rosenbaum and Rubin (1983). Within this setting, there is a large classical literature focused on low-dimensional settings that provides methods for adjusting for confounding variables including regression methods, propensity score adjustment methods, matching methods, and 'doubly-robust' combinations of these methods; see, for example, Robins and Rotnitzky (1995), Hahn (1998), Hirano et al. (2003), and Abadie and Imbens (2006) as well as the textbook overview provided in Imbens and Rubin (2015). In this section, we present results that complement this important classic work as well as the rapidly expanding body of work on estimation under unconfoundedness using ML methods; see, among others, Athey et al. (2016), Belloni et al. (2017), Belloni et al. (2014), Farrell (2015), and Imai and Ratkovic (2013).

We specifically consider estimation of average treatment effects when treatment effects are fully heterogeneous and the treatment variable is binary, D ∈ { 0 , 1 } . We consider vectors ( Y, D, X ) such that

$$Y = g0(D, X) + U,
EP [U | X, D] = 0,
(5.1)$$

$$D = m0(X) + V,
EP [V | X] = 0.
(5.2)$$

Since D is not additively separable, this model is more general than the partially linear model for the case of binary D . A common target parameter of interest in this model is the average treatment effect (ATE),

$$θ0 = EP [g0(1, X) −g0(0, X)].10$$

Another common target parameter is the average treatment effect for the treated (ATTE),

$$θ0 = EP [g0(1, X) −g0(0, X)|D = 1].$$

The confounding factors X affect the policy variable via the propensity score m 0 ( X ) and the outcome variable via the function g 0 ( D,X ). Both of these functions are unknown and potentially complicated, and we can employ ML methods to learn them.

We proceed to set up moment conditions with scores obeying orthogonality conditions. For estimation of the ATE, we employ

$$ψ(W; θ, η) := (g(1, X) −g(0, X)) + D(Y −g(1, X))
m(X)
−(1 −D)(Y −g(0, X))
1 −m(X)
−θ, (5.3)$$

where the nuisance parameter η = ( g, m ) consists of P -square-integrable functions g and m mapping the support of ( D,X ) to R and the support of X to ( ε, 1 -ε ), respectively, for some ε ∈ (0 , 1 / 2). The true value of η is η 0 = ( g 0 , m 0 ). This orthogonal moment condition is based on the influence function for the mean for missing data from Robins and Rotnitzky (1995).

For estimation of the ATTE, we use the score

$$ψ(W; θ, η) = D(Y −g(X))
p
−m(X)(1 −D)(Y −g(X))
p(1 −m(X))
−Dθ
p ,
(5.4)$$

10 Without unconfoundedness/conditional exogeneity, these quantities measure association, and could be referred to as average predictive effect (APE) and average predictive effect for the exposed (APEX). Inferential results for these objects would follow immediately from Theorem 5.1.

where the nuisance parameter η = ( g, m, p ) consists of P -square-integrable functions g and m mapping the support of X to R and to ( ε, 1 -ε ), respectively, and a constant p ∈ ( ε, 1 -ε ), for some ε ∈ (0 , 1 / 2). The true value of η is η 0 = ( g 0 , m 0 , p 0 ), where g 0 ( X ) = g 0 (0 , X ) and p 0 = E P [ D ]. Note that estimating ATTE does not require estimating g 0 (1 , X ). Note also that since p is a constant, it does not affect the DML estimators ˜ θ 0 based on the score ψ in (5.4) but having p simplifies the formula for the variance of ˜ θ 0 .

Using their respective scores, it can be easily seen that true parameter values θ 0 for ATE and ATTE obey the moment condition E P ψ ( W ; θ 0 , η 0 ) = 0 , and also that the orthogonality condition ∂ η E P ψ ( W ; θ 0 , η 0 )[ η -η 0 ] = 0 holds.

Let ( δ N ) ∞ n =1 and (∆ N ) ∞ n =1 be sequences of positive constants approaching 0. Also, let c, ε, C and q be fixed strictly positive constants such that q &gt; 2, and let K ⩾ 2 be a fixed integer. Moreover, for any η = ( ℓ 1 , . . . , ℓ l ), denote ∥ η ∥ P,q = max 1 ⩽ j ⩽ l ∥ ℓ j ∥ P,q . For simplicity, assume that N/K is an integer.

Assumption 5.1. (Regularity Conditions for ATE and ATTE Estimation) For all probability laws P ∈ P for the triple ( Y, D, X ) the following conditions hold: (a) equations (5.1)-(5.2) hold, with D ∈ { 0 , 1 } , (b) ∥ Y ∥ P,q ⩽ C , (c) P P ( ε ⩽ m 0 ( X ) ⩽ 1 -ε ) = 1 , (d) ∥ U ∥ P, 2 ⩾ c , (e) ∥ E P [ U 2 | X ] ∥ P, ∞ ⩽ C , and (f) given a random subset I of [ N ] of size n = N/K , the nuisance parameter estimator ̂ η 0 = ̂ η 0 (( W i ) i ∈ I c ) obeys the following conditions: with P -probability no less than 1 -∆ N :

$$∥bη0 −η0∥P,q ⩽C,
∥bη0 −η0∥P,2 ⩽δN,
∥bm0 −1/2∥P,∞⩽1/2 −ε,
and$$

(i) for the score ψ in (5.3), where ̂ η 0 = ( ̂ g 0 , ̂ m 0 ) and the target parameter is ATE,

$$∥bm0 −m0∥P,2 × ∥bg0 −g0∥P,2 ⩽δNN −1/2,$$

(ii) for the score ψ in (5.4), where ̂ η 0 = ( ̂ g 0 , ̂ m 0 , ̂ p 0 ) and the target parameter is ATTE,

$$∥bm0 −m0∥P,2 × ∥bg0 −g0∥P,2 ⩽δNN −1/2.$$

Remark 5.1. The only non-primitive condition here is the assumption on the rate of estimating the nuisance parameters. These rates of convergence are available for most often used ML methods and are case-specific, so we do not restate conditions that are needed to reach these rates. The conditions are not the tightest possible, but offer a set of simple conditions under which Theorem 5.1 follows as a special case of the general theorem provided in Section 3. One could obtain more refined conditions by doing customized proofs.

The following theorem follows as a corollary to the results in Section 3 by verifying Assumptions 3.1 and 3.2.

Theorem 5.1. ( DML Inference on ATE and ATTE ) Suppose that either (a) the target parameter is ATE, θ 0 = E P [ g 0 (1 , X ) -g 0 (0 , X )] , and the score ψ in (5.3) is used, or (b) the target parameter is ATTE, θ 0 = E P [ g 0 (1 , X ) -g 0 (0 , X ) | D = 1] , and the score ψ in (5.4) is used. In addition, suppose that Assumption 5.1 holds. Then the DML1 and DML2 estimators ˜ θ 0 , constructed in Definitions 3.1 and 3.2, are first-order equivalent and obey √

$$σ−1√
N(˜θ0 −θ0) ⇝N(0, 1),
(5.5)$$

uniformly over P ∈ P , where σ 2 = E P [ ψ 2 ( W ; θ 0 , η 0 )] . Moreover, the result continues to hold if σ 2 is replaced by ̂ σ 2 defined in Theorem 3.2. Consequently, confidence regions based upon the DML estimators ˜ θ 0 have uniform asymptotic validity:

$$lim
N→∞sup
P ∈P
PP

θ0 ∈[˜θ0 ± Φ−1(1 −α/2)bσ/
√
N]

−(1 −α)
 = 0.$$

The scores ψ in (5.3) and (5.4) are efficient, so both estimators are asymptotically efficient, reaching the semi-parametric efficiency bound of Hahn (1998).

Remark 5.2. (Tightness of Conditions) The conditions in Assumption 5.1 are fairly sharp though somewhat simplified for ease of presentation. The sharpness can be understood by examining the case where the regression function g 0 and the propensity function m 0 are sparse with sparsity indices s g ≪ N and s m ≪ N and are estimated by ℓ 1 -penalized estimators ̂ g 0 and ̂ m 0 that have sparsity indices of orders s g and s m and converge to g 0 and m 0 at the rates √ s g /N and √ s m /N (ignoring logs). Then the rate conditions in Assumption 5.1 require

$$p
sg/N
p
sm/N ≪N −1/2 ⇔sgsm ≪N$$

(ignoring logs) which is much weaker than the condition ( s g ) 2 + ( s m ) 2 ≪ N (ignoring logs) required without sample splitting. For example, if the propensity score m 0 is very sparse, then the regression function is allowed to be quite dense with s g &gt; √ N , and vice versa. If the propensity score is known ( s m = 0), then only consistency for ̂ g 0 is needed. Such comparisons also extend to approximately sparse models. We note that similar refined rates appeared in Farrell (2015) who considers estimation of treatment effects in a setting where an approximately sparse model holds for both the regression and propensity score functions. In interesting related work, Athey et al. (2016) show that √ N consistent estimation of an average treatment effect is possible under very weak conditions on the propensity score - allowing for the possibility that the propensity score may not be consistently estimated - under strong sparsity of the regression function such that s g ≪ √ N . Thus, the approach taken in this context and Athey et al. (2016) are complementary and one may prefer either depending on whether or not the regression function can be estimated extremely well based on a sparse method.

## 5.2. Inference on Local Average Treatment Effects

In this section, we consider estimation of local average treatment effects (LATE) with a binary treatment variable, D ∈ { 0 , 1 } , and a binary instrument, Z ∈ { 0 , 1 } . 11 As before, Y denotes the outcome variable, and X is the vector of covariates.

Consider the functions µ 0 , m 0 , and p 0 , where µ 0 maps the support of ( Z, X ) to R and m 0 and p 0 respectively map the support of ( Z, X ) and X to ( ε, 1 -ε ) for some ε ∈ (0 , 1 / 2), such that

$$Y = µ0(Z, X) + U,
EP [U | Z, X] = 0,
(5.6)$$

$$D = m0(Z, X) + V,
EP [V | Z, X] = 0,$$

$$(5.7)$$

$$Z = p0(X) + ζ,
EP [ζ | X] = 0.
(5.8)$$

11 Similar results can be provided for the local average treatment effect on the treated (LATT) by adapting the following arguments to use the orthogonal scores for the LATT. See, for example, Belloni et al. (2017).

We are interested in estimating

$$θ0 = EP [µ(1, X)] −EP [µ(0, X)]
EP [m(1, X)] −EP [m(0, X)].$$

Under the assumptions of Imbens and Angrist (1994) and Fr¨ olich (2007), θ 0 is the LATE the average treatment effect for compliers which are observations that would have D = 1 if Z were 1 and would have D = 0 if Z were 0. To estimate θ 0 , we will use the score

$$ψ(W; θ, η) := µ(1, X) −µ(0, X) + Z(Y −µ(1, X))
p(X)
−(1 −Z)(Y −µ(1, X))
1 −p(X))
−

m(1, X) −m(0, X) + Z(D −m(1, X))
p(X)
−(1 −Z)(D −m(0, X))
1 −p(X)

× θ,$$

where W = ( Y, D, X, Z ) and the nuisance parameter η = ( µ, m, p ) consists of P -squareintegrable functions µ , m , and p , with µ mapping the support of ( Z, X ) to R and m and p respectively mapping the support of ( Z, X ) and X to ( ε, 1 -ε ) for some ε ∈ (0 , 1 / 2). It is easy to verify that this score satisfies the moment condition E P ψ ( W ; θ 0 , η 0 ) = 0 and also the orthogonality condition ∂ η E P ψ ( W ; θ 0 , η 0 )[ η -η 0 ] = 0 for η 0 = ( µ 0 , m 0 , p 0 ).

Let ( δ N ) ∞ n =1 and (∆ N ) ∞ n =1 be sequences of positive constants approaching 0. Also, let c , C , and q be fixed strictly positive constants such that q &gt; 4, and let K ⩾ 2 be a fixed integer. Moreover, for any η = ( ℓ 1 , ℓ 2 , ℓ 3 ), where ℓ 1 is a function mapping the support of ( Z, X ) to R and ℓ 2 and ℓ 3 are functions respectively mapping the support of ( Z, X ) and X to ( ε, 1 -ε ) for some ε ∈ (0 , 1 / 2), denote ∥ η ∥ P,q = ∥ ℓ 1 ∥ P,q ∨∥ ℓ 2 ∥ P, 2 ∨∥ ℓ 3 ∥ P,q . For simplicity, assume that N/K is an integer.

Assumption 5.2. (Regularity Conditions for LATE Estimation) For all probability laws P ∈ P for the quadruple W = ( Y, D, X, Z ) the following conditions hold: (a) equations (5.6)-(5.8) hold, with D ∈ { 0 , 1 } and Z ∈ { 0 , 1 } ; (b) ∥ Y ∥ P,q ⩽ C ; (c) P P ( ε ⩽ p 0 ( X ) ⩽ 1 -ε ) = 1 , (d) E P [ m 0 (1 , X ) -m 0 (0 , X )] ⩾ c , (e) ∥ U -θ 0 V ∥ P, 2 ⩾ c ; (f) ∥ E P [ U 2 | X ] ∥ P, ∞ ⩽ C ; and (g) given a random subset I of [ N ] of size n = N/K , the nuisance parameter estimator ̂ η 0 = ̂ η 0 (( W i ) i ∈ I c ) obeys the following conditions: with P -probability no less than 1 -∆ N :

$$∥bη0 −η0∥P,q ⩽C,
∥bη0 −η0∥P,2 ⩽δN,
∥bp0 −1/2∥P,∞⩽1/2 −ε,
and
∥bp0 −p0∥P,2 ×

∥bµ0 −µ0∥P,2 + ∥bm0 −m0∥P,2

⩽δNN −1/2.$$

$$p0
p0∥P,2 ×
∥µ0
µ0∥P,2 + ∥m$$

The following theorem follows as a corollary to the results in Section 3 by verifying Assumptions 3.1 and 3.2.

Theorem 5.2. ( DML Inference on LATE ) Suppose that Assumption 5.2 holds. Then the DML1 and DML2 estimators ˜ θ 0 constructed in Definitions 3.1 and 3.2 and based on the score ψ above are first-order equivalent and obey

$$σ−1√
N(˜θ0 −θ0) ⇝N(0, 1),
(5.9)$$

uniformly over P ∈ P , where σ 2 = (E P [ m (1 , X ) -m (0 , X )]) -2 E P [ ψ 2 ( W ; θ 0 , η 0 )] . Moreover, the result continues to hold if σ 2 is replaced by ̂ σ 2 defined in Theorem 3.2. Consequently, confidence regions based upon the DML estimators ˜ θ 0 have uniform asymptotic

validity:

$$lim
N→∞sup
P ∈P
PP

θ0 ∈[˜θ0 ± Φ−1(1 −α/2)bσ/
√
N]

−(1 −α)
 = 0.$$

## 6. EMPIRICAL EXAMPLES

To illustrate the methods developed in the preceding sections, we consider three empirical examples. The first example reexamines the Pennsylvania Reemployment Bonus experiment which used a randomized control trial to investigate the incentive effect of unemployment insurance. In the second, we use the DML method to estimate the effect of 401(k) eligibility, the treatment variable, and 401(k) participation, a self-selected decision to receive the treatment that we instrument for with assignment to the treatment state, on accumulated assets. In this example, the treatment variable is not randomly assigned and we aim to eliminate the potential biases due to the lack of random assignment by flexibly controlling for a rich set of variables. In the third, we revisit Acemoglu et al. (2001) IV estimation of the effects of institutions on economic growth by estimating a partially linear IV model.

## 6.1. The effect of Unemployment Insurance Bonus on Unemployment Duration

In this example, we re-analyze the Pennsylvania Reemployment Bonus experiment which was conducted by the US Department of Labor in the 1980s to test the incentive effects of alternative compensation schemes for unemployment insurance (UI). This experiment has been previously studied by Bilias (2000) and Bilias and Koenker (2002). In these experiments, UI claimants were randomly assigned either to a control group or one of five treatment groups. 12 In the control group, the standard rules of the UI system applied. Individuals in the treatment groups were offered a cash bonus if they found a job within some pre-specified period of time (qualification period), provided that the job was retained for a specified duration. The treatments differed in the level of the bonus, the length of the qualification period, and whether the bonus was declining over time in the qualification period; see Bilias and Koenker (2002) for further details.

In our empirical example, we focus only on the most generous compensation scheme, treatment 4, and drop all individuals who received other treatments. In this treatment, the bonus amount is high and the qualification period is long compared to other treatments, and claimants are eligible to enroll in a workshop. Our treatment variable, D, is an indicator variable for being assigned treatment 4, and the outcome variable, Y, is the log of duration of unemployment for the UI claimants. The vector of covariates, X, consists of age group dummies, gender, race, the number of dependents, quarter of the experiment, location within the state, existence of recall expectations, and type of occupation.

We report results based on five simple methods for estimating the nuisance functions used in forming the orthogonal estimating equations. We consider three tree-based methods, labeled 'Random Forest', 'Reg. Tree', and 'Boosting', one ℓ 1 -penalization based method, labeled 'Lasso', and a neural network method, labeled 'Neural Net'. For 'Reg. Tree,' we fit a single CART tree to estimate each nuisance function with penalty parameter chosen by 10-fold cross-validation. The results in the 'Random Forest' column are

12 There are six treatment groups in the experiments. Following Bilias (2000). we merge the groups 4 and 6.

obtained by estimating each nuisance function with a random forest which averages over 1000 trees. The results in 'Boosting' are obtained using boosted regression trees with regularization parameters chosen by 10-fold cross-validation. To estimate the nuisance functions using the neural networks, we use 2 neurons and a decay parameter of 0.02, and we set activation function as logistic for classification problems and as linear for regression problems. 13 'Lasso' estimates an ℓ 1 -penalized linear regression model using the data-driven penalty parameter selection rule developed in Belloni et al. (2012). For 'Lasso', we use a set of 96 potential control variables formed from the raw set of covariates and all second order terms, i.e. all squares and first-order interactions. For the remaining methods, we use the raw set of covariates as features.

We also consider two hybrid methods labeled 'Ensemble' and 'Best'. 'Ensemble' optimally combines four of the ML methods listed above by estimating the nuisance functions as weighted averages of estimates from 'Lasso,' 'Boosting,' 'Random Forest,' and 'Neural Net'. The weights are restricted to sum to one and are chosen so that the weighted average of these methods gives the lowest average mean squared out-of-sample prediction error estimated using 5-fold cross-validation. The final column in Table 1 ('Best') reports results that combine the methods in a different way. After obtaining estimates from the five simple methods and 'Ensemble', we select the best methods for estimating each nuisance functions based on the average out-of-sample prediction performance for the target variable associated with each nuisance function obtained from each of the previously described approaches. As a result, the reported estimate in the last column uses different ML methods to estimate different nuisance functions. Note that if a single method outperformed all the others in terms of prediction accuracy for all nuisance functions, the estimate in the 'Best' column would be identical to the estimate reported under that method.

Table 1 presents DML2 estimates of the ATE on unemployment duration using the median method described in Section 3.4. We report results for heterogeneous effect model in Panel A and for the partially linear model in Panel B. Because the treatment is randomly assigned, we use the fraction of treated as the estimator of the propensity score in forming the orthogonal estimating equations. 14 For both the partially linear model and the interactive model, we report estimates obtained using 2-fold cross-fitting and 5-fold cross-fitting. All results are based on taking 100 different sample splits. We summarize results across the sample splits using the median method. For comparison, we report two different standard errors. In brackets, we report the median standard error from across the 100 splits; and we report standard errors adjusted for variability across the sample splits using the median method in parentheses.

The estimation results are consistent with the findings of previous studies which have analyzed the Pennsylvania Bonus Experiment. The ATE on unemployment duration is negative and significant across all estimation methods at the 5% level regardless of the standard error estimator used. Interestingly, we see that there is no practical difference across the two different standard errors in this example.

13 We also experimented with 'Deep Learning' methods from which we obtained similar results for some tuning parameters. However, we ran into stability and computational issues and chose not to report these results in the empirical section.

14 We also estimated the effects using nonparametric estimates of the conditional propensity score obtained from the ML procedures given in the column labels. As expected due to randomization, the results are similar to those provided in Table 1 and are not reported for brevity.

DML

Table 1 . Estimated Effect of Cash Bonus on Unemployment Duration

|                                 | Lasso                           | Reg. Tree                       | Forest                          | Boosting                        | Neural Net.                     | Ensemble                        | Best                            |
|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model |
| ATE (2 fold)                    | -0.081 [0.036] (0.036)          | -0.084 [0.036] (0.036)          | -0.074 [0.036] (0.036)          | -0.079 [0.036] (0.036)          | -0.073 [0.036] (0.036)          | -0.079 [0.036] (0.036)          | -0.078 [0.036] (0.036)          |
| ATE (5 fold)                    | -0.081 [0.036] (0.036)          | -0.085 [0.036] (0.037)          | -0.074 [0.036] (0.036)          | -0.077 [0.035] (0.036)          | -0.073 [0.036] (0.036)          | -0.078 [0.036] (0.036)          | -0.077 [0.036] (0.036)          |
| B. Partially Linear             | Regression                      | Model                           |                                 |                                 |                                 |                                 |                                 |
| ATE (2 fold)                    | -0.080 [0.036] (0.036)          | -0.084 [0.036] (0.036)          | -0.077 [0.035] (0.037)          | -0.076 [0.035] (0.036)          | -0.074 [0.035] (0.036)          | -0.075 [0.035] (0.036)          | -0.075 [0.035] (0.036)          |
| ATE (5 fold)                    | -0.080 [0.036] (0.036)          | -0.084 [0.036] (0.037)          | -0.077 [0.035] (0.036)          | -0.074 [0.035] (0.035)          | -0.073 [0.035] (0.036)          | -0.075 [0.035] (0.035)          | -0.074 [0.035] (0.035)          |

Note: Estimated ATE and standard errors from a linear model (Panel B) and heterogeneous effect model (Panel A) based on orthogonal estimating equations. Column labels denote the method used to estimate nuisance functions. Results are based on 100 splits with point estimates calculated the median method. The median standard error across the splits are reported in brackets and standard errors calculated using the median method to adjust for variation across splits are provided in parentheses. Further details about the methods are provided in the main text.

## 6.2. The effect of 401(k) Eligibility and Participation on Net Financial Assets

The key problem in determining the effect of 401(k) eligibility is that working for a firm that offers access to a 401(k) plan is not randomly assigned. To overcome the lack of random assignment, we follow the strategy developed in Poterba et al. (1994a) and Poterba et al. (1994b). In these papers, the authors use data from the 1991 Survey of Income and Program Participation and argue that eligibility for enrolling in a 401(k) plan in this data can be taken as exogenous after conditioning on a few observables of which the most important for their argument is income. The basic idea of their argument is that, at least around the time 401(k) initially became available, people were unlikely to be basing their employment decisions on whether an employer offered a 401(k) but would instead focus on income and other aspects of the job. Following this argument, whether one is eligible for a 401(k) may then be taken as exogenous after appropriately conditioning on income and other control variables related to job choice.

A key component of the argument underlying the exogeneity of 401(k) eligibility is that eligibility may only be taken as exogenous after conditioning on income and other variables related to job choice that may correlate with whether a firm offers a 401(k). Poterba et al. (1994a) and Poterba et al. (1994b) and many subsequent papers adopt this argument but control only linearly for a small number of terms. One might wonder whether such specifications are able to adequately control for income and other related confounds. At the same time, the power to learn about treatment effects decreases as one allows more flexible models. The principled use of flexible ML tools offers one resolution to this tension. The results presented below thus complement previous results which rely

Table 2 . Estimated Effect of 401(k) Eligibility on Net Financial Assets

|                                 | Lasso                           | Reg. Tree                       | Forest                          | Boosting                        | Neural Net.                     | Ensemble                        | Best                            |
|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model | A. Interactive Regression Model |
| ATE (2 fold)                    | 6830 [1282]                     | 7713 [1208] (1271)              | 7770 (1363)                     | 7806 [1159] (1202)              | 7764 [1328] (1468)              | 7702 [1149] (1170)              | 7546 (1533)                     |
|                                 | (1530)                          |                                 | [1276]                          |                                 |                                 |                                 | [1360]                          |
| ATE (5 fold)                    | 7170 [1201]                     | 7993 [1198]                     | 8105 [1242]                     | 7713 [1155]                     | 7788 [1238]                     | 7839 [1134]                     | 7753 [1237]                     |
| B. Partially Linear             | (1398) Regression               | (1236) Model                    | (1299)                          | (1177)                          | (1293)                          | (1148)                          | (1294)                          |
| ATE (2 fold)                    | 7717 [1346]                     | 8709 [1363]                     | 9116 [1302]                     | 8759 [1339]                     | 8950 [1335]                     | 9010 [1309]                     | 9125 [1304]                     |
|                                 | (1749)                          | (1427)                          | (1377)                          | (1382)                          | (1408)                          | (1344)                          | (1357)                          |
| ATE (5 fold)                    | 8187                            | 8871 [1358]                     | 9247                            | 9110 [1314]                     | 9038                            | 9166                            | 9215                            |
|                                 | [1298]                          |                                 | [1295]                          |                                 | [1322]                          | [1299]                          | [1294]                          |
|                                 | (1558)                          | (1418)                          | (1328)                          |                                 |                                 |                                 | (1312)                          |
|                                 |                                 |                                 |                                 | (1328)                          | (1355)                          | (1310)                          |                                 |

Note: Estimated ATE and standard errors from a linear model (Panel B) and heterogeneous effect model (Panel A) based on orthogonal estimating equations. Column labels denote the method used to estimate nuisance functions. Results are based on 100 splits with point estimates calculated the median method. The median standard error across the splits are reported in brackets and standard errors calculated using the median method to adjust for variation across splits are provided in parentheses. Further details about the methods are provided in the main text.

on the assumption that confounding effects can adequately be controlled for by a small number of variables chosen ex ante by the researcher.

In the example in this paper, we use the same data as in Chernozhukov and Hansen (2004). We use net financial assets - defined as the sum of IRA balances, 401(k) balances, checking accounts, U.S. saving bonds, other interest-earning accounts in banks and other financial institutions, other interest-earning assets (such as bonds held personally), stocks, and mutual funds less non-mortgage debt - as the outcome variable, Y , in our analysis. Our treatment variable, D , is an indicator for being eligible to enroll in a 401(k) plan. The vector of raw covariates, X , consists of age, income, family size, years of education, a married indicator, a two-earner status indicator, a defined benefit pension status indicator, an IRA participation indicator, and a home ownership indicator.

In Table 2, we report DML2 estimates of ATE of 401(k) eligibility on net financial assets both in the partially linear model as in (1.1) and allowing for heterogeneous treatment effects using the interactive model outlined in Section 5.1. To reduce the disproportionate impact of extreme propensity score weights in the interactive model, we trim the propensity scores at 0.01 and 0.99. We present two sets of results based on sample-splitting as discussed in Section 3 using 2-fold cross-fitting and 5-fold cross-fitting. As in the previous section, we consider 100 different sample partitions and summarize the results across different sample splits using the median method. For comparison, we report two different standard errors. In brackets, we report the median standard error from across the 100 splits; and we report standard errors adjusted for variability across the sample splits using the median method in parentheses. We consider the same methods with the same tuning choices for estimating the nuisance functions as in the previous example, with one exception, and so do not repeat details for brevity. The one exception is that

Table 3 . Estimated Effect of 401(k) Participation on Net Financial Assets

|               | Lasso       | Reg. Tree           | Forest              | Boosting            | Neural Net.         | Ensemble            | Best                |
|---------------|-------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| LATE (2 fold) | 8978 [2192] | 11073 [1749] (1849) | 11384 [1832] (1993) | 11329 [1666] (1718) | 11094 [1903] (2098) | 11119 [1653] (1689) | 10952 [1657] (1699) |
|               | (3014)      | 11459               | 11764               | 11133               | 11186 [1795]        |                     |                     |
| LATE (5 fold) | 8944        |                     | [1788]              |                     |                     | 11173               | 11113               |
| LATE (5 fold) | [2259]      | [1717]              |                     | [1661]              |                     | [1641]              | [1645]              |
| LATE (5 fold) | (3307)      | (1786)              | (1893)              | (1710)              | (1890)              | (1678)              | (1675)              |

Note: Estimated LATE based on orthogonal estimating equations. Column labels denote the method used to estimate nuisance functions. Results are based on 100 splits with point estimates calculated the median method. The median standard error across the splits are reported in brackets and standard errors calculated using the median method to adjust for variation across splits are provided in parentheses. Further details about the methods are provided in the main text.

we implement neural networks with 8 neurons and a decay parameter of 0.01 in this example.

Turning to the results, it is first worth noting that the estimated ATE of 401(k) eligibility on net financial assets is $ 19,559 with an estimated standard error of 1413 when no control variables are used. Of course, this number is not a valid estimate of the causal effect of 401(k) eligibility on financial assets if there are neglected confounding variables as suggested by Poterba et al. (1994a) and Poterba et al. (1994b). When we turn to the estimates that flexibly account for confounding reported in Table 2, we see that they are substantially attenuated relative to this baseline that does not account for confounding, suggesting much smaller causal effects of 401(k) eligibility on financial asset holdings. It is interesting and reassuring that the results obtained from the different flexible methods are broadly consistent with each other. This similarity is consistent with the theory that suggests that results obtained through the use of orthogonal estimating equations and any sensible method of estimating the necessary nuisance functions should be similar. Finally, it is interesting that these results are also broadly consistent with those reported in the original work of Poterba et al. (1994a) and Poterba et al. (1994b) which used a simple intuitively motivated functional form, suggesting that this intuitive choice was sufficiently flexible to capture much of the confounding variation in this example.

As a further illustration, we also report the LATE in this example where we take the endogenous treatment variable to be participating in a 401(k) plan. Even after controlling for features related to job choice, it seems likely that the actual choice of whether to participate in an offered plan would be endogenous. Of course, we can use eligibility for a 401(k) plan as an instrument for participation in a 401(k) plan under the conditions that were used to justify the exogeneity of eligibility for a 401(k) plan provided above in the discussion of estimation of the ATE of 401(k) eligibility.

We report DML2 results of estimating the LATE of 401(k) participation using 401(k) eligibility as an instrument in Table 3. We employ the procedure outlined in Section 5.2 using the same ML estimators to estimate the quantities used to form the orthogonal estimating equation as we employed to estimate the ATE of 401(k) eligibility outlined previously, so we omit the details for brevity. Looking at the results, we see that the estimated causal effect of 401(k) participation on net financial assets is uniformly positive and statistically significant across all of the considered methods. As when looking at the ATE of 401(k) eligibility, it is reassuring that the results obtained from the different flex-

ible methods are broadly consistent with each other. It is also interesting that the results based on flexible ML methods are broadly consistent with, though somewhat attenuated relative to, those obtained by applying the same specification for controls as used in Poterba et al. (1994a) and Poterba et al. (1994b) and using a linear IV model which returns an estimated effect of participation of $ 13,102 with estimated standard error of (1922). The mild attenuation may suggest that the simple intuitive control specification used in the original baseline specification is somewhat too simplistic.

Looking at Tables 2 and 3, there are other interesting observations that can provide useful insights into understanding the finite sample properties of the DML estimation method. First, the standard errors of the estimates obtained using 5-fold cross-fitting are lower than those obtained from 2-fold cross-fitting for all methods across all cases. This fact suggests that having more observations in the auxiliary sample may be desirable. Specifically, the 5-fold cross-fitting estimates use more observations to learn the nuisance functions than 2-fold cross-fitting and thus likely learn them more precisely. This increase in precision in learning the nuisance functions may then translate into more precisely estimated parameters of interest. While intuitive, we note that this statement does not seem to be generalizable in that there does not appear to be a general relationship between the number of folds in cross-fitting and the precision of the estimate of the parameter of interest; see the next example. Second, we also see that the standard errors of the Lasso estimates after adjusting for variation due to sample splitting are noticeably larger than the standard errors coming from the other ML methods. We believe that this is due to the fact that the out-of-sample prediction errors from a linear model tend to be larger when there is a need to extrapolate. In our framework, if the main sample includes observations that are outside of the range of the observations in the auxiliary sample, the model has to extrapolate to those observations. The fact that the standard errors are lower in 5-fold cross-fitting than in 2-fold cross-fitting for the 'Lasso' estimations also supports this hypothesis, because the higher number of observations in the auxiliary sample reduces the degree of extrapolation. We also see that there is a noticeable increase in the standard errors that account for variability due to sample splitting relative to the simple unadjusted standard errors in this case, though these differences do not qualitatively change the results.

## 6.3. The Effect of Institutions on Economic Growth

To demonstrate DML estimation of partially linear structural equation models with instrumental variables, we consider estimation of the effect of institutions on aggregate output following the work of Acemoglu et al. (2001) (AJR). Estimating the effect of institutions on output is complicated by the clear potential for simultaneity between institutions and output: Specifically, better institutions may lead to higher incomes, but higher incomes may also lead to the development of better institutions. To help overcome this simultaneity, AJR use mortality rates for early European settlers as an instrument for institution quality. The validity of this instrument hinges on the argument that settlers set up better institutions in places where they are more likely to establish long-term settlements; that where they are likely to settle for the long term is related to settler mortality at the time of initial colonization; and that institutions are highly persistent. The exclusion restriction for the instrumental variable is then motivated by the argument that GDP, while persistent, is unlikely to be strongly influenced by mortality in the previous century, or earlier, except through institutions.

Table 4 Estimated Effect of Institutions on Output

DML .

|        | Lasso       | Reg. Tree   | Forest      | Boosting    | Neural Net.        | Ensemble   | Best        |
|--------|-------------|-------------|-------------|-------------|--------------------|------------|-------------|
| 2 fold | 0.85 [0.28] | 0.81 [0.42] | 0.84 [0.38] | 0.77 [0.33] | 0.94 [0.32] (0.28) | 0.8 [0.35] | 0.83 [0.34] |
|        | (0.22)      | (0.29)      | (0.3)       | (0.27)      | 1.00               |            | (0.29)      |
|        | 0.77        | 0.95        | 0.9         | 0.73        |                    | (0.3)      |             |
| 5 fold |             |             |             |             |                    | 0.83       | 0.88        |
| 5 fold | [0.24]      | [0.46]      | [0.41]      | [0.33]      | [0.33]             | [0.37]     | [0.41]      |
| 5 fold | (0.17)      | (0.45)      | (0.4)       | (0.27)      | (0.3)              | (0.34)     | (0.39)      |

Note: Estimated coefficient from a linear instrumental variables model based on orthogonal estimating equations. Column labels denote the method used to estimate nuisance functions. Results are based on 100 splits with point estimates calculated the median method. The median standard error across the splits are reported in brackets and standard errors calculated using the median method to adjust for variation across splits are provided in parentheses. Further details about the methods are provided in the main text.

In their paper, AJR note that their instrumental variable strategy will be invalidated if other factors are also highly persistent and related to the development of institutions within a country and to the country's GDP. A leading candidate for such a factor, as they discuss, is geography. AJR address this by assuming that the confounding effect of geography is adequately captured by a linear term in distance from the equator and a set of continent dummy variables. Using DML allows us to relax this assumption and replace it by a weaker assumption that geography can be sufficiently controlled by an unknown function of distance from the equator and continent dummies which can be learned by ML methods.

We use the same set of 64 country-level observations as AJR. The data set contains measurements of GDP, settler morality, an index which measures protection against expropriation risk and geographic information. The outcome variable, Y, is the logarithm of GDP per capita and the endogenous explanatory variable, D, is a measure of the strength of individual property rights that is used as a proxy for the strength of institutions. To deal with endogeneity, we use an instrumental variable Z, which is mortality rates for early European settlers. Our raw set of control variables, X, include distance from the equator and dummy variables for Africa, Asia, North America, and South America.

We report results from applying DML2 following the procedure outlined in Section 4.2 in Table 4. The considered ML methods and tuning parameters are the same as the previous examples except for the Ensemble method, from which we exclude Neural Network since the small sample size causes stability problems in training the Neural Network. We use the raw set of covariates and all second order terms when doing lasso estimation, and we simply use the raw set of covariates in the remaining methods. As in the previous examples, we consider 100 different sample splits and report the 'Median' estimates of the coefficient and two different standard error estimates. In brackets, we report the median standard error from across the 100 splits; and we report standard errors adjusted for variability across the sample splits using the median method in parentheses. Finally, we report results from both 2-fold cross-fitting and 5-fold cross-fitting as in the other examples.

In this example, we see uniformly large and positive point estimates across all procedures considered, and estimated effects are statistically significant at the 5% level. As in the second example, we see that adjusting for variability across sample splits leads

to noticeable increases in estimated standard errors but does not result in qualitatively different conclusions. Interestingly, we see that the estimated standard errors based on 5-fold cross-fitting are larger than those on twofold cross-fitting in all procedures except lasso, which differs from the finding in the 401(k) example. Further understanding these differences and the impact of the number of folds on inference for objects of interest seems like an interesting question for future research. Finally, although the estimates are somewhat smaller than the baseline estimates reported in AJR - an estimated coefficient of 1.10 with estimated standard error of 0.46 (Acemoglu et al. (2001), Table 4, Panel A, column 7) - the results are qualitatively similar, indicating a strong and positive effect of institutions on output.

## 6.4. Comments on Empirical Results

Before closing this section we want to emphasize some important conclusions that can be drawn from these empirical examples. First, the choice of the ML method used in estimating nuisance functions does not substantively change the conclusion in any of the examples, and we obtained broadly consistent results regardless of which method we employ. The robustness of the results to the different methods is implied by the theory assuming that all of the employed methods are able to deliver sufficiently highquality approximations to the underlying nuisance functions. Second, the incorporation of uncertainty due to sample-splitting using the median method increases the standard errors relative to a baseline that does not account for this uncertainty, though these differences do not alter the main results in any of the examples. This lack of variation suggests that the parameter estimates are robust to the particular sample split used in the estimation in these examples.

## ACKNOWLEDGEMENTS

We would like to acknowledge research support from the National Science Foundation. We also thank participants of the MIT Stochastics and Statistics seminar, the Kansas Econometrics conference, the Royal Economic Society Annual Conference, The Hannan Lecture at the Australasian Econometric Society meeting, The Econometric Theory lecture at the EC 2 meetings 2016 in Toulouse, The CORE 50th Anniversary Conference, The Becker-Friedman Institute Conference on Machine Learning and Economics, The INET conferences at USC on Big Data, the World Congress of Probability and Statistics 2016, the Joint Statistical Meetings 2016, the New England Day of Statistics Conference, CEMMAP's Masterclass on Causal Machine Learning, and St. Gallen's summer school on 'Big Data', for many useful comments and questions. We would like to thank Susan Athey, Peter Aronow, Jin Hahn, Guido Imbens, Mark van der Laan, Matt Taddy, and Rui Wang for constructive comments. We thank Peter Aronow for pointing us to the literature on targeted learning on which, along with prior works of Neyman, Bickel, and the many other contributions to semiparametric learning theory, we build.

## REFERENCES

- Abadie, A. and G. W. Imbens (2006). Large sample properties of matching estimators for average treatment effects. Econometrica 74 , 235-267.
- Acemoglu, D., S. Johnson, and J. A. Robinson (2001). The colonial origins of comparative development: An empirical investigation. American Economic Review 91 , 1369-1401.

- Ai, C. and X. Chen (2012). The semiparametric efficiency bound for models of sequential moment restrictions containing unknown functions. Journal of Econometrics 170 , 442457.
- Andrews, D. W. K. (1994a). Asymptotics for semiparametric econometric models via stochastic equicontinuity. Econometrica 62 , 43-72.
- Andrews, D. W. K. (1994b). Empirical process methods in econometrics. Handbook of Econometrics, Volume IV, Chapter 37 , 2247-2294.
- Angrist, J. D. and A. B. Krueger (1995). Split-sample instrumental variables estimates of the return to schooling. Journal of Business and Economic Statistics 13 , 225-235.
- Athey, S., G. Imbens, and S. Wager (2016). Approximate residual balancing: De-biased inference of average treatment effects in high-dimensions. arXiv:1604.07125v3 . arXiv, 2016.
- Ayyagari, R. (2010). Applications of influence functions to semiparametric regression models. Ph.D. Thesis, Harvard School of Public Health, Harvard University.
- Belloni, A., D. Chen, V. Chernozhukov, and C. Hansen (2012). Sparse models and methods for optimal instruments with an application to eminent domain. Econometrica 80 , 2369-2429. arXiv, 2010.
- Belloni, A. and V. Chernozhukov (2011). ℓ 1 -penalized quantile regression for high dimensional sparse models. Annals of Statistics 39 , 82-130. arXiv, 2009.
- Belloni, A. and V. Chernozhukov (2013). Least squares after model selection in highdimensional sparse models. Bernoulli 19 , 521-547. arXiv, 2009.
- Belloni, A., V. Chernozhukov, I. Fern´ andez-Val, and C. Hansen (2017). Program evaluation with high-dimensional data. Econometrica 85 , 233--298. arXiv, 2013.
- Belloni, A., V. Chernozhukov, and C. Hansen (2010). Lasso methods for gaussian instrumental variables models. arXiv:1012.1297 .
- Belloni, A., V. Chernozhukov, and C. Hansen (2013). Inference for high-dimensional sparse econometric models. Advances in Economics and Econometrics. 10th World Congress of Econometric Society. August 2010 , III:245-295. arXiv, 2011.
- Belloni, A., V. Chernozhukov, and C. Hansen (2014). Inference on treatment effects after selection amongst high-dimensional controls. Review of Economic Studies 81 , 608-650. arXiv, 2011.
- Belloni, A., V. Chernozhukov, and K. Kato (2015). Uniform post selection inference for lad regression models and other z-estimators. Biometrika 102 , 77-94. arXiv, 2013.
- Belloni, A., V. Chernozhukov, and L. Wang (2011). Square-root-lasso: Pivotal recovery of sparse signals via conic programming. Biometrika 98 , 791-806. arXiv, 2010.
- Belloni, A., V. Chernozhukov, and L. Wang (2014). Pivotal estimation via square-root lasso in nonparametric regression. Annals of Statistics 42 , 757-788. arXiv, 2011.
- Belloni, A., V. Chernozhukov, and Y. Wei (2016). Post-selection inference for generalized linear models with many controls. Journal of Business and Economic Statistics 34 , 606-619. arXiv, 2013.
- Bera, A., G. Montes-Rojas, and W. Sosa-Escudero (2010). General specification testing with locally misspecified models. Econometric Theory 26 , 1838-1845.
- Bickel, P. and Y. Ritov (1988). Estimating integrated squared density derivatives. Sankhya A-50 , 381-393.
- Bickel, P. J. (1982). On adaptive estimation. Annals of Statistics 10 , 647-671.
- Bickel, P. J., C. A. J. Klaassen, Y. Ritov, and J. A. Wellner (1998). Efficient and Adaptive Estimation for Semiparametric Models . Springer.
- Bickel, P. J., Y. Ritov, and A. Tsybakov (2009). Simultaneous analysis of Lasso and Dantzig selector. Annals of Statistics 37 , 1705-1732. arXiv, 2008.

- Bilias, Y. (2000). Sequential testing of duration data: The case of the Pennsylvania 'reemployment bonus' experiment. Journal of Applied Econometrics 15 , 575-594.
- Bilias, Y. and R. Koenker (2002). Quantile regression for duration data: A reappraisal of the pennsylvania reemployment bonus experiments. In B. Fitzenberger, R. Koenker, and J. A. Machado (Eds.), Studies in Empirical Economics: Economic Applications of Quantile Regression , pp. 199-220. Physica-Verlag Heidelberg.
- B¨ uhlmann, P. and S. van de Geer (2011). Statistics for High-Dimensional Data . Springer Series in Statistics.
- Chamberlain, G. (1987). Asymptotic efficiency in estimation with conditional moment restrictions. Journal of Econometrics 34 , 305-334.
- Chamberlain, G. (1992). Efficiency bounds for semiparametric regression. Econometrica 60 , 567-596.
- Chen, X., O. Linton, and I. van Keilegom (2003). Estimation of semiparametric models when the criterion function is not smooth. Econometrica 71 , 1591-1608.
- Chen, X. and H. White (1999). Improved rates and asymptotic normality for nonparametric neural network estimators. IEEE Transactions on Information Theory 45 , 682-691.
- Chernozhukov, V., D. Chetverikov, and K. Kato (2014). Gaussian approximation of suprema of empirical processes. The Annals of Statistics 42 , 1564-1597. arXiv, 2012.
- Chernozhukov, V., J. Escanciano, H. Ichimura, W. Newey, and J. Robins (2016). Locally robust semiparametric estimation. arXiv:1608.00033 . arXiv, 2016.
- Chernozhukov, V. and C. Hansen (2004). The effects of 401 (k) participation on the wealth distribution: an instrumental quantile regression analysis. Review of Economics and statistics 86 , 735-751.
- Chernozhukov, V., C. Hansen, and M. Spindler (2015a). Post-selection and postregularization inference in linear models with very many controls and instruments. Americal Economic Review: Papers and Proceedings 105 , 486-490.
- Chernozhukov, V., C. Hansen, and M. Spindler (2015b). Valid post-selection and postregularization inference: An elementary, general approach. Annual Review of Economics 7 , 649-688.
- DasGupta, A. (2008). Asymptotic Theory of Statistics and Probability . Springer Texts in Statistics.
- Fan, J., S. Guo, and K. Yu (2012). Variance estimation using refitted cross-validation in ultrahigh dimensional regression. Journal of the Royal Statistical Society, Series B 74 , 37-65.
- Farrell, M. (2015). Robust inference on average treatment effects with possibly more covariates than observations. Journal of Econometrics 174 , 1-23.
- Ferguson, T. (1967). Mathematical Statistics: A Decision Theoretic Approach . Academic Press.
- Fr¨ olich, M. (2007). Nonparametric IV estimation of local average treatment effects with covariates. Journal of Econometrics 139 , 35-75.
- Gautier, E. and A. Tsybakov (2014). High-dimensional instrumental variables regression and confidence sets. arXiv:1105.2454 . arXiv, 2011.
- Hahn, J. (1998). On the role of the propensity score in efficient semiparametric estimation of average treatment effects. Econometrica 66 , 315-331.
- Hansen, L. (1982). Large sample properties of generalized method of moments estimators. Econometrica 50 , 1029-1054.
- Hasminskii, R. and I. Ibragimov (1978). On the nonparametric estimation of functionals. Proceedings of the 2nd Prague Symposium on Asymptotic Statistics , 41-51.

- Hirano, K., G. W. Imbens, and G. Ridder (2003). Efficient estimation of average treatment effects using the estimated propensity score. Econometrica 71 , 1161-1189.
- Hubbard, A. E., S. Kherad-Pajouh, and M. J. van der Laan (2016). Statistical inference for data adaptive target parameters. International Journal of Biostatistics 12 , 3-19.
- Ibragimov, I. A. and R. Z. Hasminskii (1981). Statistical Estimation: Asymptotic Theory . Springer-Verlag, New York.
- Ichimura, H. and W. Newey (2015). The influence function of semiparametric estimators. arXiv:1508.01378 . arXiv, 2015.
- Imai, K. and M. Ratkovic (2013). Estimating treatment effect heterogeneity in randomized program evaluation. Annals of Applied Statistics 7 , 443-470.
- Imbens, G. and J. Angrist (1994). Identification and estimation of local average treatment effects. Econometrica , 467-475.
- Imbens, G. W. and D. B. Rubin (2015). Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction . Cambridge University Press.
- Javanmard, A. and A. Montanari (2014a). Confidence intervals and hypothesis testing for high-dimensional regression. Journal of Machine Learning Research 15 , 2869-2909.
- Javanmard, A. and A. Montanari (2014b). Hypothesis testing in high-dimensional regression under the gaussian random design model: Asymptotic theory. IEEE Transactions on Information Theory 60 , 6522-6554. arXiv, 2013.
- Kozbur, D. (2016). Testing-based forward model selection. arXiv:1512.02666 . arXiv, 2015.
- Lee, L. (2005). A c ( α )-type gradient test in the GMM approach. Working paper, The Ohio State University.
- Levit, B. Y. (1975). On the efficiency of a class of nonparametric estimates. Theory of Probability and Its Applications 20 , 723-740.
- Linton, O. (1996). Edgeworth approximation for MINPIN estimators in semiparametric regression models. Econometric Theory 12 , 30-60.
- Luedtke, A. R. and M. J. van der Laan (2016). Optimal individualized treatments in resource-limited settings. The International Journal of Biostatistics 12 , 283-303.
- Luo, Y. and M. Spindler (2016). High-dimensional l 2 boosting: Rate of convergence. arXiv:1602.08927 . arXiv, 2016.
- Nevelson, M. (1977). On one informational lower bound. Problemy Peredachi Informatsii 13 , 26-31.
- Newey, W. (1990). Semiparametric efficiency bounds. Journal of Applied Econometrics 5 , 99-135.
- Newey, W. (1994). The asymptotic variance of semiparametric estimators. Econometrica 62 , 1349-1382.
- Newey, W. K., F. Hsieh, and J. Robins (1998). Undersmoothing and bias corrected functional estimation. Working paper, MIT Economics Dept., http://economics.mit.edu/files/11219.
- Newey, W. K., F. Hsieh, and J. M. Robins (2004). Twicing kernels and a small bias property of semiparametric estimators. Econometrica 72 , 947-962.
- Neyman, J. (1959). Optimal asymptotic tests of composite statistical hypotheses. In U. Grenander (Ed.), Probability and Statistics , pp. 416--444. New York, John Wiley. Neyman, J. (1979). c ( α ) tests and their use. Sankhya , 1-21.
- Poterba, J. M., S. F. Venti, and D. A. Wise (1994a). 401(k) plans and tax-deferred savings. In D. Wise (Ed.), Studies in the Economics of Aging , pp. 105-142. Chicago: University of Chicago Press.

- Poterba, J. M., S. F. Venti, and D. A. Wise (1994b). Do 401(k) contributions crowd out other personal saving? Journal of Public Economics 58 , 1-32.
- Robins, J., L. Li, R. Mukherjee, E. Tchetgen, and A. van der Vaart (2017). Minimax estimation of a functional on a structured high dimensional model. Annals of Statistics, forthcoming .
- Robins, J., L. Li, E. Tchetgen, and A. van der Vaart (2008). Higher order influence functions and minimax estimation of nonlinear functionals. In D. Nolan and T. Speed (Eds.), Probability and Statistics: Essays in Honor of David A. Freedman , pp. 335-421. Institute of Mathematical Statistics.
- Robins, J. and A. Rotnitzky (1995). Semiparametric efficiency in multivariate regression models with missing data. Journal of the American Statistical Association 90 , 122-129.
- Robins, J., P. Zhang, R. Ayyagari, R. Logan, E. Tchetgen, L. Li, A. Lumley, and A. van der Vaart (2013). New statistical approaches to semiparametric regression with application to air pollution research. Research Report 175, Health Effects Institute.
- Robinson, P. M. (1988). RootN -consistent semiparametric regression. Econometrica 56 , 931-954.
- Rosenbaum, P. R. and D. B. Rubin (1983). The central role of the propensity score in observational studies for causal effects. Biometrika 70 , 41-55.
- Scharfstein, D. O., A. Rotnitzky, and J. M. Robins (1999). Rejoinder to 'adjusting for non-ignorable drop-out using semiparametric non-response models'. Journal of the American Statistical Association 94 , 1135-1146.
- Schick, A. (1986). On asymptotically efficient estimation in semiparametric models. Annals of Statistics 14 , 1139-1151.
- Severini, T. A. and W. H. Wong (1992). Profile likelihood and conditionally parametric models. The Annals of Statistics 20 , 1768-1802.
- Toth, B. and M. J. van der Laan (2016). TMLE for marginal structural models based on an instrument. Working Paper 350, U.C. Berkeley Division of Biostatistics Working Paper Series.
- van de Geer, S., P. B¨ uhlmann, Y. Ritov, and R. Dezeure (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. Annals of Statistics 42 , 1166-1202. arXiv, 2013.
- van der Laan, M. and D. Rubin (2006). Targeted maximum likelihood learning. Working Paper 213, UC Berkeley Division of Biostatistics Working Paper Series.
- van der Laan, M. J. (2015). A generally efficient targeted minimum loss based estimator. Working Paper 343, U.C. Berkeley Division of Biostatistics Working Paper Series.
- van der Laan, M. J., E. C. Polley, and A. E. Hubbard (2007). Super learner. Statistical Applications in Genetics and Molecular Biology 6 . Retrieved 24 Feb. 2017, from doi:10.2202/1544-6115.1309.
- van der Laan, M. J. and S. Rose (2011). Targeted Learning: Causal Inference for Observational and Experimental Data . Springer.
- van der Vaart, A. W. (1991). On differentiable functionals. Annals of Statistics 19 , 178-204.
- van der Vaart, A. W. (1998). Asymptotic Statistics . Cambridge University Press.
- Wager, S. and G. Walther (2016). Adaptive concentration of regression trees, with application to random forests. arXiv:1503.06388 . arXiv, 2015.
- Wooldridge, J. (1991). Specification testing and quasi-maximum-likelihood estimation. Journal of Econometrics 48 , 29-55.

- Zhang, C. and S. Zhang (2014). Confidence intervals for low-dimensional parameters with high-dimensional data. Journal of the Royal Statistical Society, Series B 76 , 217-242. arXiv, 2012.
- Zheng, W., Z. Luo, and M. J. van der Laan (2016). Marginal structural models with counterfactual effect modifiers. Working Paper 348, U.C. Berkeley Division of Biostatistics Working Paper Series.
- Zheng, W. and M. J. van der Laan (2011). Cross-validated targeted minimum-loss-based estimation. In Targeted Learning , pp. 459-474. Springer.

## APPENDIX: PROOFS OF RESULTS

In this appendix, we use C to denote a strictly positive constant that is independent of n and P ∈ P N . The value of C may change at each appearance. Also, the notation a N ≲ b N means that a N ⩽ Cb N for all n and some C . The notation a N ≳ b N means that b N ≲ a N . Moreover, the notation a N = o (1) means that there exists a sequence ( b N ) n ⩾ 1 of positive numbers such that (a) | a N | ⩽ b N for all n , (b) b N is independent of P ∈ P N for all n , and (c) b N → 0 as n →∞ . Finally, the notation a N = O P ( b N ) means that for all ϵ &gt; 0, there exists C such that P P ( a N &gt; Cb N ) ⩽ 1 -ϵ for all n . Using this notation allows us to avoid repeating 'uniformly over P ∈ P N ' many times in the proofs.

Define the empirical process G n ( ψ ( W )) as a linear operator acting on measurable functions ψ : W → R such that ∥ ψ ∥ P, 2 &lt; ∞ via,

$$Gn(ψ(W)) := Gn,I(ψ(W)) :=
1
√n
X
i∈I
ψ(Wi) −
Z
ψ(w)dP(w).$$

Analogously, we defined the empirical expectation as:

$$En(ψ(W)) := En,I(ψ(W)) := 1
n
X
i∈I
ψ(Wi).$$

A.5. Useful Lemmas

The following lemma is useful particularly in the sample-splitting contexts.

Lemma 6.1. (Conditional Convergence Implies Unconditional) Let { X m } and { Y m } be sequences of random vectors. (a) If for ϵ m → 0 , P( ∥ X m ∥ &gt; ϵ m | Y m ) → P 0 , then P( ∥ X m ∥ &gt; ϵ m ) → 0 . In particular, this occurs if E[ ∥ X m ∥ q /ϵ q m | Y m ] → P 0 for some q ⩾ 1 , by Markov's inequality. (b) Let { A m } be a sequence of positive constants. If ∥ X m ∥ = O P ( A m ) conditional on Y m , namely, that for any ℓ m →∞ , P( ∥ X m ∥ &gt; ℓ m A m | Y m ) → P 0 , then ∥ X m ∥ = O P ( A m ) unconditionally, namely, that for any ℓ m → ∞ , P( ∥ X m ∥ &gt; ℓ m A m ) → 0 .

Proof . Part (a). For any ϵ &gt; 0 P( ∥ X m ∥ &gt; ϵ m ) ⩽ E[P( ∥ X m ∥ &gt; ϵ m | Y m )] → 0, since the sequence { P( ∥ X m ∥ &gt; ϵ m | Y m ) } is uniformly integrable. To show the second part note that P( ∥ X m ∥ &gt; ϵ m | Y m ) ⩽ E[ ∥ X m ∥ q /ϵ q m | Y m ] ∨ 1 → P 0 by Markov's inequality. Part (b). This follows from Part (a). ■

Let ( W i ) n i =1 be a sequence of independent copies of a random element W taking values in a measurable space ( W , A W ) according to a probability law P . Let F be a set of

suitably measurable functions f : W → R , equipped with a measurable envelope F : W → R .

Lemma 6.2. (Maximal Inequality, Chernozhukov et al. (2014)) Work with the setup above. Suppose that F ⩾ sup f ∈F | f | is a measurable envelope for F with ∥ F ∥ P,q &lt; ∞ for some q ⩾ 2 . Let M = max i ⩽ n F ( W i ) and σ 2 &gt; 0 be any positive constant such that sup f ∈F ∥ f ∥ 2 P, 2 ⩽ σ 2 ⩽ ∥ F ∥ 2 P, 2 . Suppose that there exist constants a ⩾ e and v ⩾ 1 such that

$$log sup
Q
N(ϵ∥F∥Q,2, F, ∥· ∥Q,2) ⩽v log(a/ϵ), 0 < ϵ ⩽1.$$

Then

$$EP [∥Gn∥F] ⩽K
 s
vσ2 log
a∥F∥P,2
σ

+ v∥M∥P,2
√n
log
a∥F∥P,2
σ
!
,$$

where K is an absolute constant. Moreover, for every t ⩾ 1 , with probability &gt; 1 -t -q/ 2 ,

$$∥Gn∥F ⩽(1+α)EP [∥Gn∥F]+K(q)
h
(σ+n−1/2∥M∥P,q)
√
t+α−1n−1/2∥M∥P,2t
i
, ∀α > 0,$$

where K ( q ) &gt; 0 is a constant depending only on q . In particular, setting a ⩾ n and t = log n , with probability &gt; 1 -c (log n ) -1 ,

$$∥Gn∥F ⩽K(q, c)

σ
s
v log
a∥F∥P,2
σ

+ v∥M∥P,q
√n
log
a∥F∥P,2
σ
!
,
(A.1)$$

where ∥ M ∥ P,q ⩽ n 1 /q ∥ F ∥ P,q and K ( q, c ) &gt; 0 is a constant depending only on q and c .

## A.6. Proof of Lemma 2.1

Proof. Since J exists and J ββ is invertible, (2.8) has the unique solution µ 0 given in (2.10), and so we have by (2.6) that E[ ψ ( W ; θ 0 , η 0 )] = 0 for η 0 given in (2.9). Moreover,

$$∂η′EP ψ(W; θ0, η0) =

[Jθβ −µ0Jββ], E[∂β′ℓ(W; θ0, β0)] ⊗Idθ×dθ

= 0,$$

where I d θ × d θ is the d θ × d θ identity matrix and ⊗ is the Kronecker product. Hence, the asserted claim holds by the remark after Definition 2.1. ■

## A.7. Proof of Lemma 2.2

The proof follows similarly to that of Lemma 2.1, except that now we have to verify (2.4) intead of (2.3). To do so, take any β ∈ B such that ∥ β -β 0 ∥ ∗ q ⩽ λ N /r N and any d θ × d β matrix µ . Denote η = ( β ′ , vec( µ ) ′ ) ′ . Then

$$∂ηEP ψ(W, θ0, η0)[η −η0]



 =



(Jθβ −µ0Jββ)(β −β0)



⩽∥Jθβ −µ0Jββ∥q × ∥β −β0∥∗
q ⩽rn × (λN/rN) = λN.$$

This completes the proof of the lemma.

■

## A.8. Proof of Lemma 2.3

The proof is similar to that of Lemma 2.1, except that now we have

$$∂η′EP ψ(W, θ0, η0) = [µ0Gβ, EP m(W, θ0, β0)′ ⊗Idθ×dθ] = 0,$$

where I d θ × d θ is the d θ × d θ identity matrix and ⊗ is the Kronecker product.

## A.9. Proof of Lemma 2.4

The proof follows similarly to that of Lemma 2.2, except that now for any β ∈ B such that ∥ β -β 0 ∥ 1 ⩽ λ N /r N , any d θ × k matrix µ , and η = ( β ′ , vec( µ ) ′ ) ′ , we have

$$∂ηEP ψ(W, θ0, η0)[η −η0]



 = ∥µ0Gβ(β −β0)∥
⩽∥A′Ω−1/2L −γ0L′L∥∞× ∥β −β0∥1
⩽rn × (λN/rN) = λN.$$

This completes the proof of the lemma.

## A.10. Proof of Lemma 2.5

Take any η ∈ T , and consider the function

$$Q(W; θ, r) := ℓ(W; θ, η0(θ) + r(η(θ) −η0(θ))),
θ ∈Θ, r ∈[0, 1].$$

Then and so

$$∂rEP [ψ(W; θ, η0 + r(η −η0))] = ∂rEP [∂θQ(W; θ, r)]
= ∂r∂θEP [Q(W; θ, r)] = ∂θ∂rEP [Q(W; θ, r)]
(A.2)
= ∂θ∂rEP [ℓ(W; θ, η0(θ) + r(η(θ) −η0(θ)))].$$

Hence, since

$$∂rEP [ℓ(W; θ, η0(θ) + r(η(θ) −η0(θ)))]

r=0 = 0,
for all θ ∈Θ,$$

as η 0 ( θ ) = β θ solves the optimization problem

$$max
β∈B EP [ℓ(W; θ, β)],
for all θ ∈Θ.$$

Here the regularity conditions are needed to make sure that we can interchange E P and ∂ θ and also ∂ θ and ∂ r in (A.2). This completes the proof of the lemma.

$$ψ(W; θ, η0 + r(η −η0)) = ∂θQ(W; θ, r),$$

$$∂rEP [ψ(W; θ, η0 + r(η −η0))]

r=0 = 0$$

■

## A.11. Proof of Lemma 2.6

First, we demonstrate that µ 0 ∈ L 1 ( R ; R d θ × d m ). Indeed,

$$EP [∥µ0(R)∥] ⩽EP
h
∥A(R)′Ω(R)−1∥
i
+ EP
h
∥G(Z)Γ(R)Ω(R)−1∥
i
⩽EP
h
∥A(R)∥× ∥Ω(R)∥−1i
+ EP
h
∥G(Z)∥× ∥Γ(R)∥× ∥Ω(R)∥−1i
⩽

EP [∥A(R)∥2] × EP [∥Ω(R)∥−2]
1/2
+

EP
h
∥G(Z)∥2 × ∥Γ(R)∥2i
× EP [∥Ω(R)∥−2]
1/2
,$$

which is finite by assumptions of the lemma since

$$EP
h
∥G(Z)∥2 × ∥Γ(R)∥2i
⩽

EP [∥G(Z)∥4] × EP [Γ(R)∥4]
1/2
< ∞.$$

Next, we demonstrate that

$$EP [∥ψ(W, θ0, η)∥] < ∞
for all η ∈T.$$

Indeed, for all η ∈ T , there exist µ ∈ L 1 ( R ; R d θ × d m ) and h ∈ H such that η = ( µ, h ), and so

$$EP [∥ψ(W, θ0, η)∥] = EP [∥µ(X)m(W, θ0, h(Z))∥]
⩽EP
h
∥µ(R)∥× ∥m(W, θ0, h(Z))∥
i
= EP
h
∥µ(R)∥× EP [∥m(W, θ0, h(Z)) | R]
i
⩽ChE[∥µ(R)∥],$$

which is finite by assumptions of the lemma. Further, (2.1) holds because

$$EP [ψ(W, θ0, η0)] = EP
h
µ0(R)m(W, θ0, h0(Z))
i
= EP
h
µ0(R)EP [m(W, θ0, h0(Z)) | R]
i
= 0,
(A.3)$$

where the last equality follows from (2.22).

Finally, we demonstrate that (2.3) holds. To do so, take any η = ( µ, h ) ∈ T N = T . Then

$$EP [ψ(W, θ0, η0 + r(η −η0)]
= EP
h
(µ0(R) + r(µ(R) −µ0(R)))m(W, θ0, h0(Z) + r(h(Z) −h0(Z)))
i
,$$

and so where

$$∂ηEP ψ(W, θ0, η0)[η −η0] = I1 + I2,$$

$$I1 = EP
h
(µ(R) −µ0(R))m(W, θ0, h0(Z)
i
,
I2 = EP
h
µ0(R)∂v′m(W, θ0, v)|v=h0(Z)(h(Z) −h0(Z))
i
.$$

Here I 1 = 0 by the same argument as that in (A.3) and I 2 = 0 because

$$I2 = EP
h
µ0(R)EP [∂v′m(W, θ0, v)|v=h0(Z) | X](h(Z) −h0(Z))
i
= EP
h
µ0(R)Γ(X)(h(Z) −h0(Z))
i
= EP
h
EP [µ0(R)Γ(R) | Z](h(Z) −h0(Z))
i
= 0$$

since

$$EP [µ0(R)Γ(X) | Z] = EP [A(R)′Ω(R)−1Γ(R) | Z] −EP [G(Z)Γ(R)′Ω(R)−1Γ(R) | Z]
= EP [A(R)′Ω(R)−1Γ(R) | Z] −G(Z)EP [Γ(R)′Ω(R)−1Γ(R) | Z]
= EP [A(R)′Ω(R)−1Γ(R) | Z] −EP [A(R)′Ω(R)−1Γ(R) | Z]
×

EP [Γ(R)′Ω(R)−1Γ(R) | Z]
−1
× EP [Γ(R)′Ω(R)−1Γ(R) | Z]
= EP [A(R)′Ω(R)−1Γ(R) | Z] −EP [A(R)′Ω(R)−1Γ(R) | Z] = 0.$$

This completes the proof of the lemma.

## A.12. Proof of Theorem 3.1 (DML2 case)

To start with, note that (3.11) follows immediately from the assumptions. Hence, it suffices to show that (3.10) holds uniformly over P ∈ P N .

Fix any sequence { P N } N ⩾ 1 such that P N ∈ P N for all N ⩾ 1. Since this sequence is chosen arbitrarily, to show that (3.10) holds uniformly over P ∈ P N , it suffices to show that

$$√
Nσ−1(˜θ0 −θ0) =
1
√
N
N
X
i=1
¯ψ(Wi) + OPN (ρN) ; N(0, Id).
(A.4)$$

To do so, we proceed in 5 steps. Step 1 shows the main argument, and Steps 2-5 present auxiliary calculations. In the proof, it will be convenient to denote by E N the event that ̂ η 0 ,k ∈ T N for all k ∈ [ K ]. Note that by Assumption 3.2 and the union bound, P P N ( E N ) ⩾ 1 -K ∆ n = 1 -o (1) since ∆ n = o (1).

Step 1. Denote

$$bJ0 := 1
K
K
X
k=1
En,k[ψa(W; bη0,k)],
RN,1 := bJ0 −J0,$$

$$RN,2 := 1
K
K
X
k=1
En,k[ψ(W; θ0, bη0,k)] −1
N
N
X
i=1
ψ(Wi; θ0, η0).$$

In Steps 2, 3, 4, and 5 below, we will show that

$$∥RN,1∥= OPN (N −1/2 + rN),
(A.5)$$

$$∥RN,2∥= OPN (N −1/2r′
N + λN + λ′
N),
(A.6)$$

$$∥N −1/2 PN
i=1 ψ(Wi; θ0, η0)∥= OPN (1),
(A.7)$$

$$∥σ−1∥= OPN (1),
(A.8)$$

respectively. Since N -1 / 2 + r N ⩽ ρ N = o (1) and all singular values of J 0 are bounded below from zero by Assumption 3.1, it follows from (A.5) that with P N -probability 1 -

■

o (1), all singular values of ̂ J 0 are bounded below from zero as well. Therefore, with the same P N -probability,

$$˜θ0 = −bJ−1
0
1
K
K
X
k=1
En,k[ψb(W; bη0,k)]$$

and

$$√
N(˜θ0 −θ0) = −
√
N bJ−1
0
 1
K
K
X
k=1
En,k[ψb(W; bη0,k)] + bJ0θ0

= −
√
N bJ−1
0
1
K
K
X
k=1
En,k[ψ(W; θ0, bη0,k)]
= −

J0 + RN,1
−1
×
 1
√
N
N
X
i=1
ψ(Wi; θ0, η0) +
√
NRN,2

.
(A.9)$$

In addition, given that

$$(J0 + RN,1)−1 −J−1
0
= (J0 + RN,1)−1(J0 −(J0 + RN,1))J−1
0
= −(J0 + RN,1)−1RN,1J−1
0 ,$$

it follows from (A.5) that

$$∥(J0 + RN,1)−1 −J−1
0 ∥⩽∥(J0 + RN,1)−1∥× ∥RN,1∥× ∥J−1
0 ∥
= OPN (1)OPN (N −1/2 + rN)OPN (1) = OPN (N −1/2 + rN).
(A.10)$$

√

Moreover, since r ′ N + N ( λ N + λ ′ N ) ⩽ ρ N = o (1), it follows from (A.6) and (A.7) that

$$1
√
N
N
X
i=1
ψ(Wi; θ0, η0) +
√
NRN,2



 ⩽



 1
√
N
N
X
i=1
ψ(Wi; θ0, η0)



 +



√
NRN,2



= OPN (1) + oPN (1) = OPN (1).
(A.11)$$

Combining (A.10) and (A.11) gives

$$
(J0 + RN,1)−1 −J−1
0

×
 1
√
N
N
X
i=1
ψ(Wi; θ0, η0) +
√
NRN,2



⩽



(J0 + RN,1)−1 −J−1
0



 ×



 1
√
N
N
X
i=1
ψ(Wi; θ0, η0) +
√
NRN,2



 = OPN (N −1/2 + rN).$$

Now, substituting the last bound into (A.9) yields

$$√
N(˜θ0 −θ0) = −J−1
0
×
 1
√
N
N
X
i=1
ψ(Wi; θ0, η0) +
√
NRN,2

+ OPN (N −1/2 + rN)
= −J−1
0
×
1
√
N
N
X
i=1
ψ(Wi; θ0, η0) + OPN (ρN),$$

where in the second line we used (A.6) and the definition of ρ N . Combining this with

(A.8) gives

$$√
Nσ−1(˜θ0 −θ0) =
1
√
N
N
X
i=1
¯ψ(Wi) + OPN (ρN)
(A.12)$$

by the definition of ¯ ψ given in the statement of the theorem. In turn, since ρ N = o (1), combining (A.12) with the Lindeberg-Feller CLT and the Cramer-Wold device yields (A.4). To complete the proof of the theorem, it remains to establish the bounds (A.5)(A.8). We do so in four steps below.

Step 2. Here we establish (A.5). Since K is a fixed integer, which is independent of N , it suffices to show that for any k ∈ [ K ],

$$En,k[ψa(W; bη0,k)] −EPN [ψa(W; η0)]



 = OPN (N −1/2 + rN).
(A.13)$$

To do so, fix any k ∈ [ K ] and observe that by the triangle inequality,

$$En,k[ψa(W; bη0,k)] −EPN [ψa(W; η0)]



 ⩽I1,k + I2,k,
(A.14)$$

where

$$I1,k :=



En,k[ψa(W; bη0,k)] −EPN [ψa(W; bη0,k) | (Wi)i∈Ic
k]



,
I2,k :=



EPN [ψa(W; bη0,k) | (Wi)i∈Ic
k] −EPN [ψa(W; η0)]



.$$

$$EPN [ψ (W; η0,k) | (Wi)i∈Ik]
EPN [ψ (W; η0)]$$

To bound I 2 ,k , note that on the event E N , which holds with P N -probability 1 -o (1),

$$I2,k ⩽sup
η∈TN



EPN [ψa(W; η)] −EPN [ψa(W; η0)]



 = rN,$$

and so I 2 ,k = O P N ( r N ). To bound I 1 ,k , note that conditional on ( W i ) i ∈ I c k , the estimator ̂ η 0 ,k is non-stochastic, and so on the event E N ,

$$EPN [I2
1,k | (Wi)i∈Ic
k] ⩽n−1EPN [∥ψa(W; bη0,k)∥2 | (Wi)i∈Ic
k]
⩽sup
η∈TN
n−1EPN [∥ψa(W; η)∥2] ⩽c2
1/n,$$

where the last inequality holds by Assumption 3.2. Hence, I 1 ,k = O P N ( N -1 / 2 ) by Lemma 6.1 in the Appendix. Combining the bounds I 1 ,k = O P N ( N -1 / 2 ) and I 2 ,k = O P N ( r N ) with (A.14) gives (A.13).

Step 3. Here we establish (A.6). This is the step where we invoke the Neyman orthogonality (or near-orthogonality) condition. Again, since K is a fixed integer, which is independent of N , it suffices to show that for any k ∈ [ K ],

$$En,k[ψ(W; θ0, bη0,k)] −1
n
X
i∈Ik
ψ(Wi; θ0, η0) = OPN (N −1/2r′
N + λN + λ′
N).
(A.15)$$

To do so, fix any k ∈ [ K ] and introduce the following additional empirical process notation:

$$Gn,k[ϕ(W)] =
1
√n
X
i∈Ik

ϕ(Wi) −
Z
ϕ(w)dPN

,$$

where ϕ is any P N -integrable function on W . Then observe that by the triangle inequality,

$$En,k[ψ(W; θ0, bη0,k)] −1
n
X
i∈Ik
ψ(Wi; θ0, η0)



 ⩽I3,k + I4,k
√n
,
(A.16)$$

where

$$I3,k :=



Gn,k[ψ(W; θ0, bη0,k)] −Gn,k[ψ(W; θ0, η0)]



,
I4,k := √n



EPN [ψ(W; θ0, bη0,k) | (Wi)i∈Ic
k] −EPN [ψ(W; θ0, η0)]



.$$

To bound I 3 ,k , note that, as above, conditional on ( W i ) i ∈ I c k , the estimator ̂ η 0 ,k is nonstochastic, and so on the event E N ,

$$EPN [I2
3,k | (Wi)i∈Ic
k] = EPN
h
∥ψ(W; θ0, bη0,k) −ψ(W; θ0, η0)∥2 | (Wi)i∈Ic
k
i
⩽sup
η∈TN
EPN
h
∥ψ(W; θ0, η) −ψ(W; θ0, η0)∥2 | (Wi)i∈Ic
k
i
⩽sup
η∈TN
EPN
h
∥ψ(W; θ0, η) −ψ(W; θ0, η0)∥2i
= (r′
N)2$$

by the definition of r ′ N in Assumption 3.2. Hence, I 3 ,k = O P N ( r ′ N ) by Lemma 6.1 in the Appendix. To bound I 4 ,k , introduce the function

$$fk(r) := EPN [ψ(W; θ0, η0 + r(bη0,k −η0)) | (Wi)i∈Ic
k] −EPN [ψ(W; θ0, η0)],
r ∈[0, 1].$$

Then, by Taylor's expansion,

$$fk(1) = fk(0) + f ′
k(0) + f ′′
k (˜r)/2,
for some ˜r ∈(0, 1).$$

But ∥ f k (0) ∥ = 0 since

$$EPN [ψ(W; θ0, η0) | (Wi)i∈Ic
k] = EPN [ψ(W; θ0, η0)].$$

In addition, on the event E N , by the Neyman λ N near-orthogonality condition imposed in Assumption 3.1,

$$∥f ′
k(0)∥=



∂ηEPN ψ(W; θ0, η0)[bη0,k −η0]



 ⩽λN.$$

Moreover, on the event E N ,

$$∥f ′′
k (˜r)∥⩽
sup
r∈(0,1)
∥f ′′
k (r)∥⩽λ′
N$$

by the definition λ ′ N in Assumption 3.2. Hence,

$$I4,k = √n∥fk(1)∥= OPN (√n(λN + λ′
N)).$$

Combining the bounds on I 3 ,k and I 4 ,k with (A.16) and using the fact that n -1 = O ( N -1 ) gives (A.15).

Step 4. To establish (A.7), note that

$$EPN
h


 1
√
N
N
X
i=1
ψ(Wi; θ0, η0)



2i
= EPN
h
∥ψ(W; θ0, η0)∥2i
⩽c2
1$$

by Assumption 3.2. Combining this with Markov's inequality gives (A.7).

Step 5. Here we establish (A.8). Note that all eigenvalues of the matrix

$$σ2 = J−1
0 EP [ψ(W; θ0, η0)ψ(W; θ0, η0)′](J−1
0 )′$$

are bounded from below by c 0 /c 2 1 since all singular values of J 0 are bounded from above by c 1 by Assumption 3.1 and all eigenvalues of E P [ ψ ( W ; θ 0 , η 0 ) ψ ( W ; θ 0 , η 0 ) ′ ] are bounded from below by c 0 by Assumption 3.2. Hence, given that ∥ σ -1 ∥ is the largest eigenvalue of σ -1 , it follows that ∥ σ -1 ∥ = c 1 / √ c 0 . This gives (A.8) and completes the proof of the theorem. ■

## A.13. Proof of Theorem 3.1 (DML1 case)

As in the case of the DML2 version, note that (3.11) follows immediately from the assumptions, and so it suffices to show that (3.10) holds uniformly over P ∈ P N .

Fix any sequence { P N } N ⩾ 1 such that P N ∈ P N for all N ⩾ 1. Since this sequence is chosen arbitrarily, to show that (3.10) holds uniformly over P ∈ P N , it suffices to show that

$$√
Nσ−1(˜θ0 −θ0) =
1
√
N
N
X
i=1
¯ψ(Wi) + OPN (ρN) ; N(0, Id).
(A.17)$$

To do so, for all k ∈ [ K ], denote

$$bJ0,k := En,k[ψa(W; bη0,k)],
RN,1,k := bJ0,k −J0,$$

$$RN,2,k := En,k[ψ(W; θ0, bη0,k)] −1
n
X
i∈Ik
ψ(Wi; θ0, η0).$$

Since K is a fixed integer, which is independent of n , it follows by the same arguments as those in Steps 2-5 in Section A.12 that

$$max
k∈[K] ∥RN,1,k∥= OPN (N −1/2 + rN),
(A.18)$$

$$max
k∈[K] ∥RN,2,k∥= OPN (N −1/2r′
N + λN + λ′
N),
(A.19)$$

$$max
k∈[K] ∥n−1/2 P
i∈Ik ψ(Wi; θ0, η0)∥= OPN (1),
(A.20)$$

$$∥σ−1∥= OPN (1).
(A.21)$$

Since N -1 / 2 + r N ⩽ ρ N = o (1) and all singular values of J 0 are bounded below from zero by Assumption 3.1, it follows from (A.18) that for all k ∈ [ K ], with P N -probability 1 -o (1), all singular values of ̂ J 0 ,k are bounded below from zero, and so with the same P N -probability,

$$ˇθ0,k = −bJ−1
0,kEn,k[ψb(W; bη0,k)].$$

Hence, by the same arguments as those in Step 1 in Section A.12, it follows from the bounds (A.18)-(A.21) that for all k ∈ [ K ],

$$√nσ−1(ˇθ0,k −θ0) =
1
√n
X
i∈Ik
¯ψ(Wi) + OPN (ρN).$$

Therefore,

$$√
Nσ−1(˜θ0 −θ0) =
√
Nσ−1 1
K
K
X
k=1
ˇθ0,k −θ0

=
1
√
N
N
X
i=1
¯ψ(Wi) + OPN (ρN).
(A.22)$$

In turn, since ρ N = o (1), combining (A.22) with the Lindeberg-Feller CLT and the Cramer-Wold device yields (A.17) and completes the proof of the theorem. ■

$$A.14. Proof of Theorem 3.2.$$

In this proof, all bounds hold uniformly in P ∈ P N for N ⩾ 3, and we do not repeat this qualification throughout. Also, the second asserted claim follows immediately from the first one and Theorem 3.1. Hence, it suffices to prove the first asserted claim.

In the proof of Theorem 3.1 in Section A.12, we established that ∥ ̂ J 0 -J 0 ∥ = O P ( r N + N -1 / 2 ). Hence, since ∥ J -1 0 ∥ ⩽ c -1 0 by Assumption 3.1 and

$$EP [ψ(W; θ0, η0)ψ(W; θ0, η0)′]



 ⩽EP [∥ψ(W; θ0, η0)∥2] ⩽c2
1$$

by Assumption 3.2, it suffices to show that

$$1
K
K
X
k=1
En,k[ψ(W; ˜θ0, bη0,k)ψ(W; ˜θ0, bη0,k)′] −EP [ψ(W; θ0, η0)ψ(W; θ0, η0)′]



 = OP (ϱN).$$

Moreover, since both K and d θ , the dimension of ψ , are fixed integers, which are independent of N , the last bound will follow if we show that for all k ∈ [ K ] and all j, k ∈ [ d θ ],

$$Ikjl :=
En,k[ψj(W; ˜θ0, bη0,k)ψl(W; ˜θ0, bη0,k)] −EP [ψj(W; θ0, η0)ψl(W; θ0, η0)]$$

satisfies where

$$Ikjl,1 :=
En,k[ψj(W; ˜θ0, bη0,k)ψl(W; ˜θ0, bη0,k)] −En,k[ψj(W; θ0, η0)ψl(W; θ0, η0)]
,
Ikjl,2 :=
En,k[ψj(W; θ0, η0)ψl(W; θ0, η0)] −EP [ψj(W; θ0, η0)ψl(W; θ0, η0)]
.$$

We bound I kjl, 2 first. If q ⩾ 4, then

$$EP [I2
kjl,2] ⩽n−1EP
h
(ψj(W; θ0, η0)ψl(W; θ0, η0))2i
⩽n−1
EP [ψ4
j (W; θ0, η0)] × EP [ψ4
l (W; θ0, η0)]
1/2
⩽n−1EP [∥ψ(W; θ0, η0)∥4] ⩽c4
1,$$

where the second line holds by H¨ older's inequality, and the third one by Assumption 3.2. Hence, I kjl, 2 = O P ( N -1 / 2 ). If q ∈ (2 , 4), we apply the following von Bahr-Esseen inequality with p = q/ 2: if X 1 , . . . , X n are independent random variables with mean zero,

$$Ikjl = OP (ϱN).
(A.23)$$

To do so, observe that by the triangle inequality,

$$Ikjl ⩽Ikjl,1 + Ikjl,2,
(A.24)$$

then for any p ∈ [1 , 2],

Denoting

ψ hi := ψ h ( W i ; θ 0 , η 0 ) and ̂ ψ hi := ψ h ( W i ; ˜ θ 0 , ̂ η 0 ,k ) , for ( h, i ) ∈ { j, l } × I k , and applying the inequality above with a := ψ ji , b := ψ li , a + δa := ̂ ψ ji , b + δb := ̂ ψ li , r := | ̂ ψ ji -ψ ji | ∨ | ̂ ψ li -ψ li | , and c := | ψ ji | ∨ | ψ li | gives

$$Ikjl,1 =
 1
n
X
i∈Ik
bψji bψli −ψjiψli
 ⩽1
n
X
i∈Ik
| bψji bψli −ψjiψli|
⩽2
n
X
i∈Ik

| bψji −ψji| ∨| bψli −ψli|

×

|ψji| ∨|ψli| + | bψji −ψji| ∨| bψli −ψli|

⩽
 2
n
X
i∈Ik

| bψji −ψji|2 ∨| bψli −ψli|21/2
×
 2
n
X
i∈Ik

|ψji| ∨|ψli| + | bψji −ψji| ∨| bψli −ψli|
21/2
.$$

In addition, the expression in the last line above is bounded by and so

where

Moreover,

$$ 2
n
X
i∈Ik
|ψji|2 ∨|ψli|21/2
+
 2
n
X
i∈Ik
| bψji −ψji|2 ∨| bψli −ψli|21/2
,
I2
kjl,1 ≲RN ×
 1
n
X
i∈Ik
∥ψ(Wi; θ0, η0)∥2 + RN

,
RN := 1
n
X
i∈Ik
∥ψ(Wi; ˜θ0, bη0,k) −ψ(Wi; θ0, η0)∥2.
1
n
X
i∈Ik
∥ψ(Wi; θ0, η0)∥2 = OP (1),$$

$$E
"
n
X
i=1
Xi

p#
⩽

2 −1
n

n
X
i=1
E[|Xi|p];$$

see DasGupta (2008), p. 650. This gives

$$EP [Iq/2
kjl,2] ≲n−q/2+1EP
h
(ψj(W; θ0, η0)ψl(W; θ0, η0))q/2i
⩽n−q/2+1EP [∥ψ(W; θ0, η0)∥q] ≲n−q/2+1$$

by Assumption 3.2. Hence, I kjl, 2 = O P ( N 2 /q -1 ). Conclude that

$$Ikjl,2 = OP

N −[(1−2/q)∧(1/2)]
.
(A.25)$$

Next, we bound I kjl, 1 . To do so, observe that for any numbers a , b , δa , and δb such that | a | ∨ | b | ⩽ c and | δa | ∨ | δb | ⩽ r , we have

$$(a + δa)(b + δb) −ab
 ⩽2r(c + r).$$

by Markov's inequality since

$$EP
h 1
n
X
i∈Ik
∥ψ(Wi; θ0, η0)∥2i
= EP [∥ψ(W; θ0, η0∥2] ⩽c2
1$$

by Assumption 3.2. It remains to bound R N . We have

$$RN ≲1
n
X
i∈Ik



ψa(Wi; bη0,k)(˜θ0 −θ0)



2
+ 1
n
X
i∈Ik



ψ(Wi; θ0, bη0,k)−ψ(Wi; θ0, η0)



2
. (A.26)$$

The first term on the right-hand side of (A.26) is bounded from above by

$$ 1
n
X
i∈Ik
∥ψa(Wi; bη0,k)∥2
× ∥˜θ0 −θ0∥2 = OP (1) × OP (N −1) = OP (N −1),$$

and the conditional expectation of the second term given ( W i ) i ∈ I c k on the event that ̂ η 0 ,k ∈ T N is equal to

$$EP
h
∥ψ(W; θ0, bη0,k) −ψ(W; θ0, η0)∥2 | (Wi)i∈Ic
k
i
⩽sup
η∈TN
EP
h
∥ψ(W; θ0, η) −ψ(W; θ0, η0)∥2 | (Wi)i∈Ic
k
i
= (r′
N)2.$$

Since the event that ̂ η 0 ,k ∈ T N holds with probability 1 -∆ N = 1 -o (1), it follows that R N = O P ( N -1 +( r ′ N ) 2 ) , and so

$$Ikjl,1 = OP

N −1/2 + r′
N

.
(A.27)$$

Combining the bounds (A.25) and (A.27) with (A.24) gives (A.23) and completes the proof of the theorem. ■

## Proof of Theorem 3.3.

We only consider the case of the DML1 estimator and note that the DML2 estimator can be treated similarly.

The main part of the proof is the same as that in the linear case (Theorem 3.1, DML1 case, presented in Section A.13), once we have the following lemma that establishes approximate linearity of the subsample DML estimators ˇ θ 0 ,k .

Lemma 6.3. ( Linearization for Subsample DML in Nonlinear Problems ) Under the conditions of Theorem 3.3, for any k = 1 , . . . , K , the estimator ˇ θ 0 = ˇ θ 0 ,k defined by equation (3.2) obeys

$$√nσ−1
0 (ˇθ0 −θ0) =
1
√n
X
i∈I
¯ψ(Wi) + OP (ρ′
n)
(A.28)$$

uniformly over P ∈ P N , where ρ ′ n = n -1 / 2 + r N + r ′ N + n 1 / 2 λ N + n 1 / 2 λ ′ N ≲ δ N and where ¯ ψ ( · ) := -σ -1 0 J -1 0 ψ ( · , θ 0 , η 0 ) .

Proof of Lemma 6.3. Fix any k = 1 , . . . , K and any sequence { P N } N ⩾ 1 such that P N ∈ P N for all N ⩾ 1. To prove the asserted claim, it suffices to show that the estimator ˇ θ 0 = ˇ θ 0 ,k satisfies (A.28) with P replaced by P N . To do so, we split the proof

into four steps. In the proof, we will use E n , G n , I , and ̂ η 0 instead of E n,k , G n,k , I k , and ̂ η 0 ,k , respectively.

Step 1. (Preliminary Rate Result). We claim that with P N -probability 1 -o (1),

$$∥ˇθ0 −θ0∥⩽τN.
(A.29)$$

To show this claim, note that the definition of ˇ θ 0 implies that

$$En[ψ(W; ˇθ0, bη0)]



 ⩽



En[ψ(W; θ0, bη0)]



 + ϵN,$$

which in turn implies via the triangle inequality that, with P N -probability 1 -o (1),

$$EPN [ψ(W; θ, η0)]|θ=ˇθ0



 ⩽ϵN + 2I1 + 2I2,
(A.30)$$

where

$$I1 :=
sup
θ∈Θ,η∈TN



EPN [ψ(W; θ, η)] −EPN [ψ(W; θ, η0)]



,
I2 :=
max
η∈{η0,bη0} sup
θ∈Θ



En[ψ(W; θ, η)] −EPN [ψ(W; θ, η)]



.$$

Here ϵ N = o ( τ N ) because ϵ N = o ( δ N N -1 / 2 ), δ N = o (1), and τ N ⩾ c 0 N -1 / 2 log n . Also, I 1 = r N ⩽ δ N τ N = o ( τ N ) by Assumption 3.4(c). Moreover, applying Lemma 6.2 to the function class F 1 ,η for η = η 0 and η = ̂ η 0 defined in Assumption 3.4, conditional on ( W i ) i ∈ I c and I c , so that ̂ η 0 is fixed after conditioning, shows that with P N -probability 1 -o (1),

$$I2 ≲N −1/2(1 + N −1/2+1/q log n) ≲N −1/2 = o(τN).$$

Hence, it follows from (A.30) and Assumption 3.3 that with P N -probability 1 -o (1),

$$∥J0(ˇθ0 −θ0)∥∧c0 ⩽



 EPN [ψ(W; θ, η0)]|θ=ˇθ0



 = o(τN).
(A.31)$$

Combining this bound with the fact that the singular values of J 0 are bounded away from zero, which holds by Assumption 3.3, gives the claim of this step.

Step 2. (Linearization) Here we prove the claim of the lemma. First, by definition of ˇ θ 0 , we have

$$√n



En[ψ(W; ˇθ0, bη0)]



 ⩽inf
θ∈Θ
√n



En[ψ(W; θ, bη0)]



 + ϵN
√n.
(A.32)$$

Also, it will be shown in Step 4 that

$$I3 := inf
θ∈Θ
√n∥En[ψ(W; θ, bη0)]∥
= OPN (n−1/2+1/q log n + r′
N log1/2(1/r′
N) + λN
√n + λ′
N
√n).
(A.33)$$

Moreover, for any θ ∈ Θ and η ∈ T N , we have

$$√nEn[ψ(W; θ, η)] = √nEn[ψ(W; θ0, η0)] + Gn[ψ(W; θ, η) −ψ(W; θ0, η0)]
+ √n

EPN [ψ(W; θ, η)

,
(A.34)$$

where we are using that E P N [ ψ ( W ; θ 0 , η 0 )]= 0. Finally, by Taylor's expansion of the

function r ↦→ E P N [ ψ ( W ; θ 0 + r ( θ -θ 0 ) , η 0 + r ( η -η 0 ))], which vanishes at r = 0,

$$EPN [ψ(W; θ, η)] = J0(θ −θ0) + ∂ηEPN ψ(W; θ0, η0)[η −η0]
+
Z 1
0
2−1∂2
rEPN [W; θ0 + r(θ −θ0), η0 + r(η −η0)]dr.
(A.35)$$

Therefore, since ∥ ˇ θ 0 -θ 0 ∥ ⩽ τ N and η ∈ T N with P N -probability 1 -o (1), and since by Neyman λ N -near orthogonality,

$$∥∂ηEPN [ψ(W; θ0, η0)][bη0 −η0]∥⩽λN,$$

applying (A.34) with θ = ˇ θ 0 and η = ̂ η 0 , we have with P N -probability 1 -o (1),

$$√n



En[ψ(W; θ0, η0)] + J0(ˇθ0 −θ0)



 ⩽λN
√n + ϵN
√n + I3 + I4 + I5,$$

where by Assumption 3.4,

$$I4 := √n
sup
∥θ−θ0∥⩽τN ,η∈TN



Z 1
0
2−1∂2
rEPN [W; θ0 + r(θ −θ0), η0 + r(η −η0)]dr



 ⩽λ′
N
√n,$$

and by Step 3 below, with P N -probability 1 -o (1),

$$I5 :=
sup
∥θ−θ0∥⩽τN



Gn

ψ(W; θ, bη0) −ψ(W; θ0, η0)



(A.36)$$

$$⩽r′
N log1/2(1/r′
N) + n−1/2+1/q log n.
(A.37)$$

Therefore, since all singular values of J 0 are bounded below from zero by Assumption 3.3(d), it follows that

$$J−1
0
√nEn[ψ(W; θ0, η0)] + √n(ˇθ0 −θ0)



= OPN (n−1/2+1/q log n + r′
N log1/2(1/r′
N) + (ϵN + λN + λ′
N)√n.$$

The asserted claim now follows by multiplying both parts of the display by Σ -1 / 2 0 (under the norm on the left-hand side) and noting that singular values of Σ 0 are bounded below from zero by Assumptions 3.3 and 3.4.

Step 3. Here we derive a bound on I 5 in (A.37). We have

$$I5 ≲sup
f∈F2
|Gn(f)|,
F2 =

ψj(·, θ, bη0) −ψj(·, θ0, η0): j = 1, ..., dθ, ∥θ −θ0∥⩽τn

.$$

To bound sup f ∈F 2 | G n ( f ) | , we apply Lemma 6.2 conditional on ( W i ) i ∈ I c and I c so that ̂ η 0 can be treated as fixed. Observe that with P N -probability 1 -o (1), sup f ∈F 2 ∥ f ∥ P N , 2 ≲ r ′ N where we used Assumption 3.4. Thus, an application of Lemma 6.2 to the empirical process { G n ( f ) , f ∈ F 2 } with an envelope F 2 = F 1 , ̂ η 0 + F 1 ,η 0 and σ = Cr ′ N for sufficiently large constant C conditional on ( W i ) i ∈ I c and I c yields that with P N -probability 1 -o (1),

$$sup
f∈F2
|Gn(f)| ≲r′
N log1/2(1/r′
N) + n−1/2+1/q log n.
(A.38)$$

This follows since ∥ F 2 ∥ P,q = ∥ F 1 , ̂ η 0 + F 1 ,η 0 ∥ P,q ⩽ 2 C 1 by Assumption 3.4(b) and the triangle inequality, and

$$log sup
Q
N(ϵ∥F2∥Q,2, F2, ∥· ∥Q,2) ⩽2v log(2a/ϵ),
for all 0 < ϵ ⩽1,$$

because F 2 ⊂ F 1 , ̂ η 0 -F 1 ,η 0 for F 1 ,η defined in Assumption 3.4(b), and

$$log sup
Q
N(ϵ∥F1,bη0 + F1,η0∥Q,2, F1,bη0 −F1,η0, ∥· ∥Q,2)
⩽log sup
Q
N((ϵ/2)∥F1,bη0∥Q,2, F1,bη0, ∥· ∥Q,2) + log sup
Q
N((ϵ/2)∥F1,η0∥Q,2, F1,η0, ∥· ∥Q,2$$

$$)$$

by the proof of Theorem 3 in Andrews (1994b). The claim of this step follows.

Step 4. Here we derive a bound on I 3 in (A.33). Let ¯ θ 0 = θ 0 -J -1 0 E n [ ψ ( W ; θ 0 , η 0 )]. Then ∥ ¯ θ 0 -θ 0 ∥ = O P N (1 / √ n ) = o P N ( τ n ) since E P N [ ∥ √ n E n [ ψ ( W ; θ 0 , η 0 )] ∥ ] is bounded and the singular values of J 0 are bounded below from zero by Assumption 3.3(d). Therefore, ¯ θ 0 ∈ Θ with P N -probability 1 -o (1) by Assumption 3.3(a). Hence, with the same probability,

$$inf
θ∈Θ
√n



En[ψ(W; θ, bη0)]



 ⩽√n



En[ψ(W; ¯θ0, bη0)]



,$$

and so it suffices to show that with P N -probability 1 -o (1),

$$√n



En[ψ(W; ¯θ0, bη0)]



 = O(n−1/2+1/q log n + r′
N log1/2(1/r′
N) + λN
√n + λ′
N
√n).$$

To prove it, substitute θ = ¯ θ 0 and η = ̂ η 0 into (A.34) and use Taylor's expansion in (A.35). This shows that with P N -probability 1 -o (1),

$$√n



En[ψ(W; ¯θ0, bη0)]



 ⩽√n



En[ψ(W; θ0, η0)] + J0(¯θ0 −θ0)



 + λN
√n + I4 + I5
= λN
√n + I4 + I5,$$

Combining this with the bounds on I 4 and I 5 derived above gives the claim of this step and completes the proof of the theorem. ■

$$Proof of Theorems 4.1 and 4.2$$

Since Theorem 4.1 is a special case of Theorem 4.2 (with Z = D ), it suffices to prove the latter. Also, we only consider the DML estimators based on the score (4.7) and note that the estimators based on the score (4.8) can be treated similarly.

Observe that the score ψ in (4.7) is linear in θ :

$$ψ(W; θ, η) = (Y −Dθ −g(X))(Z −m(X)) = ψa(W; η)θ + ψb(W; η),
ψa(W; η) = D(m(X) −Z),
ψb(W; η) = (Y −g(X))(Z −m(X)).$$

Therefore, all asserted claims of Theorem 4.2 follow from Theorems 3.1 and 3.2 and Corollary 3.1 as long as we can verify Assumptions 3.1 and 3.2, which we do here. We do so with T N being the set of all η = ( g, m ) consisting of P -square-integrable functions g and m such that

$$∥η −η0∥P,∞⩽C,
∥η −η0∥P,2 ⩽δN,
∥m −m0∥P,2 × ∥g −g0∥P,2 ⩽δNN −1/2$$

$$.$$

Also, we replace the constant q and the sequence ( δ N ) N ⩾ 1 in Assumptions 3.1 and 3.2 by q/ 2 and ( δ ′ N ) N ⩾ 1 with δ ′ N = ( C +2 √ C +2)( δ N ∨ N -[(1 -4 /q ) ∧ (1 / 2)] ) for all N (recall that we assume that q &gt; 4, and the analysis in Section 3 only requires that q &gt; 2; also, δ ′ N satisfies δ ′ N ⩾ N -[(1 -4 /q ) ∧ (1 / 2)] , which is required in Theorems 3.1 and 3.2). We proceed

in five steps. All bounds in the proof hold uniformly over P ∈ P but we omit this qualifier for brevity).

Step 1. We first verify Neyman orthogonality. We have that E P ψ ( W ; θ 0 , η 0 ) = 0 by definition of θ 0 of η 0 . Also, for any η = ( g, m ) ∈ T N , the Gateaux derivative in the direction η -η 0 = ( g -g 0 , m -m 0 ) is given by

$$∂ηEP ψ(W; θ0, η0)[η −η0] = EP
h
(g(X) −g0(X))(m0(X) −Z)
i
+ EP
h
(m0(X) −m(X))(Y −Dθ0 −g0(X))
i
= 0,$$

by the law of iterated expectations, since V = Z -m 0 ( X ) and U = ( Y -Dθ 0 -g 0 ( X )) obey E P [ V | X ] = 0 and E P [ U | Z, X ] = 0. This gives Assumption 3.1(d) with λ N = 0.

Step 2. Note that

$$|J0| = |EP [ψa(W; η0)]| = |EP [D(m0(X) −Z)]| = |EP [DV ]| ⩾c > 0$$

by Assumption 4.2(c). In addition,

$$|EP [ψa(W; η0)]| = |EP [D(m0(X) −Z)]| ⩽∥D∥P,2∥m0(X)∥P,2 + ∥D∥P,2∥Z∥P,2
⩽2∥D∥P,2∥Z∥P,2 ⩽2∥D∥P,q∥Z∥P,q ⩽2C2$$

by the triangle inequality, H¨ older's inequality, Jensen's inequality, and Assumption 4.2(b). This gives Assumption 3.1(e). Hence, given that Assumptions 3.1(i,ii,iii) hold trivially, Steps 1 and 2 together show that all conditions of Assumption 3.1 hold.

Step 3. Note that Assumption 3.2(a) holds by construction of the set T N and Assumption 4.2(e). Also, note that ψ ( W ; θ 0 , η 0 ) = UV , and so

$$EP [ψ(W; θ0, η0)ψ(W; θ0, η0)′] = EP [U 2V 2] ⩾c4 > 0,$$

by Assumption 4.2(c), which gives Assumption 3.2(d).

Step 4. Here we verify Assumption 3.2(b). For any η = ( g, m ) ∈ T N , we have

$$(EP [∥ψa(W; η)∥q/2])2/q = ∥ψa(W; η)∥P,q/2 = ∥D(m(X) −Z)∥P,q/2
⩽∥D(m(X) −m0(X))∥P,q/2 + ∥Dm0(X)∥P,q/2 + ∥DZ∥P,q/2
⩽∥D∥P,q∥m(X) −m0(X)∥P,q + ∥D∥P,q∥m0(X)∥P,q + ∥D∥P,q∥Z∥P,q
⩽C∥D∥P,q + 2∥D∥P,q∥Z∥P,q ⩽3C2$$

by Assumption 4.2(b), which gives the bound on m ′ N in Assumption 3.2(b). Also, since

$$|EP [(D −r0(X))(Z −m0(X))]| = |EP [DV ]| ⩾c$$

by Assumption 4.2(c), it follows that θ 0 satisfies

$$|θ0| = |EP [(Y −ℓ0(X))(Z −m0(X))]|
|EP [(D −r0(X))(Z −m0(X))]|
⩽c−1
∥Y ∥P,2 + ∥ℓ0(X)∥P,2

∥Z∥P,2 + ∥m0(X)∥P,2

⩽4c−1∥Y ∥P,2∥Z∥P,2 ⩽4C2/c.$$

Hence,

$$(EP [∥ψ(W; θ0, η)∥q/2])2/q = ∥ψ(W; θ0, η)∥P,q/2
= ∥(Y −Dθ0 −g(X))(Z −m(X))∥P,q/2
⩽∥U(Z −m(X))∥P,q/2 + ∥(g(X) −g0(X))(Z −m(X))∥P,q/2
⩽∥U∥P,q∥Z −m(X)∥P,q + ∥g(X) −g0(X)∥P,q∥Z −m(X)∥P,q
⩽(∥U∥P,q + C)∥Z −m(X)∥P,q
⩽(∥Y −Dθ0∥P,q + ∥g0(X)∥P,q + C)
× (∥Z∥P,q + ∥m0(X)∥P,q + ∥m(X) −m0(X)∥P,q)
⩽(2∥Y −Dθ0∥P,q + C)(2∥Z∥P,q + C)
⩽(2∥Y ∥P,q + 2∥D∥P,q|θ0| + C)(2∥Z∥P,q + C)
⩽3C(3C + 8C3/c),$$

where we used the fact that since g 0 ( X ) = E P [ Y -Dθ 0 | X ], ∥ g 0 ( X ) ∥ P,q ⩽ ∥ Y -Dθ 0 ∥ P,q by Jensen's inequality. This gives the bound on m N in Assumption 3.2(b). Hence, Assumption 3.2(b) holds.

Step 5. Finally, we verify Assumption 3.2(c). For any η = ( g, m ) ∈ T N , we have

$$∥EP [ψa(W; η)] −EP [ψa(W; η0)]∥= |EP [ψa(W; η) −ψa(W; η0)]|
= |EP [D(m(X) −m0(X))]|
⩽∥D∥P,2∥m(X) −m0(X)∥P,2 ⩽CδN ⩽δ′
N,$$

which gives the bound on r N in Assumption 3.2(c). Further,

$$(EP [∥ψ(W; θ0, η) −ψ(W; θ0, η0)∥2])1/2
= ∥ψ(W; θ0, η) −ψ(W; θ0, η0)∥P,2
= ∥(U + g0(X) −g(X))(Z −m(X)) −U(Z −m0(X))∥P,2
⩽∥U(m(X) −m0(X))∥P,2 + ∥(g(X) −g0(X))(Z −m(X))∥P,2
⩽
√
C∥m(X) −m0(X)∥P,2 + ∥V (g(X) −g0(X))∥P,2
+ ∥(g(X) −g0(X))(m(X) −m0(X))∥P,2
⩽
√
C∥m(X) −m0(X)∥P,2 +
√
C∥g(X) −g0(X)∥P,2 + C∥m(X) −m0(X)∥P,2
⩽(2
√
C + C)δN ⩽δ′
N,$$

which gives the bound on r ′ N in Assumption 3.2(c). Finally, let

$$f(r) := EP [ψ(W; θ0, η0 + r(η −η0)],
r ∈(0, 1).$$

Then for any r ∈ (0 , 1),

$$f(r) = EP [(U −r(g(X) −g0(X)))(V −r(m(X) −m0(X)))],$$

and so

$$∂f(r) = −EP [(g(X) −g0(X))(V −r(m(X) −m0(X)))]
−EP [(U −r(g(X) −g0(X)))(m(X) −m0(X))],
∂2f(r) = 2EP [(g(X) −g0(X))(m(X) −m0(X))].$$

Hence,

$$|∂2f(r)| ⩽2∥g(X) −g0(X)∥P,2 × ∥m(X) −m0(X)∥P,2 ⩽2δNN −1/2 ⩽δ′
NN −1/2,$$

which gives the bound on λ ′ N in Assumption 3.2(c). Thus, all conditions of Assumptions 3.1 are verified. This completes the proof. ■

$$Proof of Theorems 5.1 and 5.2$$

The proof of Theorem 5.2 is similar to that of Theorem 5.1 and therefore omitted. In turn, regarding Theorem 5.1, we show the proof for the case of ATE and note that the proof for the case of ATTE is similar.

Observe that the score ψ in (5.3) is linear in θ :

$$ψ(W; θ, η) = ψa(W; η)θ + ψb(W; η),
ψa(W; η) = −1,
ψb(W; η) = (g(1, X) −g(0, X)) + D(Y −g(1, X))
m(X)
−(1 −D)(Y −g(0, X))
1 −m(X)$$

$$.$$

Therefore, all asserted claims of Theorem 5.1 follow from Theorems 3.1 and 3.2 and Corollary 3.1 as long as we can verify Assumptions 3.1 and 3.2, which we do here. We do so with T N being the set of all η = ( g, m ) consisting of P -square-integrable functions g and m such that

$$∥η −η0∥P,q ⩽C,
∥η −η0∥P,2 ⩽δN,
∥m −1/2∥P,∞⩽1/2 −
∥m −m0∥P,2 × ∥g −g0∥P,2 ⩽δNN −1/2.$$

$$ε,$$

Also, we replace the sequence ( δ N ) N ⩾ 1 in Assumptions 3.1 and 3.2 by ( δ ′ N ) N ⩾ 1 with δ ′ N = C ε ( δ N ∨ N -[(1 -4 /q ) ∧ (1 / 2)] ) for all N , where C ε is a sufficiently large constant that depends only on ε and C (note that δ ′ N satisfies δ ′ N ⩾ N -[(1 -4 /q ) ∧ (1 / 2)] , which is required in Theorems 3.1 and 3.2). We proceed in five steps. All bounds in the proof hold uniformly over P ∈ P but we omit this qualifier for brevity.

- Step 1. We first verify Neyman orthogonality. We have that E ψ ( W ; θ 0 , η 0 ) = 0 by definition of θ 0 and η 0 . Also, for any η = ( g, m ) ∈ T N , the Gateaux derivative in the direction η -η 0 = ( g -g 0 , m -m 0 ) is given by

$$∂ηEP ψ(W; θ0, η0)[η −η0] = EP
h
g(1, X) −g0(1, X)
i
−EP
h
g(0, X) −g0(0, X)
i
−EP
hD(g(1, X) −g0(1, X))
m0(X)
i
+ EP
h(1 −D)(g(0, X) −g0(0, X))
1 −m0(X)
i
−EP
hD(Y −g0(1, X))(m(X) −m0(X))
m2
0(X)
i
−EP
h(1 −D)(Y −g0(0, X))(m(X) −m0(X))
(1 −m0(X))2
i
,$$

which is 0 by the law of iterated expectations, since

$$EP [D | X] = m0(X),
EP [1 −D | X] = 1 −m0(X)$$

$$EP [D(Y −g0(1, X)) | X] = 0,
EP [(1 −D)(Y −g0(0, X)) | X] = 0$$

This gives Assumption 3.1(d) with λ N = 0.

Step 2. Note that J 0 = -1, and so Assumption 3.1(e) holds trivially. Hence, given that Assumptions 3.1(i,ii,iii) hold trivially as well, Steps 1 and 2 together show that all conditions of Assumption 3.1 hold.

$$,
X] = 0.$$

Step 3. Note that Assumption 3.2(a) holds by construction of the set T N and Assumption 5.1(f). Also,

$$EP
h
ψ2(W; θ0, η0)
i
= EP
h
EP [ψ2(W; θ0, η0) | X]
i
= EP
h
EP [(g0(1, X) −g0(0, X) −θ0)2 | X]
+ EP
hD(Y −g0(1, X))
m0(X)
−(1 −D)(Y −g0(0, X))
1 −m0(X)
2
| X
ii
⩾EP
hD(Y −g0(1, X))
m0(X)
−(1 −D)(Y −g0(0, X))
1 −m0(X)
2i
= EP
hD2(Y −g0(1, X))2
m0(X)2
+ (1 −D)2(Y −g0(0, X))2
(1 −m0(X))2
i
⩾
1
(1 −ε)2 EP
h
D2(Y −g0(1, X))2 + (1 −D)2(Y −g0(0, X))2i
=
1
(1 −ε)2 EP [DU 2 + (1 −D)U 2] =
1
(1 −ε)2 EP [U 2] ⩾
c2
(1 −ε)2 .$$

This gives Assumption 3.2(d).

Step 4. Here we verify Assumption 3.2(b). We have

$$∥g0(D, X)∥P,q = (EP [|g0(D, X)|q])1/q
⩾

EP
h
|g0(1, X)|qPP (D = 1 | X) + |g0(0, X)|qPP (D = 0 | X)
i1/q
⩾ε1/q
EP [|g0(1, X)|q] + EP [|g0(0, X)|q]
1/q
⩾ε1/q
EP [|g0(1, X)|q] ∨EP [|g0(0, X)|q]
1/q
⩾ε1/q
∥g0(1, X)∥P,q ∨∥g0(0, X)∥P,q

,$$

where in the third line, we used the facts that P P ( D = 1 | X ) = m 0 ( X ) ⩾ ε and P P ( D = 0 | X ) = 1 -m 0 ( X ) ⩾ ε . Hence, given that ∥ g 0 ( D,X ) ∥ P,q ⩽ ∥ Y ∥ P,q ⩽ C by Jensen's inequality and Assumption 5.1(b), it follows that

$$∥g0(1, X)∥P,q ⩽C/ε1/q
and
∥g0(0, X)∥P,q ⩽C/ε1/q.$$

Similarly, for any η ∈ ( g, m ) ∈ T N ,

$$∥g(1, X) −g0(1, X)∥P,q ⩽C/ε1/q
and
∥g(0, X) −g0(0, X)∥P,q ⩽C/ε1/q$$

since ∥ g ( D,X ) -g 0 ( D,X ) ∥ P,q ⩽ C . In addition,

$$|θ0| = |EP [g0(1, X) −g0(0, X)]| ⩽∥g0(1, X)∥P,2 + ∥g0(0, X)∥P,2 ⩽2C/ε1/q.$$

Therefore, for any η = ( g, m ) ∈ T N , we have

$$(EP [|ψ(W; θ0, η)|q])1/q = ∥ψ(W; θ0, η)∥P,q
⩽(1 + ε−1)

∥g(1, X)∥P,q + ∥g(0, X)∥P,q

+ 2∥Y ∥P,q/ε + |θ0|
⩽(1 + ε−1)

∥g(1, X) −g0(1, X)∥P,q + ∥g(0, X) −g0(0, X)∥P,q

+ (1 + ε−1)

∥g0(1, X)∥P,q + ∥g0(0, X)∥P,q

+ 2C/ε + 2C/ε1/q
⩽4C(1 + ε−1)/ε1/q + 2C/ε + 2C/ε1/q.$$

This gives the bound on m N in Assumption 3.2(b). Also, we have

$$(EP [|ψa(W; η)|q])1/q = 1.$$

This gives the bound on m ′ N in Assumption 3.2(b). Hence, Assumption 3.2(b) holds.

Step 5. Finally, we verify Assumption 3.2(c). For any η = ( g, m ) ∈ T N , we have

$$∥EP [ψa(W; η) −ψa(W; η0)]∥= |1 −1| = 0 ⩽δ′
N,$$

which gives the bound on r N in Assumption 3.2(c). Further, by the triangle inequality,

$$(EP [∥ψ(W; θ0, η) −ψ(W; θ0, η0)∥2])1/2 = ∥ψ(W; θ0, η) −ψ(W; θ0; η0)∥P,2
⩽I1 + I2 + I3,$$

where

$$I1 :=



g(1, X) −g0(1, X)



P,2 +



g(0, X) −g0(0, X)



P,2,
I2 :=



D(Y −g(1, X))
m(X)
−D(Y −g0(1, X))
m0(X)



P,2,
I3 :=



(1 −D)(Y −g(0, X))
1 −m(X)
−(1 −D)(Y −g0(0, X))
1 −m0(X)



P,2.$$

To bound I 1 , note that by the same argument as that used in Step 4,

∥ g (1 , X ) -g 0 (1 , X ) ∥ P, 2 ⩽ δ N /ε 1 / 2 and ∥ g (0 , X ) -g 0 (0 , X ) ∥ P, 2 ⩽ δ N /ε 1 / 2 , and so I 1 ⩽ 2 δ N /ε 1 / 2 . To bound I 2 , we have

$$I2 ⩽ε−2


Dm0(X)(Y −g(1, X)) −Dm(X)(Y −g0(1, X))



P,2
⩽ε−2


m0(X)(g0(1, X) + U −g(1, X)) −m(X)U



P,2
⩽ε−2


m0(X)(g(1, X) −g0(1, X))



P,2 +



(m(X) −m0(X))U



P,2

⩽ε−2
∥g(1, X) −g0(1, X)∥P,2 +
√
C∥m(X) −m0(X)∥P,2

⩽ε−2(ε−1/2 +
√
C)δN,$$

where the first inequality follows from the bounds ε ⩽ m 0 ( X ) ⩽ 1 -ε and ε ⩽ m ( X ) ⩽ 1 -ε , the second from the facts that D ∈ { 0 , 1 } and for D = 1, Y = g 0 (1 , X ) + U , the third from the triangle inequality, the fourth from the facts that m 0 ( X ) ⩽ 1 and E P [ U 2 | X ] ⩽ C , and the fifth from (A.39). Similarly, I 3 ⩽ ε -2 ( ε -1 / 2 + √ C ) δ N . Combining these inequalities shows that

$$(EP [∥ψ(W; θ0, η) −ψ(W; θ0, η0)∥2])1/2 ⩽2(ε−1/2 + ε−5/2 +
√
Cε−2)δN ⩽δ′
N,$$

$$(A.39)$$

as long as C ε in the definition of δ ′ N satisfies C ε ⩾ 2( ε -1 / 2 + ε -5 / 2 + √ Cε -2 ). This gives the bound on r ′ N in Assumption 3.2(c).

Finally, let

$$f(r) := EP [ψ(W; θ0, η0 + r(η −η0))],
r ∈(0, 1).$$

Then for any r ∈ (0 , 1),

$$∂2f(r) = EP
hD(g(1, X) −g0(1, X))(m(X) −m0(X))
(m0(X) + r(m(X) −m0(X)))2
i
+ EP
h(1 −D)(g(0, X) −g0(0, X))(m(X) −m0(X))
(1 −m0(X) −r(m(X) −m0(X)))2
i
+ EP
h(g(1, X) −g0(1, X))(m(X) −m0(X))
(m0(X) + r(m(X) −m0(X)))2
i
+ 2EP
hD(Y −g0(1, X) −r(g(1, X) −g0(1, X)))(m(X) −m0(X))2
(m0(X) + r(m(X) −m0(X)))3
i
+ EP
h(g(0, X) −g0(0, X))(m(X) −m0(X))
(1 −m0(X) −r(m(X) −m0(X)))2
i
−2EP
h(1 −D)(Y −g0(0, X) −r(g(0, X) −g0(0, X)))(m(X) −m0(X))2
(1 −m0(X) −r(m(X) −m0(X)))3
i
,$$

and so, given that

$$D(Y −g0(1, X)) = DU,
(1 −D)(Y −g0(0, X)) = (1 −D)U,$$

$$EP [U | D, X] = 0,
|m(X) −m0(X)| ⩽2,$$

it follows that for some constant C ′ ε that depends only on ε and C ,

$$|∂2f(r)| ⩽C′
ε∥m −m0∥P,2 × ∥g −g0∥P,2 ⩽δ′
NN −1/2,$$

as long as the constant C ε in the definition of δ ′ N satisfies C ε ⩾ C ′ ε . This gives the bound on λ ′ N in Assumption 3.2(c). Thus, all conditions of Assumptions 3.1 are verified. This completes the proof. ■
