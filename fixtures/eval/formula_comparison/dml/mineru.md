
# Double/Debiased Machine Learning for Treatment and Structural Parameters

Victor Chernozhukov $\dag$ , Denis Chetverikov $^ { \ddagger }$ , Mert Demirer† Esther Duflo $\dag$ , Christian Hansen $^ \mathrm { \ S }$ , Whitney Newey $^ \dagger$ , James Robins⋆

$^ \dag$ Massachusetts Institute of Technology, 50 Memorial Drive, Cambridge, MA, 02139, USA E-mail: vchern@mit. edu, mdemirer@mit.edu, duflo@mit. edu, wnewey@mit. edu $^ { \dag }$ University of California Los Angeles, 315 Portola Plaza, Los Angeles, CA 90095 E-mail: chetverikov@econ. ucla. edu §University of Chicago, 5807 S. Woodlawn Ave., Chicago, IL 60637 E-mail: chansen1@chicagobooth. edu $\star$ Harvard University, 677 Huntington Avenue Boston, Massachusetts 02115 E-mail: robins@hsph. harvard. edu Received: June 2014

Summary We revisit the classic semiparametric problem of inference on a low dimensional parameter $\theta _ { 0 }$ in the presence of high-dimensional nuisance parameters $\eta _ { 0 }$ . We depart from the classical setting by allowing for $\eta _ { 0 }$ to be so high-dimensional that the traditional assumptions, such as Donsker properties, that limit complexity of the parameter space for this object break down. To estimate $\eta _ { 0 }$ , we consider the use of statistical or machine learning (ML) methods which are particularly well-suited to estimation in modern, very high-dimensional cases. ML methods perform well by employing regularization to reduce variance and trading off regularization bias with overfitting in practice. However, both regularization bias and overfitting in estimating $\eta _ { 0 }$ cause a heavy bias in estimators of $\theta _ { 0 }$ that are obtained by naively plugging ML estimators of $\eta _ { 0 }$ into estimating equations for $\theta _ { 0 }$ . This bias results in the naive estimator failing to be $N ^ { - 1 / 2 }$ consistent, where $N$ is the sample size. We show that the impact of regularization bias and overfitting on estimation of the parameter of interest $\theta _ { 0 }$ can be removed by using two simple, yet critical, ingredients: (1) using Neyman-orthogonal moments /scores that have reduced sensitivity with respect to nuisance parameters to estimate $\theta _ { 0 }$ , and (2) making use of cross-fitting which provides an efficient form of data-splitting. We call the resulting set of methods double or debiased ML (DML). We verify that DML delivers point estimators that concentrate in a $N ^ { - 1 / 2 }$ -neighborhood of the true parameter values and are approximately unbiased and normally distributed, which allows construction of valid confidence statements. The generic statistical theory of DML is elementary and simultaneously relies on only weak theoretical requirements which will admit the use of a broad array of modern ML methods for estimating the nuisance parameters such as random forests, lasso, ridge, deep neural nets, boosted trees, and various hybrids and ensembles of these methods. We illustrate the general theory by applying it to provide theoretical properties of DML applied to learn the main regression parameter in a partially linear regression model, DML applied to learn the coefficient on an endogenous variable in a partially linear instrumental variables model, DML applied to learn the average treatment effect and the average treatment effect on the treated under unconfoundedness, and DML applied to learn the local average treatment effect in an instrumental variables setting. In addition to these theoretical applications, we also illustrate the use of DML in three empirical examples.

# 1. INTRODUCTION AND MOTIVATION

# 1.1. Motivation

We develop a series of simple results for obtaining root- $N$ consistent estimation, where $N$ is the sample size, and valid inferential statements about a low-dimensional parameter of interest, $\theta _ { 0 }$ , in the presence of a high-dimensional or “highly complex” nuisance parameter, $\eta _ { 0 }$ . The parameter of interest will typically be a causal parameter or treatment effect parameter, and we consider settings in which the nuisance parameter will be estimated using machine learning (ML) methods such as random forests, lasso or post-lasso, neural nets, boosted regression trees, and various hybrids and ensembles of these methods. These ML methods are able to handle many covariates and provide natural estimators of nuisance parameters when these parameters are highly complex. Here, highly complex formally means that the entropy of the parameter space for the nuisance parameter is increasing with the sample size in a way that moves us outside of the traditional framework considered in the classical semi-parametric literature where the complexity of the nuisance parameter space is taken to be sufficiently small. Offering a general and simple procedure for estimating and doing inference on $\theta _ { 0 }$ that is formally valid in these highly complex settings is the main contribution of this paper.

Example 1.1. (Partially Linear Regression) As a lead example, consider the following partially linear regression (PLR) model as in Robinson (1988):

$$
\begin{array} { c c } { { Y = D \theta _ { 0 } + g _ { 0 } ( X ) + U , } } & { { \operatorname { E } [ U \mid X , D ] = 0 , } } \\ { { D = m _ { 0 } ( X ) + V , } } & { { \operatorname { E } [ V \mid X ] = 0 , } } \end{array}
$$

where $Y$ is the outcome variable, $D$ is the policy/treatment variable of interest, vector

$$
\boldsymbol { X } = ( X _ { 1 } , . . . , X _ { p } )
$$

consists of other controls, and $U$ and $V$ are disturbances. The first equation is the main equation, and $\theta _ { 0 }$ is the main regression coefficient that we would like to infer. If $D$ is exogenous conditional on controls $X$ , $\theta _ { 0 }$ has the interpretation of the treatment effect (TE) parameter or “lift” parameter in business applications. The second equation keeps track of confounding, namely the dependence of the treatment variable on controls. This equation is not of interest per se but is important for characterizing and removing regularization bias. The confounding factors $X$ affect the policy variable $D$ via the function $m _ { 0 } ( X )$ and the outcome variable via the function $g _ { 0 } ( X )$ . In many applications, the dimension $p$ of vector $X$ is large relative to $N$ . To capture the feature that $p$ is not vanishingly small relative to the sample size, modern analyses then model $p$ as increasing with the sample size, which causes traditional assumptions that limit the complexity of the parameter space for the nuisance parameters $\eta _ { 0 } = ( m _ { 0 } , g _ { 0 } )$ to fail.

Regularization Bias. A naive approach to estimation of $\theta _ { 0 }$ using ML methods would be, for example, to construct a sophisticated ML estimator $D \widehat { \theta _ { 0 } } + \widehat { g } _ { 0 } ( X )$ for learning the regression function $D \theta _ { 0 } + g _ { 0 } ( X )$ .2 Suppose, for the sake of clarity, that we randomly split the sample into two parts: a main part of size $n$ , with observation numbers indexed by $i \in I$ , and an auxiliary part of size $N - n$ , with observations indexed by $i \in I ^ { c }$ . For simplicity, we take $n = N / 2$ for the moment and turn to more general cases which cover unequal split-sizes, using more than one split, and achieving the same efficiency as if the full sample were used for estimating $\theta _ { 0 }$ in the formal development in Section 3. Suppose $\widehat { g } _ { 0 }$ is obtained using the auxiliary sample and that, given this $\widehat { g } _ { 0 }$ , the final estimate of $\theta _ { 0 }$ ibs obtained using the main sample:

$$
\widehat { \theta } _ { 0 } = \Big ( \frac { 1 } { n } \sum _ { i \in I } D _ { i } ^ { 2 } \Big ) ^ { - 1 } \frac { 1 } { n } \sum _ { i \in I } D _ { i } \big ( Y _ { i } - \widehat { g } _ { 0 } ( X _ { i } ) \big ) .
$$

The estimator $\widehat { \theta _ { 0 } }$ will generally have a slower than $1 / \sqrt { n }$ rate of convergence, namely,

$$
\vert \sqrt { n } ( \widehat { \theta } _ { 0 } - \theta _ { 0 } ) \vert  _ { P } \infty .
$$

As detailed below, the driving force behind this “inferior” behavior is the bias in learning $g _ { 0 }$ . Figure provides a numerical illustration of this phenomenon for a naive ML estimator based on a random forest in a simple computational experiment.

To heuristically illustrate the impact of the bias in learning $g _ { 0 }$ , we can decompose the scaled estimation error in $\widehat { \theta _ { 0 } }$ as

$$
\sqrt { n } ( \widehat { \theta } _ { 0 } - \theta _ { 0 } ) = \underbrace { \Big ( \frac { 1 } { n } \sum _ { i \in I } D _ { i } ^ { 2 } \Big ) ^ { - 1 } \frac { 1 } { \sqrt { n } } \sum _ { i \in I } D _ { i } U _ { i } } _ { : = a } + \underbrace { \Big ( \frac { 1 } { n } \sum _ { i \in I } D _ { i } ^ { 2 } \Big ) ^ { - 1 } \frac { 1 } { \sqrt { n } } \sum _ { i \in I } D _ { i } \big ( g _ { 0 } \big ( X _ { i } \big ) - \widehat { g } _ { 0 } \big ( X _ { i } \big ) \big ) } _ { : = b } .
$$

The first term is well-behaved under mild conditions, obeying $a \sim N ( 0 , \Sigma )$ for some $\Sigma$ . Term $b$ is the regularization bias term, which is not centered and diverges in general. Indeed, we have

$$
b = ( \operatorname { E D } _ { i } 2 ) ^ { - 1 } { \frac { 1 } { \sqrt { n } } } \sum _ { i \in I } m _ { 0 } ( X _ { i } ) ( g _ { 0 } ( X _ { i } ) - { \widehat { g } } _ { 0 } ( X _ { i } ) ) + o _ { P } ( 1 )
$$

to the first order. Heuristically, $b$ is the sum of $n$ terms that do not have mean zero, $m _ { 0 } ( X _ { i } ) ( g _ { 0 } ( X _ { i } ) - { \widehat g } _ { 0 } ( X _ { i } ) )$ , divided by $\sqrt { n }$ . These terms have non-zero mean because, in high dimensionabl or otherwise highly complex settings, we must employ regularized estimators - such as lasso, ridge, boosting, or penalized neural nets - for informative learning to be feasible. The regularization in these estimators keeps the variance of the estimator from exploding but also necessarily induces substantive biases in the estimator $\widehat { g } _ { 0 }$ of $g _ { 0 }$ . Specifically, the rate of convergence of (the bias of) $\widehat { g } _ { 0 }$ to $g _ { 0 }$ in the root mean sbquared error sense will typically be $n ^ { - \varphi _ { g } }$ with $\varphi _ { g } < 1 / 2$ . Hebnce, we expect $b$ to be of stochastic order ${ \sqrt { n } } n ^ { - \varphi _ { g } }  \infty$ since $D _ { i }$ is centered at $m _ { 0 } ( X _ { i } ) \neq 0$ , which then implies (1.4).

Overcoming Regularization Biases using Orthogonalization. Now consider a second construction that employs an “orthogonalized” formulation obtained by directly partialling out the effect of $X$ from $D$ to obtain the orthogonalized regressor $V = D -$ $m _ { 0 } ( X )$ . Specifically, we obtain $\hat { V } = D - \widehat { m } _ { 0 } ( X )$ , where $\widehat { \ b { m } } _ { 0 }$ is an ML estimator of m0 obtained using the auxiliary sample of observations. We are now solving an auxiliary prediction problem to estimate the conditional mean of $D$ given $X$ , so we are doing “double prediction” or “double machine learning”.

![](/tmp/mineru_hhuwcznw/images/80373f6955ccb78fabcc01f4954389e59fe13b6fd30521109db3de645ad226c0.jpg)
Figure 1. Left Panel: Behavior of a conventional (non -orthogonal) ML estimator, $\widehat { \theta _ { 0 } }$ , in the partially linear model in a simple simulation experiment where we learn g0 using a random forbest. The $g _ { 0 }$ in this experiment is a very smooth function of a small number of variables, so the experiment is seemingly favorable to the use of random forests a priori. The histogram shows the simulated distribution of the centered estimator, $\widehat { \theta } _ { 0 } - \theta _ { 0 }$ . The estimator is badly biased, shifted much to the right relative to the true value $\theta _ { 0 }$ . The dibstribution of the estimator (approximated by the blue histogram) is substantively different from a normal approximation (shown by the red curve) derived under the assumption that the bias is negligible. Right Panel: Behavior of the orthogonal, DML estimator, $\check { \theta } _ { 0 }$ , in the partially linear model in a simple experiment where we learn nuisance functions using random forests. Note that the simulated data are exactly the same as those underlying left panel. The simulated distribution of the centered estimator, $\check { \theta } _ { 0 } - \theta _ { 0 } ^ { \check { \mathbf { \alpha } } }$ , (given by the blue histogram) illustrates that the estimator is approximately unbiased, concentrates around θ0, and is well-approximated by the normal approximation obtained in Section 3 (shown by the red curve).

After partialling the effect of $X$ out from $D$ and obtaining a preliminary estimate of $g _ { 0 }$ from the auxiliary sample as before, we may formulate the following “debiased” machine learning estimator for $\theta _ { 0 }$ using the main sample of observations:

$$
\check { \theta } _ { 0 } = \Big ( \frac { 1 } { n } \sum _ { i \in I } \widehat { V } _ { i } D _ { i } \Big ) ^ { - 1 } \frac { 1 } { n } \sum _ { i \in I } \widehat { V } _ { i } \big ( Y _ { i } - \widehat { g } _ { 0 } ( X _ { i } ) \big ) . ^ { 3 }
$$

By approximately orthogonalizing $D$ with respect to $X$ and approximately removing the direct effect of confounding by subtracting an estimate of $g _ { 0 }$ , $\check { \theta } _ { 0 }$ removes the effect of regularization bias that contaminates (1.3). The formulation of $\check { \theta } _ { 0 }$ also provides direct links to both the classical econometric literature, as the estimator can clearly be interpreted as a linear instrumental variable (IV) estimator, and to the more recent literature on debiased lasso in the context where $g _ { 0 }$ is taken to be well-approximated by a sparse linear combination of prespecified functions of $X$ ; see, e.g., Belloni et al. (2013); Zhang and Zhang (2014); Javanmard and Montanari (2014b); van de Geer et al. (2014); Belloni et al. (2014); and Belloni et al. (2014).4

To illustrate the benefits of the auxiliary prediction step and estimating $\theta _ { 0 }$ with $\check { \theta } _ { 0 }$ , we sketch the properties of $\check { \theta } _ { 0 }$ here. We can decompose the scaled estimation error of $\check { \theta } _ { 0 }$ into three components:

$$
\sqrt { n } ( \check { \theta } _ { 0 } - \theta _ { 0 } ) = a ^ { * } + b ^ { * } + c ^ { * } .
$$

The leading term, $a ^ { * }$ , will satisfy

$$
a ^ { * } = ( \mathrm { E } V ^ { 2 } ) ^ { - 1 } \frac { 1 } { \sqrt { n } } \sum _ { i \in I } V _ { i } U _ { i } \sim N ( 0 , \Sigma )
$$

under mild conditions. The second term, $b ^ { * }$ , captures the impact of regularization bias in estimating $g _ { 0 }$ and $m _ { 0 }$ . Specifically, we will have

$$
b ^ { * } = ( \mathrm { E } V ^ { 2 } ) ^ { - 1 } \frac { 1 } { \sqrt { n } } \sum _ { i \in I } ( \widehat { m } _ { 0 } ( X _ { i } ) - m _ { 0 } ( X _ { i } ) ) ( \widehat { g } _ { 0 } ( X _ { i } ) - g _ { 0 } ( X _ { i } ) ) ,
$$

which now depends on the product of the estimation errors in $\widehat { m } _ { 0 }$ and $\widehat { g } _ { 0 }$ . Because this term depends only on the product of the estimation errors, it bn vanishb under a broad range of data-generating processes. Indeed, this term is upper-bounded by $\sqrt { n } n ^ { - ( \varphi _ { m } + \varphi _ { g } ) }$ , where $n ^ { - \varphi _ { m } }$ and $n ^ { - \varphi _ { g } }$ are respectively the rates of convergence of $\hat { m } _ { 0 }$ to $m _ { 0 }$ and $\widehat { g } _ { 0 }$ to $g _ { 0 }$ ; and this upper bound can clearly vanish even though both m0 nbd $g _ { 0 }$ are estimbated at relatively slow rates. Verifying that $\check { \theta } _ { 0 }$ has good properties then requires that the remainder term, $c ^ { * }$ , is sufficiently well-behaved. Sample-splitting will play a key role in allowing us to guarantee that $c ^ { * } = o _ { P } ( 1 )$ under weak conditions as outlined below and discussed in detail in Section 3.

The Role of Sample Splitting in Removing Bias Induced by Overfitting. Our analysis makes use of sample-splitting which plays a key role in establishing that remainder terms, like $c ^ { * }$ , vanish in probability. In the partially linear model, we have that the remainder $c ^ { * }$ contains terms like

$$
{ \frac { 1 } { \sqrt { n } } } \sum _ { i \in I } V _ { i } ( { \widehat { g } } _ { 0 } ( X _ { i } ) - g _ { 0 } ( X _ { i } ) )
$$

that involve $1 / \sqrt { n }$ normalized sums of products of structural unobservables from model (1.1)-(1.2) with estimation errors in learning the nuisance functions $g _ { 0 }$ and $m _ { 0 }$ and need to be shown to vanish in probability. The use of sample splitting allows simple and tight control of such terms. To see this, assume that observations are independent and recall that $\widehat { g } _ { 0 }$ is estimated using only observations in the auxiliary sample. Then, conditioning on thbe auxiliary sample and recalling that $\operatorname { E } [ V _ { i } | X _ { i } ] = 0$ , it is easy to verify that term (1.6) has mean zero and variance of order

$$
{ \frac { 1 } { n } } \sum _ { i \in I } ( { \widehat { g } } _ { 0 } ( X _ { i } ) - g _ { 0 } ( X _ { i } ) ) ^ { 2 } \to _ { P } 0 .
$$

Thus, the term (1.6) vanishes in probability by Chebyshev’s inequality.

While sample splitting allows us to deal with remainder terms such as $c ^ { * }$ , its direct application does have the drawback that the estimator of the parameter of interest only makes use of the main sample which may result in a substantial loss of efficiency as we are only making use of a subset of the available data. However, we can flip the role of the main and auxiliary samples to obtain a second version of the estimator of the parameter of interest. By averaging the two resulting estimators, we may regain full efficiency. Indeed, the two estimators will be approximately independent, so simply averaging them offers an efficient procedure. We call this sample splitting procedure where we swap the roles of main and auxiliary samples to obtain multiple estimates and then average the results cross-fitting. We formally define this procedure and discuss a $K$ -fold version of cross-fitting in Section 3.

Without sample splitting, terms such as (1.6) may not vanish and can lead to poor performance of estimators of $\theta _ { 0 }$ . The difficulty arises because model errors, such as $V _ { i }$ , and estimation errors, such as ${ \widehat { g } } _ { 0 } ( X _ { i } ) - g _ { 0 } ( X _ { i } )$ , are generally related because the data for observation $i$ is used in formbing the estimator $\widehat { g } _ { 0 }$ . The association may then lead to poor performance of an estimator of $\theta _ { 0 }$ that makesb use of $\widehat { g } _ { 0 }$ as a plug-in estimator for $g _ { 0 }$ even when this estimator converges at a very favorable rbate, say N −1/2+ϵ.

As an artificial but illustrative example of the problems that may result from overfitting, let $\widehat { g } _ { 0 } ( X _ { i } ) = g _ { 0 } ( X _ { i } ) + ( Y _ { i } - g _ { 0 } ( X _ { i } ) ) / N ^ { 1 / 2 - \epsilon }$ for any $i$ in the sample used to form estimatorb $\widehat { g } _ { 0 }$ , and note that the second term provides a simple model that captures overfitting of tbhe outcome variable within the estimation sample. This estimator is excellent in terms of rates: If the $U _ { i }$ ’s and $D _ { i }$ ’s are bounded, $\widehat { g } _ { 0 }$ converges uniformly to $g _ { 0 }$ at the nearly parametric rate $N ^ { - 1 / 2 + \epsilon }$ . Despite this fast bate of convergence, term $c ^ { * }$ c now explodes if we do not use sample splitting. For example, suppose that the full sample is used to estimate both $\widehat { g } _ { 0 }$ and $\check { \theta } _ { 0 }$ . A simple calculation then reveals that term $c ^ { * }$ becomes

$$
\frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } V _ { i } ( \widehat { g } _ { 0 } ( X _ { i } ) - g _ { 0 } ( X _ { i } ) ) \propto N ^ { \epsilon }  \infty .
$$

This bias due to overfitting is illustrated in the left panel of Figure 2. The histogram in the figure gives a simulated distribution for the studentized $\check { \theta }$ resulting from using the full sample and the contrived estimator $\widehat g ( X _ { i } )$ given above. We can see that the histogram is shifted markedly to the left demonstrbating substantial bias resulting from overfitting. The right panel of Figure 2 also illustrates that this bias is completely removed by sample splitting. The results the right panel of Figure 2 make use of the two-fold crossfitting procedure discussed above using the estimator $\check { \theta }$ and the contrived estimator $\widehat g ( X _ { i } )$ exactly as in the left panel. The difference is that $\widehat g ( X _ { i } )$ is formed in one half of tbhe sample and then $\check { \theta }$ is estimated using the other half obf the sample. This procedure is then repeated swapping the roles of the two samples and the results are averaged. We can see that the substantial bias from the full sample estimator has been removed and that the spread of the histogram corresponding to the cross-fit estimator is roughly the same as that of the full sample estimator clearly illustrating the bias-reduction property and efficiency of the cross -fitting procedure.

A less contrived example that highlights the improvements brought by sample-splitting is the sparse high-dimensional instrumental variable (IV) model analyzed in Belloni et al. (2012). Specifically, they consider the IV model

$$
Y = D \theta _ { 0 } + \epsilon
$$

where $\mathrm { E } [ \epsilon | D ] \ \ne \ 0$ but instruments $Z$ exist such that $\mathrm { E } [ D | Z ]$ is not a constant and $\mathrm { E } [ \epsilon | Z ] = 0$ . Within this model, Belloni et al. (2012) focus on the problem of estimating the optimal instrument, $\eta _ { 0 } ( Z ) = \mathrm { E } [ D | Z ]$ using lasso-type methods. If $\eta _ { 0 } ( Z )$ is approximately sparse in the sense that only $s$ terms of the dictionary of series transformations $B ( Z ) =$ $( B _ { 1 } ( Z ) , \ldots , B _ { p } ( Z ) )$ are needed to approximate the function accurately, Belloni et al. (2012) require that $s ^ { 2 } \ll n$ to establish their asymptotic results when sample splitting is not used but show that these results continue to hold under the much weaker requirement that $s \ll n$ if one employs sample splitting. We note that this example provides a prototypical example where Neyman orthogonality holds and ML methods can usefully be adopted to aid in learning structural parameters of interest. We also note that the weaker conditions required when using sample sample-splitting would also carry over to sparsity-based estimators in the partially linear model cited above. We discuss this in more detail in Section 4.

![](/tmp/mineru_hhuwcznw/images/c670b5e9402d7984459839183c59de2869b941213fbb60a6e018dae2b157ab78.jpg)
Figure 2. This figure illustrates how the bias resulting from overfitting in the estimation of nuisance functions can cause the main estimator $\check { \theta } _ { 0 }$ to be biased and how sample splitting completely eliminates this problem. Left Panel: The histogram shows the finite-sample distribution of $ { \bar { \theta _ { 0 } } }$ in the partially linear model where nuisance parameters are estimated with overfitting using the full sample, i.e. without sample splitting. The finite-sample distribution is clearly shifted to the left of the true parameter value demonstrating the substantial bias. Right Panel: The histogram shows the finite-sample distribution of $\check { \theta } _ { 0 }$ in the partially linear model where nuisance parameters are estimated with overfitting using the cross-fitting sample-splitting estimator. Here, we see that the use of sample-splitting has completely eliminated the bias induced by overfitting.

While we find substantial appeal in using sample-splitting, one may also use empirical process methods to verify that biases introduced due to overfitting are negligible. For example, consider the problematic term in the partially linear model described previously, $\begin{array} { r } { \frac { 1 } { \sqrt { n } } \sum _ { i \in I } V _ { i } ( \widehat { g } _ { 0 } ( X _ { i } ) - g _ { 0 } ( X _ { i } ) ) } \end{array}$ . This term is clearly bounded by

$$
\operatorname* { s u p } _ { g \in { \mathcal { G } } _ { N } } { \Big | } { \frac { 1 } { \sqrt { n } } } \sum _ { i \in I } V _ { i } { \big ( } g ( X _ { i } ) - g _ { 0 } ( X _ { i } ) { \big ) } { \Big | } ,
$$

where $\mathcal { G } _ { N }$ is the smallest class of functions that contains estimators of $g _ { 0 }$ , $\widehat g$ , with high probability. In conventional semiparametric statistical and econometric analysbis, the complexity of $\mathcal { G } _ { N }$ is controlled by invoking Donsker conditions which allow verification that terms such as (1.7) vanish asymptotically. Importantly, Donsker conditions require that $\mathcal { G } _ { N }$ has bounded complexity, specifically a bounded entropy integral. Because of the latter property, Donsker conditions are inappropriate in settings using ML methods where the dimension of $X$ is modeled as increasing with the sample size and estimators necessarily live in highly complex spaces. For example, Donsker conditions rule out even the simplest linear parametric model with high-dimensional regressors with parameter space given by the Euclidean ball with the unit radius:

$$
\mathcal { G } _ { N } = \{ x \mapsto g ( x ) = x ^ { \prime } \theta ; ~ \theta \in \mathbb { R } ^ { p _ { N } } : \| \theta \| \leqslant 1 \} .
$$

The entropy of this model, as measured by the logarithm of the covering number, grows at the rate $p _ { N }$ . Without invoking Donsker conditions, one may still show that terms such as (1.7) vanish as long as $\mathcal { G } _ { N }$ ’s entropy does not increase with $N$ too rapidly. A fairly general treatment is given in Belloni et al. (2017) who provide a set of conditions under which terms like $c ^ { * }$ can vanish making use of the full sample. However, these conditions on the growth of entropy could result in unnecessarily strong restrictions on model complexity, such as very strict requirements on sparsity in the context of lasso estimation as demonstrated in IV example mentioned above. Sample splitting allows one to obtain good results under very weak conditions.

Neyman Orthogonality and Moment Conditions. Now we turn to a generalization of the orthogonalization principle above. The first “conventional” estimator $\widehat { \theta _ { 0 } }$ given in (1.3) can be viewed as a solution to estimating equations

$$
\frac { 1 } { n } \sum _ { i \in I } \varphi ( W ; \widehat { \theta } _ { 0 } , \widehat { g } _ { 0 } ) = 0 ,
$$

where $\varphi$ is a known “score” function and $\widehat { g } _ { 0 }$ is the estimator of the nuisance parameter $g _ { 0 }$ . For example, in the partially linear mobdel above, the score function is $\varphi ( W ; \theta , g ) =$ $( Y - \theta D - g ( X ) ) D$ . It is easy to see that this score function $\varphi$ is sensitive to biased estimation of $g$ . Specifically, the Gateaux derivative operator with respect to $g$ does not vanish:

$$
\partial _ { g } \mathrm { E } \varphi ( W ; \theta _ { 0 } , g _ { 0 } ) [ g - g _ { 0 } ] \neq 0 . ^ { ! }
$$

The proofs of the general results in Section 3 show that this term’s vanishing is a key to establishing good behavior of an estimator for θ0.

By contrast the orthogonalized or double/debiased ML estimator $\check { \theta } _ { 0 }$ given in (1.5) solves

$$
\frac { 1 } { n } \sum _ { i \in I } \psi ( W ; \check { \theta } _ { 0 } , \widehat { \eta } _ { 0 } ) = 0 ,
$$

where $\widehat { \eta _ { 0 } }$ is the estimator of the nuisance parameter $\eta _ { 0 }$ and $\psi$ is an orthogonalized or debia bd “score” function that satisfies the property that the Gateaux derivative operator with respect to $\eta$ vanishes when evaluated at the true parameter values:

$$
\partial _ { \eta } \mathrm { E } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] = 0 .
$$

We refer to property (1.8) as “Neyman orthogonality” and to $\psi$ as the Neyman orthogonal score function due to the fundamental contributions in Neyman (1959) and Neyman (1979), where this notion was introduced. Intuitively, the Neyman orthogonality condition means that the moment conditions used to identify $\theta _ { 0 }$ are locally insensitive to the value of the nuisance parameter which allows one to plug-in noisy estimates of these parameters without strongly violating the moment condition. In the partially linear model (1.1)-(1.2), the estimator $\check { \theta } _ { 0 }$ uses the score function $\psi ( W ; \theta , \eta ) = ( Y - D \alpha - g ( X ) ) ( D - m ( X ) )$ , with the nuisance parameter being $\eta = ( m , g )$ . It is easy to see that these score functions $\psi$ are not sensitive to biased estimation of $\eta _ { 0 }$ in the sense that (1.8) holds. The proofs of the general results in Section 3 show that this property and sample splitting are two generic keys that allow establishing good behavior of an estimator for $\theta _ { 0 }$ .

# 1.2. Literature Overview

Our paper builds upon two important bodies of research within the semiparametric literature. The first is the literature on obtaining $\sqrt { N }$ -consistent and asymptotically normal estimates of low-dimensional objects in the presence of high-dimensional or nonparametric nuisance functions. The second is the literature on the use of sample-splitting to relax entropy conditions. We provide links to each of these literatures in turn.

The problem we study is obviously related to the classical semiparametric estimation framework which focuses on obtaining $\sqrt { N }$ -consistent and asymptotically normal estimates for low-dimensional components with nuisance parameters estimated by conventional nonparametric estimators such as kernels or series. See, for example, the work by Levit (1975), Ibragimov and Hasminskii (1981), Bickel (1982), Robinson (1988), Newey (1990), van der Vaart (1991), Andrews (1994a), Newey (1994), Newey et al. (1998), Robins and Rotnitzky (1995), Linton (1996), Bickel et al. (1998), Chen et al. (2003), Newey et al. (2004), van der Laan and Rose (2011), and Ai and Chen (2012). Neyman orthogonality (1.8), introduced by Neyman (1959), plays a key role in optimal testing theory and adaptive estimation, semiparametric learning theory and econometrics, and, more recently, targeted learning theory. For example, Andrews (1994a), Newey (1994) and van der Vaart (1998) provide a general set of results on estimation of a low-dimensional parameter $\theta _ { 0 }$ in the presence of nuisance parameters $\eta _ { 0 }$ . Andrews (1994a) uses Neyman orthogonality (1.8) and Donsker conditions to demonstrate the key equicontinuity condition

$$
\frac { 1 } { \sqrt { n } } \sum _ { i \in I } \Big ( \psi ( W _ { i } ; \theta _ { 0 } , \widehat { \eta } ) - \int \psi ( w ; \theta _ { 0 } , \widehat { \eta } ) d P ( w ) - \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) \Big ) \to _ { P } 0 ,
$$

which reduces to (1.6) in the partially linear regression model. Newey (1994) gives conditions on estimating equations and nuisance function estimators so that nuisance function estimators do not affect the limiting distribution of parameters of interest, providing a semiparametric version of Neyman orthogonality. van der Vaart (1998) discusses use of semiparametrically efficient scores to define estimators that solve estimating equations setting averages of efficient scores to zero. He also uses efficient scores to define k-step estimators, where a preliminary estimator is used to estimate the efficient score and then updating is done to further improve estimation; see also comments below on the use of sample-splitting.

There is also a related targeted maximum likelihood learning approach, introduced in Scharfstein et al. (1999) in the context of treatments effects analysis and substantially generalized by van der Laan and Rubin (2006). van der Laan and Rubin (2006) use maximum likelihood in a least favorable direction and then perform “one-step” or “k-step” updates using the estimated scores in an effort to better estimate the target parameter.6 This procedure is like the least favorable direction approach in semiparametrics; see, for example, Severini and Wong (1992). The introduction of the likelihood introduces major benefits such as allowing simple and natural imposition of constraints inherent in the data, such as support restrictions when the outcome is binary or censored, and permitting the use of likelihood cross-validation to choose the nuisance parameter estimator. This data adaptive choice of the nuisance parameter has been dubbed the “super learner” by van der Laan et al. (2007). In subsequent work, van der Laan and Rose (2011) emphasize the use of ML methods to estimate the nuisance parameters for use with the super learner. Much of this work, including recent work such as Luedtke and van der Laan (2016), Toth and van der Laan (2016), and Zheng et al. (2016), focuses on formal results under a Donsker condition, though the use of sample splitting to relax these conditions has also been advocated in the targeted maximum likelihood setting as discussed below.

The Donsker condition is a powerful classical condition that allows rich structures for fixed function classes $\mathcal { G }$ , but it is unfortunately unsuitable for high-dimensional settings. Examples of function classes where a Donsker condition holds include functions of a single variable that have total variation bounded by 1 and functions $x \mapsto f ( x )$ that have $r > \dim ( x ) / 2$ uniformly bounded derivatives. As a further example, functions composed from function classes with VC dimensions bounded by $p$ p through a fixed number of algebraic and monotone transforms are Donsker. However, this property will no longer hold if we let $\dim ( x )$ grow to infinity with the sample size as this increase in dimension would require that the VC dimension also increases with $n$ . More generally, Donsker conditions are easily violated once dimensions get large. A major point of departure of the present work from the classical literature on semiparametric estimation is its explicit focus on high-complexity/entropy cases. One way to analyze the problem of estimation in high-entropy cases is to see to what degree equicontinuity results continue to hold while allowing moderate growth of the complexity/entropy of $\mathcal { G } _ { N }$ . Examples of papers taking this approach in an approximately sparse settings are Belloni et al. (2017), Belloni et al. (2014), Belloni et al. (2016), Chernozhukov et al. (2015b), Javanmard and Montanari (2014a), van de Geer et al. (2014), and Zhang and Zhang (2014). In all of these examples, entropy growth must be limited in what may be very restrictive ways. The entropy conditions rule out the contrived overfitting example mentioned above, which does approximate realistic examples, and may otherwise place severe restrictions on the model. For example, in Belloni et al. (2010) and Belloni et al. (2012), the optimal instrument needs to be sparse of order $s \ll \sqrt { n }$ .

A key device that we use to avoid strong entropy conditions is cross-fitting via sample splitting. Cross-fitting is a practical, efficient form of data splitting. Importantly, its use here is not simply as a device to make proofs elementary (which it does), but as a practical method to allow us to overcome the overfitting/high-complexity phenomena that commonly arise in data analysis based on highly adaptive ML methods. Our treatment builds upon the sample-splitting ideas employed in Belloni et al. (2010) and Belloni et al. (2012) who considered sample-splitting in a high-dimensional sparse optimal IV model to weaken the sparsity condition mentioned in the previous paragraph to $s \ll n$ . This work in turn was inspired by Angrist and Krueger (1995). We also build on Ayyagari (2010) and Robins et al. (2013), where ML methods and sample splitting were used in the estimation of a partially linear model of the effects of pollution while controlling for several covariates. We use the term “cross-fitting” to characterize our recommended procedure, partly borrowing the jargon from Fan et al. (2012) which employed a slightly different form of sample-splitting to estimate the scale parameter in a high-dimensional sparse regression. Of course, the use of sample-splitting to relax entropy conditions has a long history in semiparametric estimation problems. For example, Bickel (1982) considered estimating nuisance functions using a vanishing fraction of the sample, and these results were extended to sample splitting into two equal halves and discretization of the parameter space by Schick (1986). Similarly, van der Vaart (1998) uses 2-way sample splitting and discretization of the parameter space to give weak conditions for k-step estimators using the efficient scores where sample splitting is used to estimate the “updates”; see also Hubbard et al. (2016). Robins et al. (2008) and Robins et al. (2017) use sample splitting in the construction of higher-order influence function corrections in semiparametric estimation. Some recent work in the targeted maximum likelihood literature, for example Zheng and van der Laan (2011), also notes the utility of sample splitting in the context of k-step updating, though this sample splitting approach is different from the cross-fitting approach we pursue.

Plan of the Paper. We organize the rest of the paper as follows. In Section 2, we formally define Neyman orthogonality and provide a brief discussion that synthesizes various models and frameworks that may be used to produce estimating equations satisfying this key condition. In Section 3, we carefully define DML estimators and develop their general theory. We then illustrate this general theory by applying it to provide theoretical results for using DML to estimate and do inference for key parameters in the partially linear regression model and for using DML to estimate and do inference for coefficients on endogenous variables in a partially linear instrumental variables model in Section 4. In Section 5, we provide a further illustration of the general theory by applying it to develop theoretical results for DML estimation and inference for average treatment effects and average treatment effects on the treated under unconfoundedness and for DML estimation of local average treatment effects in an IV context within the potential outcomes framework; see Imbens and Rubin (2015). Finally, we apply DML in three empirical illustrations in Section 6. In an appendix, we define additional notation and present proofs.

Notation. The symbols $\mathrm { P }$ and $\mathrm { E }$ denote probability and expectation operators with respect to a generic probability measure that describes the law of the data. If we need to signify the dependence on a probability measure $P$ , we use $P$ as a subscript in $\mathrm { P } _ { P }$ and $\mathrm { E } _ { P }$ . We use capital letters, such as $W$ , to denote random elements and use the corresponding lower case letters, such as $w$ , to denote fixed values that these random elements can take. In what follows, we use $\| \cdot \| _ { P , q }$ to denote the $L ^ { q } ( P )$ norm; for example, we denote $\begin{array} { r } { \Vert f \Vert _ { P , q } : = \Vert f ( W ) \Vert _ { P , q } : = \left( \int | f ( w ) | ^ { q } d P ( w ) \right) ^ { 1 / q } } \end{array}$ , where $\| f \| _ { P , \infty }$ stands for the essential supremum. We use $x ^ { \prime }$ to denote the transpose of a column vector $x$ . For a differentiable map $x \mapsto f ( x )$ , mapping $\mathbb { R } ^ { d }$ to $\mathbb { R } ^ { k }$ , we use $\partial _ { x ^ { \prime } } f$ to abbreviate the partial derivatives $( \partial / \partial x ^ { \prime } ) f$ , and we correspondingly use the expression $\partial _ { x ^ { \prime } } f ( x _ { 0 } )$ to mean $\partial _ { x ^ { \prime } } f ( x ) \mid _ { x = x _ { 0 } }$ , etc.

# 2. CONSTRUCTION OF NEYMAN ORTHOGONAL SCORE/MOMENT FUNCTIONS

Here we formally introduce the model and discuss several methods for generating orthogonal scores in a wide variety of settings, including the classical Neyman’s construction. We also use this as an opportunity to synthesize some recent developments in the literature.

We are interested in the true value $\theta _ { 0 }$ of the low-dimensional target parameter $\theta \in \Theta$ , where $\Theta$ is a non-empty measurable subset of $\mathbb { R } ^ { d _ { \theta } }$ . We assume that $\theta _ { 0 }$ satisfies the moment conditions

$$
\begin{array} { r } { \mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] = 0 , } \end{array}
$$

where $\psi = ( \psi _ { 1 } , \ldots , \psi _ { d _ { \theta } } ) ^ { \prime }$ is a vector of known score functions, $W$ is a random element taking values in a measurable space $( \mathcal { W } , A _ { \mathcal { W } } )$ with law determined by a probability measure $P \in \mathcal P _ { N }$ , and $\eta _ { 0 }$ is the true value of the nuisance parameter $\eta \in T$ , where $T$ is a convex subset of some normed vector space with the norm denoted by $\| \cdot \| _ { T }$ We assume that the score functions $\psi _ { j } : \mathcal { W } \times \Theta \times T  \mathbb { R }$ are measurable once we equip $\Theta$ and $T$ with their Borel $\sigma$ -fields, and we assume that a random sample $( W _ { i } ) _ { i = 1 } ^ { N }$ from the distribution of $W$ is available for estimation and inference.

As discussed in the Introduction, we require the Neyman orthogonality condition for the score $\psi$ . To introduce the condition, for $\widetilde T = \{ \eta - \eta _ { 0 } : \eta \in T \}$ we define the pathwise (or the Gateaux) derivative map $\mathrm { D } _ { r } \colon \tilde { T }  \mathbb { R } ^ { d _ { \theta } }$ ,

$$
\begin{array} { r } { \mathrm { D } _ { r } [ \eta - \eta _ { 0 } ] : = \partial _ { r } \bigg \{ \mathrm { E } _ { P } \Big [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) \Big ] \bigg \} , \quad \eta \in T , } \end{array}
$$

for all $r \in [ 0 , 1 )$ , which we assume to exist. For convenience, we also denote

$$
\partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] : = \mathrm { D } _ { 0 } [ \eta - \eta _ { 0 } ] , \quad \eta \in T .
$$

Note that $\psi ( W ; \theta _ { 0 } , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) )$ here is well-defined because for all $r \in [ 0 , 1 )$ and $\eta \in T$ ,

$$
\eta _ { 0 } + r ( \eta - \eta _ { 0 } ) = ( 1 - r ) \eta _ { 0 } + r \eta \in T
$$

since $T$ is a convex set. In addition, let $\mathcal { T } _ { N } \subset T$ be a nuisance realization set such that the estimators $\hat { \eta _ { 0 } }$ of $\eta _ { 0 }$ specified below take values in this set with high probability. In practice, we typbically assume that $\mathcal { T } _ { N }$ is a properly shrinking neighborhood of $\eta _ { 0 }$ . Note that $\mathcal { T } _ { N } - \eta _ { 0 }$ is the nuisance deviation set, which contains deviations of $\widehat { \eta _ { 0 } }$ from $\eta _ { 0 }$ , $\hat { \eta } _ { 0 } - \eta _ { 0 }$ , with high probability. The Neyman orthogonality condition requires tbhat the derbivative in (2.2) vanishes for all $\eta \in \mathcal { T } _ { N }$ .

Definition 2.1. (Neyman orthogonality) The score $\psi = ( \psi _ { 1 } , \ldots , \psi _ { d _ { \theta } } ) ^ { \prime }$ obeys the orthogonality condition at $( \theta _ { 0 } , \eta _ { 0 } )$ with respect to the nuisance realization set $\mathcal { T } _ { N } \subset T$ if (2.1) holds and the pathwise derivative map $\mathrm { D } _ { r } [ \eta - \eta _ { 0 } ]$ exists for all $r \in [ 0 , 1 )$ and $\eta \in \mathcal { T } _ { N }$ and vanishes at $r = 0$ ; namely,

$$
\partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] = 0 , \quad f o r \ a l l \ \eta \in \mathcal { T } _ { N } .
$$

We remark here that condition (2.3) holds with $\mathcal { T } _ { N } = T$ when $\eta$ is a finite-dimensional vector as long as $\partial _ { \eta } \mathrm { E } _ { P } [ \psi _ { j } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] = 0$ for all $j = 1 , \ldots , d _ { \theta }$ , where $\partial _ { \eta } \mathrm { E } _ { P } [ \psi _ { j } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ]$ denotes the vector of partial derivatives of the function $\eta \mapsto \operatorname { E } _ { P } [ \psi _ { j } ( W ; \theta _ { 0 } , \eta ) ]$ for $\eta = \eta _ { 0 }$ .

Sometimes it will also be helpful to use an approximate Neyman orthogonality condition as opposed to the exact one given in Definition 2.1:

Definition 2.2. (Neyman Near-Orthogonality) The score $\psi = ( \psi _ { 1 } , \dots , \psi _ { d _ { \theta } } ) ^ { \prime }$ obeys the $\lambda _ { N }$ near-orthogonality condition at $( \theta _ { 0 } , \eta _ { 0 } )$ with respect to the nuisance realization set

$\mathcal { T } _ { N } \subset T$ if (2.1) holds and the pathwise derivative map $\mathrm { D } _ { r } [ \eta - \eta _ { 0 } ]$ exists for all $r \in [ 0 , 1 )$ and $\eta \in \mathcal { T } _ { N }$ and is small at $r = 0$ ; namely,

$$
\left\| \partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] \right\| \leqslant \lambda _ { N } , \quad f o r ~ a l l ~ \eta \in \mathcal { T } _ { N } ,
$$

where $\left\{ \lambda _ { N } \right\} _ { N \geqslant 1 }$ is a sequence of positive constants such that $\lambda _ { N } = o ( N ^ { - 1 / 2 } )$ .

# 2.2. Construction of Neyman Orthogonal Scores

If we start with a score $\varphi$ that does not satisfy the orthogonality condition above, we first transform it into a score $\psi$ that does. Here we outline several methods for doing so.

2.2.1. Neyman Orthogonal Scores for Likelihood and Other M-Estimation Problems with Finite-Dimensional Nuisance Parameters

First, we describe the construction used by Neyman (1959) to derive his celebrated orthogonal score and $C ( \alpha )$ -statistic in a maximum likelihood setting. Such construction also underlies the concept of local unbiasedness in construction of optimal tests in e.g. Ferguson (1967) and was extended to non-likelihood settings by Wooldridge (1991). The discussion of Neyman’s construction here draws on Chernozhukov et al. (2015a).

To describe the construction, let $\theta \in \Theta \subset \mathbb { R } ^ { d _ { \theta } }$ and $\beta \in B \subset \mathbb { R } ^ { d _ { \beta } }$ , where $\boldsymbol { B }$ is a convex set, be the target and the nuisance parameters, respectively. Further, suppose that the true parameter values $\theta _ { 0 }$ and $\beta _ { 0 }$ solve the optimization problem

$$
\operatorname* { m a x } _ { \theta \in \Theta , \ \beta \in B } \mathrm { E } _ { P } [ \ell ( W ; \theta , \beta ) ] ,
$$

where $\ell ( W ; \theta , \beta )$ is a known criterion function. For example, $\ell ( W ; \theta , \beta )$ can be the loglikelihood function associated to observation $W$ . More generally, we refer to $\ell ( W ; \theta , \beta )$ as the quasi-log-likelihood function. Then, under mild regularity conditions, $\theta _ { 0 }$ and $\beta _ { 0 }$ satisfy

$$
\begin{array} { r } { \mathrm { E } _ { P } [ \partial _ { \theta } \ell ( W ; \theta _ { 0 } , \beta _ { 0 } ) ] = 0 , \quad \mathrm { E } _ { P } [ \partial _ { \beta } \ell ( W ; \theta _ { 0 } , \beta _ { 0 } ) ] = 0 . } \end{array}
$$

Note that the original score function $\varphi ( W ; \theta , \beta ) = \partial _ { \theta } \ell ( W ; \theta , \beta )$ for estimating $\theta _ { 0 }$ will not generally satisfy the orthogonality condition. Now consider the new score function, which we refer to as the Neyman orthogonal score,

$$
\psi ( W ; \theta , \eta ) = \partial _ { \theta } \ell ( W ; \theta , \beta ) - \mu \partial _ { \beta } \ell ( W ; \theta , \beta ) ,
$$

where the nuisance parameter is

$$
\eta = ( \beta ^ { \prime } , \mathrm { v e c } ( \mu ) ^ { \prime } ) ^ { \prime } \in T = \mathcal { B } \times \mathbb { R } ^ { d _ { \theta } d _ { \beta } } \subset \mathbb { R } ^ { p } , \quad p = d _ { \beta } + d _ { \theta } d _ { \beta } ,
$$

and $\mu$ is the $d _ { \theta } \times d _ { \beta }$ orthogonalization parameter matrix whose true value $\mu _ { 0 }$ solves the equation

$$
J _ { \theta \beta } - \mu J _ { \beta \beta } = 0
$$

for

$$
\begin{array} { r } { J = \left( \begin{array} { l l } { J _ { \theta \theta } } & { J _ { \theta \beta } } \\ { J _ { \beta \theta } } & { J _ { \beta \beta } } \end{array} \right) = \partial _ { ( \theta ^ { \prime } , \beta ^ { \prime } ) } \mathrm { E } _ { P } \Big [ \partial _ { ( \theta ^ { \prime } , \beta ^ { \prime } ) ^ { \prime } } \ell ( W ; \theta , \beta ) \Big ] \Big | _ { \theta = \theta _ { 0 } ; ~ \beta = \beta _ { 0 } } . } \end{array}
$$

7The $C ( \alpha )$ -statistic, or the orthogonal score statistic, has been explicitly used for testing and estimation in high-dimensional sparse models in Belloni et al. (2015).

The true value of the nuisance parameter $\eta$ is

$$
\eta _ { 0 } = ( \beta _ { 0 } ^ { \prime } , \mathrm { v e c } ( \mu _ { 0 } ) ^ { \prime } ) ^ { \prime } ;
$$

and when $J _ { \beta \beta }$ is invertible, (2.8) has the unique solution,

$$
\mu _ { 0 } = J _ { \theta \beta } J _ { \beta \beta } ^ { - 1 } .
$$

The following lemma shows that the score $\psi$ in (2.7) satisfies the Neyman orthogonality condition.

Lemma 2.1. (Neyman Orthogonal Scores for Quasi-Likelihood Settings) If (2.6) holds, $J$ exists, and $J _ { \beta \beta }$ is invertible, the score $\psi$ in (2.7) is Neyman orthogonal at $( \theta _ { 0 } , \eta _ { 0 } )$ with respect to the nuisance realization set $\mathcal { T } _ { N } = T$ .

Remark 2.1. (Additional nuisance parameters) Note that the orthogonal score $\psi$ in (2.7) has nuisance parameters consisting of the elements of $\mu$ in addition to the elements of $\beta$ , and Lemma 2.1 shows that Neyman orthogonality holds both with respect to $\beta$ and with respect to $\mu$ . We will find that Neyman orthogonal scores in other settings, including infinite-dimensional ones, have a similar property.

Remark 2.2. (Efficiency) Note that in this example, $\mu _ { 0 }$ not only creates the necessary orthogonality but also creates the efficient score for inference on the target parameter $\theta$ when the quasi-log-likelihood function is the true (possibly conditional) log-likelihood, as demonstrated by Neyman (1959).

Example 2.1. (High-Dimensional Linear Regression) As an application of the construction above, consider the following linear predictive model:

$$
\begin{array} { r } { Y = D \theta _ { 0 } + X ^ { \prime } \beta _ { 0 } + U , \quad \mathrm { E } _ { P } [ U ( X ^ { \prime } , D ) ^ { \prime } ] = 0 , } \\ { D = X ^ { \prime } \gamma _ { 0 } + V , \qquad \mathrm { E } _ { P } [ V X ] = 0 , \qquad } \end{array}
$$

where for simplicity we assume that $\theta _ { 0 }$ is a scalar. The first equation here is the main predictive model, and the second equation only plays a role in the construction of the Neyman orthogonal scores. It is well-known that $\theta _ { 0 }$ and $\beta _ { 0 }$ in this model solve the optimization problem (2.5) with

$$
\ell ( W ; \theta , \beta ) = - \frac { ( Y - D \theta - X ^ { \prime } \beta ) ^ { 2 } } { 2 } , \quad \theta \in \Theta =  { \mathbb { R } } , \ \beta \in  { \mathcal { B } } =  { \mathbb { R } } ^ { d _ { \beta } } ,
$$

where we denoted $W = ( Y , D , X ^ { \prime } ) ^ { \prime }$ . Hence, equations (2.6) hold with

$$
\theta , \beta ) = ( Y - D \theta - X ^ { \prime } \beta ) D , \quad \partial \ell _ { \beta } ( W ; \theta , \beta ) = ( Y -
$$

and the matrix $J$ satisfies

$$
{ \cal J } _ { \theta \beta } = - \mathrm { E } _ { P } [ D X ^ { \prime } ] , \quad { \cal J } _ { \beta \beta } = - \mathrm { E } _ { P } [ X X ^ { \prime } ] .
$$

The Neyman orthogonal score is then given by

$$
\ddot { \cal W } ; \theta , \eta ) = ( { \cal Y } - { \cal D } \theta - { \cal X } ^ { \prime } \beta ) ( { \cal D } - \mu { \cal X } ) ; \eta = ( \beta ^ { \prime } , \mathrm { v e c } )
$$

$$
\psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) = U ( D - \mu _ { 0 } X ) ; \mu _ { 0 } = \mathrm { E } _ { P } [ D X ^ { \prime } ] ( \mathrm { E } _ { P } [ X X ^ { \prime } ] ) ^ { - 1 } = \gamma _ { 0 } ^ { \prime } .
$$

If the vector of covariates $X$ here is high-dimensional but the vectors of parameters $\beta _ { 0 }$

and $\gamma _ { 0 }$ are approximately sparse, we can use $\ell _ { 1 }$ -penalized least squares, $\ell _ { 2 }$ -boosting, or forward selection methods to estimate $\beta _ { 0 }$ and $\gamma _ { 0 } = \mu _ { 0 } ^ { \prime }$ , and hence $\mu _ { 0 } = ( \beta _ { 0 } ^ { \prime } , \mathrm { v e c } ( \mu _ { 0 } ) ^ { \prime } ) ^ { \prime }$ ; see references cited in the Introduction.

If $J _ { \beta \beta }$ is not invertible, equation (2.8) typically has multiple solutions. In this case, it is convenient to focus on a minimal norm solution,

$$
\mu _ { 0 } = \mathrm { a r g } \operatorname* { m i n } \| \mu \| \ \mathrm { s u c h } \operatorname { t h a t } \ \| J _ { \theta \beta } - \mu J _ { \beta \beta } \| _ { q } = 0
$$

for a suitably chosen norm · ∥q on the space of $d _ { \theta } \times d _ { \beta }$ matrices. With an eye on solving the empirical version of this problem, we may also consider the relaxed version of this problem,

$$
\mu _ { 0 } = \mathrm { a r g } \operatorname* { m i n } \| \mu \| \mathrm { s u c h } \operatorname { t h a t } \| J _ { \theta \beta } - \mu J _ { \beta \beta } \| _ { q } \leqslant r _ { N }
$$

for some $r _ { N } > 0$ such that $r _ { N }  0$ as $N  \infty$ . This relaxation is also helpful when $J _ { \beta \beta }$ is invertible but ill-conditioned. The following lemma shows that using $\mu _ { 0 }$ in $( 2 . 1 4 )$ leads to Neyman near-orthogonal scores. The proof of this lemma can be found in the Appendix.

Lemma 2.2. (Neyman Near-Orthogonal Scores for Quasi-Likelihood Settings) If (2.6) holds, $J$ exists, the solution of the optimization problem (2.14) exists, and $\mu _ { 0 }$ is taken to be this solution, the score $\psi$ defined in (2.7) is Neyman $\lambda _ { N }$ near-orthogonal at $( \theta _ { 0 } , \eta _ { 0 } )$ with respect to the nuisance realization set $\mathcal { T } _ { N } = \{ \beta \in \mathcal { B } \colon \| \beta - \beta _ { 0 } \| _ { q } ^ { * } \leqslant \lambda _ { N } / r _ { N } \} \times$ $\mathbb { R } ^ { d _ { \theta } d _ { \beta } }$ , where the norm $\| \cdot \| _ { q } ^ { * }$ on $\mathbb { R } ^ { d _ { \beta } }$ is defined by $\| \beta \| _ { q } ^ { * } = \operatorname* { s u p } _ { A } \| A \beta \|$ with the supremum being taken over all $d _ { \theta } \times d _ { \beta }$ matrices $A$ such that $\| A \| _ { q } \leqslant 1$ .

Example 2.1. (High-Dimensional Linear Regression, Continued) In the highdimensional linear regression example above, the relaxation $( 2 . 1 4 )$ is helpful when $J _ { \beta \beta } =$ $\mathrm { E } _ { P } [ X X ^ { \prime } ]$ is ill-conditioned. Specifically, if one suspects that $\mathrm { E } _ { P } [ X X ^ { \prime } ]$ is ill-conditioned, one can define $\mu _ { 0 }$ as the solution to the following optimization problem:

$$
\operatorname* { m i n } \| \mu \| \mathrm { s u c h \ t h a t \ } \| \mathrm { E } _ { P } [ D X ^ { \prime } ] - \mu \mathrm { E } _ { P } [ X X ^ { \prime } ] \| _ { \infty } \leqslant r _ { N } .
$$

Lemma 2.2 above then shows that using this $\mu _ { 0 }$ leads to a score $\psi$ that obeys the Neyman near-orthogonality condition. Alternatively, one can define $\mu _ { 0 }$ as the solution of the following closely related optimization problem,

$$
\operatorname* { m i n } _ { \mu } \Big ( \mu \mathrm { E } _ { P } [ X X ^ { \prime } ] \mu ^ { \prime } - \mu \mathrm { E } _ { P } [ D X ] + r _ { N } \| \mu \| _ { 1 } \Big ) ,
$$

whose solution also obeys $\| \mathrm { E } _ { P } [ D X ] - \mu \mathrm { E } _ { P } [ X X ^ { \prime } ] \| _ { \infty } \leqslant r _ { N }$ which follows from the first order conditions. An empirical version of either problem leads to a Lasso-type estimator of the regularized solution $\mu _ { 0 }$ ; see Javanmard and Montanari (2014a).

Remark 2.3. (Giving up Efficiency) Note that the regularized $\mu _ { 0 }$ in (2.14) creates the necessary near-orthogonality at the cost of giving up somewhat on efficiency of the score $\psi$ . At the same time, regularization may generate additional robustness gains since achieving full efficiency by estimating $\mu _ { 0 }$ in (2.10) may require stronger conditions.

Remark 2.4. (Concentrating-out Approach) The approach for constructing Neyman orthogonal scores described above is closely related to the following concentrating-out approach which has been used, for example, in Newey (1994), to show Neyman orthogonality when $\beta$ is infinite dimensional. For all $\theta \in \Theta$ , let $\beta _ { \theta }$ be the solution of the following

optimization problem:

$$
\operatorname* { m a x } _ { \beta \in \mathcal { B } } \mathrm { E } _ { P } [ \ell ( W ; \theta , \beta ) ] .
$$

Under mild regularity conditions, $\beta _ { \theta }$ satisfies

$$
\partial _ { \beta } \mathrm { E } _ { P } [ \ell ( W ; \theta , \beta _ { \theta } ) ] = 0 , \quad \mathrm { f o r ~ a l l ~ } \theta \in \Theta .
$$

Differentiating (2.16) with respect to $\theta$ and interchanging the order of differentiation gives

$$
\begin{array} { r l } & { 0 = \partial _ { \theta } \partial _ { \beta } \mathrm { E } _ { P } \Big [ \ell ( W ; \theta , \beta _ { \theta } ) \Big ] = \partial _ { \beta } \partial _ { \theta } \mathrm { E } _ { P } \Big [ \ell ( W ; \theta , \beta _ { \theta } ) \Big ] } \\ & { \quad = \partial _ { \beta } \mathrm { E } _ { P } \Big [ \partial _ { \theta } \ell ( W ; \theta , \beta _ { \theta } ) + \big [ \partial _ { \theta } \beta _ { \theta } \big ] ^ { \prime } \partial _ { \beta } \ell ( W ; \theta , \beta _ { \theta } ) \Big ] } \\ & { \quad = \partial _ { \beta } \mathrm { E } _ { P } \Big [ \psi ( W ; \theta , \beta , \partial _ { \theta } \beta _ { \theta } ) \Big ] \Big | _ { \beta = \beta _ { \theta } } , } \end{array}
$$

where we denoted

$$
\psi ( W ; \theta , \beta , \partial _ { \theta } \beta _ { \theta } ) : = \partial _ { \theta } \ell ( W ; \theta , \beta ) + [ \partial _ { \theta } \beta _ { \theta } ] ^ { \prime } \partial _ { \beta } \ell ( W ; \theta , \beta ) .
$$

This vector of functions is a score with nuisance parameters $\eta = ( \beta ^ { \prime } , \mathrm { v e c } ( \partial _ { \theta } \beta _ { \theta } ) ) ^ { \prime }$ . As before, additional nuisance parameters, $\partial _ { \theta } \beta _ { \theta }$ in this case, are introduced when the orthogonal score is formed. Evaluating these equations at $\theta _ { 0 }$ and $\beta _ { 0 }$ , it follows from the previous equation that $\psi ( W ; \theta , \beta , \partial _ { \theta } \beta _ { \theta } )$ is orthogonal with respect to $\beta$ and from $\mathrm { E } _ { P } [ \partial _ { \beta } \ell ( W ; \theta _ { 0 } , \beta _ { 0 } ) ] = 0$ that we have orthogonality with respect to $\partial _ { \theta } \beta _ { \theta }$ . Thus, maximizing the expected objective function with respect to the nuisance parameters, plugging that maximum back in, and differentiating with respect to the parameters of interest produces an orthogonal moment condition. See also Section 2.2.3.

# 2.2.2. Neyman Orthogonal Scores in GMM Problems

The construction in the previous section gives a Neyman orthogonal score whenever the moment conditions (2.6) hold, and, as discussed in Remark 2.2, the resulting score is efficient as long as $\ell ( W ; \theta , \beta )$ is the log-likelihood function. The question, however, remains about constructing the efficient score when $\ell ( W ; \theta , \beta )$ is not necessarily a loglikelihood function. In this section, we answer this question and describe a GMM-based method of constructing an efficient and Neyman orthogonal score in this more general case. The discussion here is related to Lee (2005), Bera et al. (2010), and Chernozhukov et al. (2015b).

Since GMM does not require that the equations (2.6) are obtained from the first-order conditions of the optimization problem (2.5), we use a different notation for the moment conditions. Specifically, we consider parameters $\theta \in \Theta \subset \mathbb { R } ^ { d _ { \theta } }$ and $\beta \in B \subset \mathbb { R } ^ { d _ { \beta } }$ , where $\boldsymbol { B }$ is a convex set, whose true values, $\theta _ { 0 }$ and $\beta _ { 0 }$ , solve the moment conditions

$$
\begin{array} { r } { \mathrm { E } _ { P } [ m ( W ; \theta _ { 0 } , \beta _ { 0 } ) ] = 0 , } \end{array}
$$

where $m \colon \mathcal { W } \times \Theta \times \mathcal { B }  \mathbb { R } ^ { d _ { m } }$ is a known vector-valued function, and $d _ { m } \geqslant d _ { \theta } + d _ { \beta }$ is the number of moment conditions. In this case, a Neyman orthogonal score function is

$$
\psi ( W ; \theta , \eta ) = \mu m ( W ; \theta , \beta ) ,
$$

where the nuisance parameter is

$$
\eta = ( \beta ^ { \prime } , \mathrm { v e c } ( \mu ) ^ { \prime } ) ^ { \prime } \in T = \mathcal { B } \times \mathbb { R } ^ { d _ { \theta } d _ { m } } \subset \mathbb { R } ^ { p } , \quad p = d _ { \beta } + d _ { \theta } d _ { m } ,
$$

and $\mu$ is the $d _ { \theta } \times d _ { m }$ orthogonalization parameter matrix whose true value is

$$
\mu _ { 0 } = \Big ( A ^ { \prime } \Omega ^ { - 1 } - A ^ { \prime } \Omega ^ { - 1 } G _ { \beta } ( G _ { \beta } ^ { \prime } \Omega ^ { - 1 } G _ { \beta } ) ^ { - 1 } G _ { \beta } ^ { \prime } \Omega ^ { - 1 } \Big ) ,
$$

where

$$
\begin{array} { r l } & { G _ { \gamma } = \partial _ { \gamma ^ { \prime } } \mathrm { E } _ { P } [ m ( W ; \theta , \beta ) ] \Big | _ { \gamma = \gamma _ { 0 } } } \\ & { \qquad = \Big [ \partial _ { \theta ^ { \prime } } \mathrm { E } _ { P } [ m ( W ; \theta , \beta ) ] , \partial _ { \beta ^ { \prime } } \mathrm { E } _ { P } [ m ( W ; \theta , \beta ) ] \Big ] \Big | _ { \gamma = \gamma _ { 0 } } = : \Big [ G _ { \theta } , G _ { \beta } \Big ] , } \end{array}
$$

for $\gamma = ( \theta ^ { \prime } , \beta ^ { \prime } ) ^ { \prime }$ and $\gamma _ { 0 } = ( \theta _ { 0 } ^ { \prime } , \beta _ { 0 } ^ { \prime } ) ^ { \prime }$ , $A$ is a $d _ { m } \times d _ { \theta }$ moment selection matrix, $\Omega$ is a $d _ { m } \times d _ { m }$ positive definite weighting matrix, and both $A$ and $\Omega$ can be chosen arbitrarily. Note that setting

$$
A = G _ { \theta } { \mathrm { ~ a n d ~ } } \Omega = \mathrm { V a r } _ { P } ( m ( W ; \theta _ { 0 } , \beta _ { 0 } ) ] ) = \operatorname { E } _ { P } \left[ m ( W ; \theta _ { 0 } , \beta _ { 0 } ) m ( W ; \theta _ { 0 } , \beta _ { 0 } ) ^ { \prime } \right]
$$

leads to the efficient score in the sense of yielding an estimator of $\theta _ { 0 }$ having the smallest variance in the class of GMM estimators (Hansen, 1982), and, in fact, to the semiparametrically efficient score; see Levit (1975), Nevelson (1977), and Chamberlain (1987). Let $\eta _ { 0 } = ( \beta _ { 0 } ^ { \prime } , \mathrm { v e c } ( \mu _ { 0 } ) ^ { \prime } ) ^ { \prime }$ be the true value of the nuisance parameter $\eta = ( \beta ^ { \prime } , \mathrm { v e c } ( \mu ) ^ { \prime } ) ^ { \prime }$ . The following lemma shows that the score $\psi$ in (2.18) satisfies the Neyman orthogonality condition.

Lemma 2.3. (Neyman Orthogonal Scores for GMM Settings) If (2.17) holds, $G _ { \gamma }$ exists, and $\Omega$ is invertible, the score $\psi$ in (2.18) is Neyman orthogonal at $( \theta _ { 0 } , \eta _ { 0 } )$ with respect to the nuisance realization set $\mathcal { T } _ { N } = \mathcal { T }$ .

As in the quasi-likelihood case, we can also consider near-orthogonal scores. Specifically, note that one of the orthogonality conditions that the score $\psi$ in (2.18) has to satisfy is that $\mu _ { 0 } G _ { \beta } = 0$ , which can be rewritten as

$$
A ^ { \prime } \Omega ^ { - 1 / 2 } ( I - L ( L ^ { \prime } L ) ^ { - 1 } L ^ { \prime } ) L = 0 , \mathrm { w h e r e } L = \Omega ^ { - 1 / 2 } G _ { \beta }
$$

Here, the part $A ^ { \prime } \Omega ^ { - 1 / 2 } L ( L ^ { \prime } L ) ^ { - 1 } L ^ { \prime }$ can be expressed as $\gamma _ { 0 } L ^ { \prime }$ , where $\gamma _ { 0 } = A ^ { \prime } \Omega ^ { - 1 / 2 } L ( L ^ { \prime } L ) ^ { - 1 }$ solves the optimization problem

$$
\operatorname* { m i n } \| \gamma \| _ { o } \ \mathrm { s u c h \ t h a t \ } \| A ^ { \prime } \Omega ^ { - 1 / 2 } L - \gamma L ^ { \prime } L \| _ { \infty } = 0 _ { : }
$$

for a suitably chosen norm $\| \cdot \| _ { o }$ . When $L ^ { \prime } L$ is close to being singular, this problem can be relaxed:

$$
\operatorname* { m i n } \| \gamma \| _ { o } \ \mathrm { s u c h \ t h a t \ } \| A ^ { \prime } \Omega ^ { - 1 / 2 } L - \gamma L ^ { \prime } L \| _ { \infty } \leqslant r _ { N } .
$$

This relaxation leads to Neyman near-orthogonal scores:

Lemma 2.4. (Neyman Near-Orthogonal Scores for GMM settings) In the set-up above, with $\gamma _ { 0 }$ denoting the solution of (2.19), we have for $\mu _ { 0 } : = A ^ { \prime } \Omega ^ { - 1 } - \gamma _ { 0 } L ^ { \prime } \Omega ^ { - 1 / 2 }$ and $\eta _ { 0 } ~ = ~ ( \beta _ { 0 } ^ { \prime } , v e c ( \mu _ { 0 } ) ^ { \prime } ) ^ { \prime }$ that $\psi$ defined in (2.18) is the Neyman $\lambda _ { N }$ near-orthogonal score at $( \theta _ { 0 } , \eta _ { 0 } )$ with respect to the nuisance realization set $\mathcal { T } _ { N } = \{ \beta \in \boldsymbol { B } \colon \| \beta - \beta _ { 0 } \| _ { 1 } \leqslant$ $\lambda _ { N } / r _ { N } \} \times \mathbb { R } ^ { d _ { \theta } d _ { m } }$ Rdθdm.

2.2.3. Neyman Orthogonal Scores for Likelihood and Other M-Estimation Problems with Infinite-Dimensional Nuisance Parameters

Here we show that the concentrating-out approach described in Remark 2.4 for the case of finite-dimensional nuisance parameters can be extended to the case of infinitedimensional nuisance parameters. Let $\ell ( W ; \theta , \beta )$ be a known criterion function, where $\theta$ and $\beta$ are the target and the nuisance parameters taking values in $\Theta$ and $\boldsymbol { B }$ , respectively and assume that the true values of these parameters, $\theta _ { 0 }$ and $\beta _ { 0 }$ , solve the optimization problem (2.5). The function $\ell ( W ; \theta , \beta )$ is analogous to that discussed above but now, instead of assuming that $\boldsymbol { B }$ is a (convex) subset of a finite-dimensional space, we assume that $\boldsymbol { B }$ is some (convex) set of functions, so that $\beta$ is the functional nuisance parameter. For example, $\ell ( W ; \theta , \beta )$ could be a semiparametric log-likelihood where $\beta$ is the nonparametric part of the model. More generally, $\ell ( W ; \theta , \beta )$ could be some other criterion function such as the negative of a squared residual. Also let

$$
\beta _ { \theta } = \arg \operatorname* { m a x } _ { \beta \in \mathcal { B } } \operatorname { E } _ { P } [ \ell ( W ; \theta , \beta ) ]
$$

be the “concentrated-out” nonparametric part of the model. Note that $\beta _ { \theta }$ is a functionvalued function. Now consider the score function

$$
\psi ( W ; \theta , \eta ) = \frac { d \ell ( W ; \theta , \eta ( \theta ) ) } { d \theta } ,
$$

where the nuisance parameter is $\eta \colon \Theta \to B$ , and its true value $\eta _ { 0 }$ is given by

$$
\eta _ { 0 } ( \theta ) = \beta _ { \theta } , \quad { \mathrm { f o r ~ a l l ~ } } \theta \in \Theta .
$$

Here, the symbol $d / d \theta$ denotes the full derivative with respect to $\theta$ , so that we differentiate with respect to both $\theta$ arguments in $\ell ( W ; \theta , \eta ( \theta ) )$ . The following lemma shows that the score $\psi$ in (2.21) satisfies the Neyman orthogonality condition.

Lemma 2.5. (Neyman Orthogonal Scores via Concentrating-Out Approach) Suppose that (2.5) holds, and let $T$ be a convex set of functions mapping $\Theta$ into $\boldsymbol { B }$ such that $\eta _ { 0 } ~ \in ~ T$ . Also, suppose that for each $\eta \in T$ , the function $\theta \mapsto \ell ( W ; \theta , \eta ( \theta ) )$ is continuously differentiable almost surely. Then, under mild regularity conditions, the score $\psi$ in (2.21) is Neyman orthogonal at $( \theta _ { 0 } , \eta _ { 0 } )$ with respect to the nuisance realization set $\mathcal { T } _ { N } = T$ .

As an example, consider the partially linear model from the Introduction. Let

$$
\ell ( W ; \theta , \beta ) = - \frac { 1 } { 2 } ( Y - D \theta - \beta ( X ) ) ^ { 2 } ,
$$

and let $\boldsymbol { B }$ be the set of functions of $X$ with finite mean square. Then

$$
( \theta _ { 0 } , \beta _ { 0 } ) = \arg \operatorname* { m a x } _ { \theta \in \Theta , \beta \in B } \operatorname { E } _ { P } [ \ell ( W ; \theta , \beta ) ]
$$

and

$$
\beta _ { \theta } ( X ) = \mathrm { E } _ { P } [ Y - D \theta | X ] , \quad \theta \in \Theta .
$$

Hence, (2.21) gives the following Neyman orthogonal score:

$$
\begin{array} { l } { \displaystyle \psi ( W ; \theta , \beta _ { \theta } ) = - \frac 1 2 \frac { d \{ Y - D \theta - \mathrm { E } _ { P } [ Y - D \theta | X ] \} ^ { 2 } } { d \theta } } \\ { \displaystyle = ( D - \mathrm { E } _ { P } [ D | X ] ) \times ( Y - \mathrm { E } _ { P } [ Y | X ] - ( D - \mathrm { E } _ { P } [ D | X ] ) \theta ) } \\ { \displaystyle = ( D - m _ { 0 } ( X ) ) \times ( Y - D \theta - g _ { 0 } ( X ) ) , } \end{array}
$$

which corresponds to the estimator $\theta _ { 0 }$ described in the Introduction in (1.5).

It is important to note that the concentrating-out approach described here gives a Neyman orthogonal score without requiring that $\ell ( W ; \theta , \beta )$ is the log-likelihood function. Except for the technical conditions needed to ensure the existence of derivatives and their interchangeability, the only condition that is required is that $\theta _ { 0 }$ and $\beta _ { 0 }$ solve the optimization problem (2.5). If $\ell ( W ; \theta , \beta )$ is the log-likelihood function, however, it follows from Newey (1994), p. 1359, that the concentrating-out approach actually yields the efficient score. An alternative, but closely related, approach to derive the efficient score in the likelihood setting would be to apply Neyman’s construction described above for a one-dimensional least favorable parametric sub-model; see Severini and Wong (1992) and Chap. 25 of van der Vaart (1998).

Remark 2.5. (Generating Orthogonal Scores by Varying $\boldsymbol { B }$ ) When we calculate the “concentrated-out” nonparametric part $\beta _ { \theta }$ , we can use some other set of functions $\Upsilon$ instead of $\boldsymbol { B }$ on the right-hand side of (2.20):

$$
\beta _ { \theta } = \arg \operatorname* { m a x } _ { \beta \in \Upsilon } \operatorname { E } _ { P } [ \ell ( W ; \theta , \beta ) ] .
$$

By replacing $\boldsymbol { B }$ by $\Upsilon$ we can generate a different Neyman orthogonal score. Of course, this replacement may also change the true value $\theta _ { 0 }$ of the parameter of interest, which is an important consideration for the selection of $\Upsilon$ . For example, consider the partially linear model and assume that $X$ has two components, $X _ { 1 }$ and $X _ { 2 }$ . Now, consider what would happen if we replaced $\boldsymbol { B }$ , which is the set of functions of $X$ with finite mean square, by the set of functions $\Upsilon$ that is the mean square closure of functions that are additive in $X _ { 1 }$ and $X _ { 2 }$ :

$$
\Upsilon = \overline { { \{ h ( X _ { 1 } ) + h ( X _ { 2 } ) \} } } .
$$

Let $\mathrm { E } _ { P }$ denote the least squares projection on $\Upsilon$ . Then, applying the previous calculation with $\mathrm { E } _ { P }$ replacing $\mathrm { E } _ { P }$ gives

$$
\psi ( W ; \theta , \beta _ { \theta } ) = ( D - \bar { \mathrm { E } } _ { P } [ D | X ] ) \times ( Y - \bar { \mathrm { E } } _ { P } [ Y | X ] + ( D - \bar { \mathrm { E } } _ { P } [ D | X ] ) \theta ) ,
$$

which provides an orthogonal score based on additive function of $X _ { 1 }$ and $X _ { 2 }$ . Here, it is important to note that the solution to $\mathrm { E } _ { P } [ \psi ( W , \theta , \beta _ { \theta } ) ] = 0$ will be the true $\theta _ { 0 }$ only when the true function of $X$ in the partially linear model is additive. More generally, the solution of the moment condition would be the coefficient of $D$ in the least squares projection of $Y$ on functions of the form $D \theta + h _ { 1 } ( X _ { 1 } ) + h _ { 1 } ( X _ { 2 } ) $ . Note though that the corresponding score is orthogonal by virtue of additivity being imposed in the estimation of $\bar { \mathrm { E } } _ { P } [ Y | X ]$ and $\bar { \mathrm { E } } _ { P } [ D | X ]$ .

2.2.4. Neyman Orthogonal Scores for Conditional Moment Restriction Problems with Infinite-Dimensional Nuisance Parameters

Next we consider the conditional moment restrictions framework studied in Chamberlain (1992). To define the framework, let $W$ , $R$ , and $Z$ be random vectors taking values in $\mathcal { W } \subset \mathbb { R } ^ { d _ { w } }$ , $\mathcal { R } \subset \mathbb { R } ^ { d _ { r } }$ , and $\mathcal { Z } \subset \mathbb { R } ^ { d _ { z } }$ , respectively. Assume that $Z$ is a sub-vector of $R$ and $R$ is a sub-vector of $W$ , so that $d _ { z } \leqslant d _ { r } \leqslant d _ { w }$ . Also, let $\theta \in \Theta \subset \mathbb { R } ^ { d _ { \theta } }$ be a finitedimensional parameter whose true value $\theta _ { 0 }$ is of interest, and let $h$ be a vector-valued functional nuisance parameter taking values in a convex set of functions $\mathcal { H }$ mapping $\mathcal { Z }$ to $\mathbb { R } ^ { d _ { h } }$ , with the true value of $h$ being $h _ { 0 }$ . The conditional moment restrictions framework assumes that $\theta _ { 0 }$ and $h _ { 0 }$ satisfy the moment conditions

$$
\mathrm { E } _ { P } [ m ( W ; \theta _ { 0 } , h _ { 0 } ( Z ) ) \mid R ] = 0 ,
$$

where $m \colon \mathcal { W } \times \Theta \times \mathbb { R } ^ { d _ { h } } \to \mathbb { R } ^ { d _ { m } }$ is a known vector-valued function. This framework is of interest because it covers a rich variety of models without having to explicitly rely on the likelihood formulation.

To build a Neyman orthogonal score $\psi ( W ; \theta , \eta )$ for estimating $\theta _ { 0 }$ , consider the matrixvalued functional parameter $\mu \colon \mathcal { R }  \mathbb { R } ^ { d _ { \theta } \times d _ { m } }$ whose true value is given by

$$
\mu _ { 0 } ( R ) = A ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } - G ( Z ) \Gamma ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } ,
$$

where the moment selection matrix-valued function $A \colon \mathcal { R }  \mathbb { R } ^ { d _ { m } \times d _ { \theta } }$ Rdm and the weighting positive definite matrix-valued function $\Omega \colon \mathcal { R }  \mathbb { R } ^ { d _ { m } \times d _ { m } }$ can be chosen arbitrarily, and the matrix-valued functions $\Gamma \colon \mathcal { R }  \mathbb { R } ^ { d _ { m } \times d _ { \theta } }$ and $G \colon \mathcal { Z } \to \mathbb { R } ^ { d _ { \theta } \times d _ { m } }$ are given by

$$
\begin{array} { r l } & { \Gamma ( R ) = \partial _ { v ^ { \prime } } \mathrm { E } _ { P } \Big [ m ( W ; \theta _ { 0 } , v ) \mid R \Big ] \Big | _ { v = h _ { 0 } ( Z ) } , \ \mathrm { a n d } } \\ & { G ( Z ) = \mathrm { E } _ { P } \Big [ A ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z \Big ] \times \Big ( \mathrm { E } _ { P } [ \Gamma ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] \Big ) ^ { - 1 } . } \end{array}
$$

Note that $\mu _ { 0 }$ in (2.23) is well-defined even though the right-hand side of (2.23) contains both $R$ and $Z$ since $Z$ is a sub-vector of $R$ . Then a Neyman orthogonal score is

$$
\psi ( W ; \theta , \eta ) = \mu ( R ) m ( W ; \theta , h ( Z ) ) ,
$$

where the nuisance parameter is

$$
\eta = ( \mu , h ) \in T = \mathcal { L } ^ { 1 } ( \mathcal { R } ; \mathbb { R } ^ { d _ { \theta } \times d _ { m } } ) \times \mathcal { H } .
$$

Here, $\mathcal { L } ^ { 1 } ( \mathcal { R } ; \ \mathbb { R } ^ { d _ { \theta } \times d _ { m } } )$ is the vector space of matrix-valued functions $f \colon \mathcal { R }  \mathbb { R } ^ { d _ { \theta } \times d _ { m } }$ satisfying $\mathrm { E } _ { P } [ \| f ( R ) \| ] < \infty$ . Also, note that even though the matrix-valued functions $A$ and $\Omega$ can be chosen arbitrarily, setting

$$
\begin{array} { r l } & { A ( R ) = \partial _ { \theta ^ { \prime } } \mathrm { E } _ { P } \Big [ m ( W ; \theta , h _ { 0 } ( Z ) ) \mid R \Big ] \Big | _ { \theta = \theta _ { 0 } } \mathrm { a n d } } \\ & { \Omega ( R ) = \mathrm { E } _ { P } \Big [ m ( W ; \theta _ { 0 } , h _ { 0 } ( Z ) ) m ( W ; \theta _ { 0 } , h _ { 0 } ( Z ) ) ^ { \prime } \mid R \Big ] } \end{array}
$$

leads to an asymptotic variance equal to the semiparametric bound of Chamberlain (1992). Let $\eta _ { 0 } = ( \mu _ { 0 } , h _ { 0 } )$ be the true value of the nuisance parameter $\boldsymbol { \eta } = \left( \boldsymbol { \mu } , \boldsymbol { h } \right)$ . The following lemma shows that the score $\psi$ in (2.26) satisfies the Neyman orthogonality condition.

Lemma 2.6. (Neyman Orthogonal Scores for Conditional Moment Settings) Suppose that (a) (2.22) holds, $( b )$ the matrices $\mathrm { E } _ { P } [ \| \Gamma ( R ) \| ^ { 4 } ]$ , $\operatorname { E } _ { P } [ \| G ( Z ) \| ^ { 4 } ]$ , $\mathrm { E } _ { P } [ \| A ( R ) \| ^ { 2 } ]$ , and $\mathrm { E } _ { P } [ \| \Omega ( R ) \| ^ { - 2 } ]$ are finite, and (c) for all $h \in \mathcal { H }$ , there exists a constant $C _ { h } > 0$ such that $\mathrm { P } _ { P } ( \mathrm { E } _ { P } [ \| m ( W ; \theta _ { 0 } , h ( Z ) ) \| ~ | ~ R ] \leqslant C _ { h } ) = 1$ . Then the score $\psi$ in (2.26) is Neyman orthogonal at $( \theta _ { 0 } , \eta _ { 0 } )$ with respect to the nuisance realization set $\mathcal { T } _ { N } = \mathcal { T }$ .

As an application of the conditional moment restrictions framework, let us derive Neyman orthogonal scores in the partially linear regression example using this framework. The partially linear regression model (1.1) is equivalent to

$$
\mathrm { E } _ { P } [ Y - D \theta _ { 0 } - g _ { 0 } ( X ) \mid X , D ] = 0 ,
$$

which can be written in the form of the conditional moment restrictions framework $\left( 2 . 2 2 \right)$ with $W = ( Y , D , X ^ { \prime } ) ^ { \prime }$ , $R = ( D , X ^ { \prime } ) ^ { \prime }$ , $Z = X$ , $h ( Z ) = g ( X )$ , and $m ( W ; \theta , v ) =$ $Y - D \theta - v$ . Hence, using (2.27) and (2.28) and denoting $\sigma ( D , X ) ^ { 2 } = \operatorname { E } _ { P } [ U ^ { 2 } \mid D , X ]$ for $U = Y - D \theta _ { 0 } - g _ { 0 } ( X )$ , we can take

$$
A ( R ) = - D , \Omega ( R ) = \operatorname { E } _ { P } [ U ^ { 2 } \mid D , X ] = \sigma ( D , X ) ^ { 2 } .
$$

With this choice of $A ( R )$ and $\Omega ( R )$ , we have

$$
\Gamma ( R ) = - 1 , G ( Z ) = { \left( \mathrm { E } _ { P } { \left[ \frac { D } { \sigma ( D , X ) ^ { 2 } } \mid X \right]  } \times { \left( \mathrm { E } _ { P } { \left[ \frac { 1 } { \sigma ( D , X ) ^ { 2 } } \mid X \right]  } ^ { - 1 } } , }\right)\right)
$$

and so (2.23) and (2.26) give

$$
\stackrel { \psi ( W ; \theta , \eta _ { 0 } ) } { = } \frac { 1 } { \sigma ( D , X ) ^ { 2 } } \Big ( D - \mathrm { E } _ { P } \Big [ \frac { D } { \sigma ( D , X ) ^ { 2 } } \mid X \Big ] \Big / \mathrm { E } _ { P } \Big [ \frac { 1 } { \sigma ( D , X ) ^ { 2 } } \mid X \Big ] \Big ) \times \Big ( Y - D \theta - g _ { 0 } ( X ) \Big ) .
$$

By construction, the score $\psi$ above is efficient and Neyman orthogonal. Note, however, that using this score would require estimating the heteroscedasticity function $\sigma ( D , X ) ^ { 2 }$ which would requires the imposition of some additional smoothness assumptions over this conditional variance function. Instead, if are willing to give up on efficiency to gain some robustness, we can take

$$
A ( R ) = - D , \quad \Omega ( R ) = 1 ;
$$

in which case we have

$$
\Gamma ( R ) = - 1 , G ( Z ) = \operatorname { E } _ { P } [ D \mid X ] .
$$

(2.23) and (2.26) then give

$$
\begin{array} { l } { { \psi ( W ; \theta , \eta _ { 0 } ) = ( D - \mathrm { E } _ { P } [ D \mid X ] ) \times ( Y - D \theta - g _ { 0 } ( X ) ) } } \\ { { { } } } \\ { { { } = ( D - m _ { 0 } ( X ) ) \times ( Y - D \theta - g _ { 0 } ( X ) ) . } } \end{array}
$$

This score $\psi$ is Neyman orthogonal and corresponds to the estimator of $\theta _ { 0 }$ described in the Introduction in (1.5). Note, however, that this score $\psi$ is efficient only f $\sigma ( X , D )$ is a constant.

# 2.2.5. Neyman Orthogonal Scores and Influence Functions

Neyman orthogonality is a joint property of the score $\psi ( W ; \theta , \eta )$ , the true parameter value $\eta _ { 0 }$ , the parameter set $T$ , and the distribution of $W$ . It is not determined by any particular model for the parameter $\theta$ . Nevertheless, it is possible to use semiparametric efficiency calculations to construct the orthogonal score from the original score as in Chernozhukov et al. (2016). Specifically, an orthogonal score can be constructed by adding to the original score the influence function adjustment for estimation of the nuisance functions that is analyzed in Newey (1994). The resulting orthogonal score will be the influence function of the limit of the average of the original score.

To explain, consider the original score $\varphi ( W ; \theta , \beta )$ , where $\beta$ is some function, and let ${ \widehat { \beta } } _ { 0 }$ be a nonparametric estimator of $\beta _ { 0 }$ , the true value of $\beta$ . Here, $\beta$ is implicitly allowed tbo depend on $\theta$ , though we suppress that dependence for notational convenience. The corresponding orthogonal score can be formed when there is $\phi ( W ; \theta , \eta )$ such that

$$
\int \varphi ( w ; \theta _ { 0 } , \widehat { \beta } _ { 0 } ) d P ( w ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \phi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) + o _ { P } { \left( n ^ { - 1 / 2 } \right) } ,
$$

where $\eta$ is a vector of nuisance functions that includes $\beta$ . $\phi ( W ; \theta , \eta )$ is an adjustment for the presence of the estimated function ${ \widehat { \beta } } _ { 0 }$ in the original score $\varphi ( W ; \theta , \beta )$ . The decomposition (2.29) typically holds when $\widehat { \beta }$ beither a kernel or a series estimator with a suitably chosen tuning parameter. The Nbeyman orthogonal score is given by

$$
\psi ( W ; \theta , \eta ) = \varphi ( W ; \theta , \beta ) + \phi ( W ; \theta , \eta ) .
$$

Here $\psi ( W ; \theta _ { 0 } , \eta _ { 0 } )$ is the influence function of the limit of $\begin{array} { r } { n ^ { - 1 } \sum _ { i = 1 } ^ { n } \varphi ( W _ { i } ; \theta _ { 0 } , \widehat { \beta } _ { 0 } ) } \end{array}$ , as analyzed in Newey (1994), with the restriction $\mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] = 0$ identifying $\theta _ { 0 }$ .

The form of the adjustment term $\phi ( W ; \theta , \eta )$ depends on the estimator ${ \widehat { \beta } } _ { 0 }$ and, of course, on the form of $\varphi ( W ; \theta , \beta )$ . Such adjustment terms have been derived fbor various ${ \widehat { \beta } } _ { 0 }$ by Newey (1994). Also Ichimura and Newey (2015) show how the adjustment term abn be computed from the limit of a certain derivative. Any of these results can be applied to a particular starting score $\varphi ( W ; \theta , \beta )$ and estimator ${ \widehat { \beta } } _ { 0 }$ to obtain an orthogonal score.

For example, consider again the partially linear bmodel with the original score

$$
\varphi ( W ; \theta , \beta ) = D ( Y - D \theta - g _ { 0 } ( X ) ) .
$$

Here ${ \widehat { \beta } } _ { 0 } = { \widehat { g } } _ { 0 }$ is a nonparametric regression estimator. From Newey (1994), we know that wbe ob abin the influence function adjustment by taking the conditional expectation of the derivative of the score with respect to $g _ { 0 } ( x )$ (obtaining $- m _ { 0 } ( X ) = - \mathrm { E } _ { P } [ D | X ] )$ and multiplying the result by the nonparametric residual to obtain

$$
\phi ( W , \theta , \eta ) = - m _ { 0 } ( X ) \{ Y - D \theta - \beta ( X , \theta ) \} .
$$

The corresponding orthogonal score is then simply

$$
\begin{array} { r l } & { \psi ( W ; \theta , \eta ) = \{ D - m _ { 0 } ( X ) \} \{ Y - D \theta - \beta ( X , \theta ) \} , } \\ & { \beta _ { 0 } ( X , \theta ) = \mathrm { E } _ { P } [ Y - D \theta | X ] , m _ { 0 } ( X ) = \mathrm { E } _ { P } [ D | X ] , } \end{array}
$$

illustrating that an orthogonal score for the partially linear model can be derived from an influence function adjustment.

Influence functions have been used to estimate functionals of nonparametric estimators by Hasminskii and Ibragimov (1978) and Bickel and Ritov (1988). Newey et al. (1998, 2004) showed that $n ^ { - 1 / 2 } \sum _ { i = 1 } ^ { n } \psi ( W _ { i } ; \theta _ { 0 } , \widehat { \eta _ { 0 } } )$ from equation (2.30) will have a second order remainder in $\widehat { \eta _ { 0 } }$ , which is the key asympt obtic property of orthogonal scores. Orthogonality of influence fbunctions in semiparametric models follows from van der Vaart (1991), as shown for higher order counterparts in Robins et al. (2008, 2017). Chernozhukov et al. (2016) point out that in general an orthogonal score can be constructed from an original score and nonparametric estimator ${ \widehat { \beta } } _ { 0 }$ by adding to the original score the adjustment term for estimation of $\beta _ { 0 }$ as described above. This construction provides a way of obtaining an orthogonal score from any initial score $\varphi ( W ; \theta , \beta )$ and nonparametric estimator ${ \widehat { \beta } } _ { 0 }$ .

# 3. DML: POST-REGULARIZED INFERENCE BASED ON NEYMAN-ORTHOGONAL ESTIMATING EQUATIONS

3.1. Definition of DML and Its Basic Properties

We assume that we have a sample $( W _ { i } ) _ { i = 1 } ^ { N }$ , modeled as i.i.d. copies of $W$ , whose law is determined by the probability measure $P$ on $\mathcal { W }$ . Estimation will be carried out using the finite-sample analog of the estimating equations (2.1).

We assume that the true value $\eta _ { 0 }$ of the nuisance parameter $\eta$ can be estimated by $\widehat { \eta _ { 0 } }$ using a part of the data $( W _ { i } ) _ { i = 1 } ^ { N }$ . Different structured assumptions on $\eta _ { 0 }$ allow us to usbe different machine-learning tools for estimating $\eta _ { 0 }$ . For instance,

1. approximate sparsity for $\eta _ { 0 }$ with respect to some dictionary calls for the use of forward selection, lasso, post-lasso, $\ell _ { 2 }$ -boosting, or some other sparsity-based technique;
2. well-approximability of $\eta _ { 0 }$ by trees calls for the use of regression trees and random forests;
3. well-approximability of $\eta _ { 0 }$ by sparse neural and deep neural nets calls for the use of $\ell _ { 1 }$ -penalized neural and deep neural networks;
4. well-approximability of $\eta _ { 0 }$ by at least one model mentioned in 1)-3) above calls for the use of an ensemble/aggregated method over the estimation methods mentioned in 1)-3).

There are performance guarantees for most of these ML methods that make it possible to satisfy the conditions stated below. Ensemble and aggregation methods ensure that the performance guarantee is approximately no worse than the performance of the best method.

We assume that $N$ is divisible by $K$ in order to simplify the notation. The following algorithm defines the simple cross-fitted DML as outlined in the Introduction.

Definition 3.1. (DML1) 1) Take a $K$ -fold random partition $( I _ { k } ) _ { k = 1 } ^ { K }$ of observation indices $[ N ] = \{ 1 , . . . , N \}$ such that the size of each fold $I _ { k }$ is $n = N / K$ . Also, for each $k \in [ K ] = \{ 1 , . . . , K \}$ , define $I _ { k } ^ { c } : = \{ 1 , . . . , N \} \setminus I _ { k } . ~ \mathcal { Q }$ ) For each $k \in [ K ]$ , construct a ML estimator

$$
\widehat { \eta } _ { 0 , k } = \widehat { \eta } _ { 0 } \big ( ( W _ { i } ) _ { i \in I _ { k } ^ { c } } \big )
$$

of $\eta _ { 0 }$ , where $\widehat { \eta } _ { 0 , k }$ is a random elembent in $T$ , and where randomness depends only on the subset of da ab indexed by $I _ { k } ^ { c }$ . 3) For each $k \in [ K ]$ , construct the estimator $\check { \theta } _ { 0 , k }$ as the solution of the following equation:

$$
\mathbb { E } _ { n , k } [ \psi ( W ; \check { \theta } _ { 0 , k } , \widehat { \eta } _ { 0 , k } ] = 0 ,
$$

where $\psi$ is the Neyman orthogonal score, and En,k is the empirical expectation over the $k$ -th fold of the data; that is, $\begin{array} { r } { \mathbb { E } _ { n , k } [ \psi ( W ) ] = n ^ { - 1 } \sum _ { i \in I _ { k } } \psi ( W _ { i } ) } \end{array}$ . If achievement of exact 0 is not possible, define the estimator $\check { \theta } _ { 0 , k }$ of $\theta _ { 0 }$ as an approximate $\epsilon N$ -solution:

$$
\Big \| \mathbb { E } _ { n , k } [ \psi ( W ; \check { \theta } _ { 0 , k } , \widehat { \eta } _ { 0 , k } ) ] \Big \| \leqslant \operatorname* { i n f } _ { \theta \in \Theta } \Big \| \mathbb { E } _ { n , k } [ \psi ( W ; \theta , \widehat { \eta } _ { 0 , k } ) ] \Big \| + \epsilon _ { N } , \quad \epsilon _ { N } = o ( \delta _ { N } N ^ { - 1 / 2 } ) ,
$$

where $\left( \delta _ { N } \right) _ { N \geqslant 1 }$ is some sequence of positive constants converging to zero. 4) Aggregate

the estimators:

$$
\tilde { \theta } _ { 0 } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \check { \theta } _ { 0 , k } .
$$

This approach generalizes the 50-50 cross-fitting method mentioned in the Introduction. We now define a variation of this basic cross-fitting approach that may behave better in small samples.

Definition 3.2. (DML2) 1) Take a $K$ -fold random partition $( I _ { k } ) _ { k = 1 } ^ { K }$ of observation indices $[ N ] = \{ 1 , . . . , N \}$ such that the size of each fold $I _ { k }$ is $n = N / K$ . Also, for each $k \in [ K ] = \{ 1 , \ldots , K \}$ , define $I _ { k } ^ { c } : = \{ 1 , . . . , N \} \setminus I _ { k } . ~ \mathcal { Q } \big )$ For each $k \in [ K ]$ , construct a ML estimator

$$
\widehat { \eta } _ { 0 , k } = \widehat { \eta } _ { 0 } \big ( ( W _ { i } ) _ { i \in I _ { k } ^ { c } } \big )
$$

of $\eta _ { 0 }$ , where $\widehat { \eta } _ { 0 , k }$ is a random elembent in $T$ , and where randomness depends only on the subset of datab indexed by $I _ { k } ^ { c }$ . 3) Construct the estimator $\tilde { \theta } _ { 0 }$ as the solution to the following equation:

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } [ \psi ( W ; \tilde { \theta } _ { 0 } , \widehat { \eta } _ { 0 , k } ) ] = 0 ,
$$

where $\psi$ is the Neyman orthogonal score, and En,k is the empirical expectation over the $k$ -th fold of the data; that is, $\begin{array} { r } { \mathbb { E } _ { n , k } [ \psi ( W ) ] = n ^ { - 1 } \sum _ { i \in I _ { k } } \psi ( W _ { i } ) } \end{array}$ . If achievement of exact $\boldsymbol { \mathit { 0 } }$ is not possible define the estimator $\tilde { \theta } _ { 0 }$ of $\theta _ { 0 }$ as an approximate $\epsilon _ { N }$ -solution:

$$
\Big \| \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } [ \psi ( W ; \widetilde { \theta } _ { 0 } , \widehat { \eta } _ { 0 , k } ) ] ] \Big \| \leqslant \operatorname* { i n f } _ { \theta \in \Theta } \Big \| \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } [ \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) ] ] \Big \| + \epsilon _ { N } ,
$$

for $\epsilon _ { N } = o ( \delta _ { N } N ^ { - 1 / 2 } )$ , where $\left( \delta _ { N } \right) _ { N \geqslant 1 }$ is some sequence of positive constants converging to zero.

Remark 3.1. (Recommendations) The choice of $K$ has no asymptotic impact under our conditions but, of course, the choice of K may matter in small samples. Intuitively, larger values of $K$ provide more observations in $I _ { k } ^ { c }$ from which to estimate the high-dimensional nuisance functions, which seems to be the more difficult part of the problem. We have found moderate values of $K$ , such as 4 or 5, to work better than $K = 2$ in a variety of empirical examples and in simulations. Moreover, we generally recommend DML2 over DML1 though in some problems like estimation of ATE in the interactive model, which we discuss later, there is no difference between the two approaches. In most other problems, DML2 is better behaved since the pooled empirical Jacobian for the equation in (3.4) exhibits more stable behavior than the separate empirical Jacobians for the equation in (3.1).

# 3.2. Moment Condition Models with Linear Scores

We first consider the case of linear scores, where

$$
\psi ( w ; \theta , \eta ) = \psi ^ { a } ( w ; \eta ) \theta + \psi ^ { b } ( w ; \eta ) , \quad \mathrm { f o r ~ a l l ~ } w \in \mathcal { W } , \ \theta \in \Theta , \ \eta \in T .
$$

Let $c _ { 0 } > 0$ , $c _ { 1 } > 0$ , $s > 0$ , $q > 2$ be some finite constants such that $c _ { 0 } \leqslant c _ { 1 }$ ; and let $\{ \delta _ { N } \} _ { N \geqslant 1 }$ and $\{ \Delta _ { N } \} _ { N \geqslant 1 }$ be some sequences of positive constants converging to zero such that $\delta _ { N } \geqslant N ^ { - 1 / 2 }$ . Also, let $\textit { K } \geqslant \ 2$ be some fixed integer, and let $\{ \mathcal P _ { N } \} _ { N \geqslant 1 }$ be some sequence of sets of probability distributions P of $W$ on $\mathcal { W }$ .

Assumption 3.1. (Linear Scores with Approximate Neyman Orthogonality) For all $N \geqslant$ 3 and $P \in \mathcal { P } _ { N }$ , the following conditions hold. (a) The true parameter value $\theta _ { 0 }$ obeys (2.1). (b) The score $\psi$ is linear in the sense of $( 3 . 6 )$ . (c) The map $\eta \mapsto \operatorname { E } _ { P } [ \psi ( W ; \theta , \eta ) ]$ is twice continuously Gateaux-differentiable on $T$ . (d) The score $\psi$ obeys the Neyman orthogonality or, more generally, the Neyman $\lambda _ { N }$ near-orthogonality condition at $( \theta _ { 0 } , \eta _ { 0 } )$ with respect to the nuisance realization set $\mathcal { T } _ { N } \subset T$ for

$$
\lambda _ { N } : = \operatorname* { s u p } _ { \eta \in \mathcal { T } _ { N } } \left\| \partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] \right\| \leqslant \delta _ { N } N ^ { - 1 / 2 } .
$$

(e) The identification condition holds; namely, the singular values of the matrix

$$
{ \cal J } _ { 0 } : = \mathrm { E } _ { P } [ \psi ^ { a } ( W ; \eta _ { 0 } ) ]
$$

are between $c _ { 0 }$ and c1.

Assumption 3.1 requires scores to be Neyman orthogonal or near-orthogonal and imposes mild smoothness requirements as well as the canonical identification condition.

Assumption 3.2. (Score Regularity and Quality of Nuisance Parameter Estimators) For all $\textit { N } \geqslant 3$ and $P \in \mathcal { P } _ { N }$ , the following conditions hold. (a) Given a random subset $I$ of $[ N ]$ of size $n = N / K$ , the nuisance parameter estimator $\widehat { \eta } _ { 0 } = \widehat { \eta } _ { 0 } ( ( W _ { i } ) _ { i \in I ^ { c } } )$ belongs to the realization set $\mathcal { T } _ { N }$ with probability at least $1 - \Delta _ { N }$ , wbhere $\mathcal { T } _ { N }$ contains $\eta _ { 0 }$ and is constrained by the next conditions. (b) The moment conditions hold:

$$
\begin{array} { r l } & { m _ { N } : = \displaystyle \operatorname* { s u p } _ { \eta \in \mathcal { T } _ { N } } ( \mathrm { E } _ { P } [ \| \psi ( W ; \theta _ { 0 } , \eta ) \| ^ { q } ] ) ^ { 1 / q } \leqslant c _ { 1 } , } \\ & { m _ { N } ^ { \prime } : = \displaystyle \operatorname* { s u p } _ { \eta \in \mathcal { T } _ { N } } ( \mathrm { E } _ { P } [ \| \psi ^ { a } ( W ; \eta ) \| ^ { q } ] ) ^ { 1 / q } \leqslant c _ { 1 } . } \end{array}
$$

(c) The following conditions on the statistical rates $r _ { N }$ , $r _ { N } ^ { \prime }$ , and $\lambda _ { N } ^ { \prime }$ hold:

$$
\begin{array} { r l } & { r _ { N } : = \underset { \eta \in \mathcal { T } _ { N } } { \operatorname* { s u p } } \left. \mathrm { E } _ { P } [ \psi ^ { a } ( W ; \eta ) ] - \mathrm { E } _ { P } [ \psi ^ { a } ( W ; \eta _ { 0 } ) ] \right. \leqslant \delta _ { N } , } \\ & { r _ { N } ^ { \prime } : = \underset { \eta \in \mathcal { T } _ { N } } { \operatorname* { s u p } } \left( \mathrm { E } _ { P } [ \| \psi ( W ; \theta _ { 0 } , \eta ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } ] \right) ^ { 1 / 2 } \leqslant \delta _ { N } , } \\ & { \lambda _ { N } ^ { \prime } : = \underset { r \in ( 0 , 1 ) , \eta \in \mathcal { T } _ { N } } { \operatorname* { s u p } } \left. \partial _ { r } ^ { 2 } \mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) ) ] \right. \leqslant \delta _ { N } / \sqrt { N } . } \end{array}
$$

(d) The variance of the score $\psi$ is non-degenerate: All eigenvalues of the matrix

$$
\operatorname { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ^ { \prime } ]
$$

are bounded from below by $c _ { 0 }$ .

Assumptions 3.2(a)-(c) state that the estimator of the nuisance parameter belongs to the realization set $\mathcal { T } _ { N } \subset T$ , which is a shrinking neighborhood of $\eta _ { 0 }$ , which contracts around $\boldsymbol { \eta } _ { 0 }$ with the rate determined by the “statistical” rates $r _ { N }$ , $r _ { N } ^ { \prime }$ , and $\lambda _ { N } ^ { \prime }$ . These rates are not given in terms of the norm $\| \cdot \| _ { T }$ on $T$ , but rather are the intrinsic rates that are most connected to the statistical problem at hand. However, in smooth problems, as discussed below this translates, in the worst cases, to the crude requirement that the nuisance parameters are estimated at the rate $o \big ( N ^ { - 1 / 4 } \big )$ −1/4).

The conditions in Assumption 3.2 embody refined requirements on the quality of nuisance parameter estimators. In many applications, where $( \theta , \eta ) \mapsto \psi ( W ; \theta , \eta )$ is smooth, we can bound

$$
r _ { N } \lesssim \varepsilon _ { N } , \quad r _ { N } ^ { \prime } \lesssim \varepsilon _ { N } , \quad \lambda _ { N } ^ { \prime } \lesssim \varepsilon _ { N } ^ { 2 } ,
$$

where $\varepsilon _ { N }$ is the upper bound on the rate of convergence of $\widehat { \eta _ { 0 } }$ to $\eta _ { 0 }$ with respect to the norm $\| \cdot \| _ { T } = \| \cdot \| _ { P , 2 }$ :

$$
\| \widehat { \eta } _ { 0 } - \eta \| _ { T } \lesssim \varepsilon _ { N } .
$$

Note that $\mathcal { T } _ { N }$ can be chosen as the bset of $\eta$ that is within a neighborhood of size εN of $\eta _ { 0 }$ , possibly with other restrictions, in this case. If only (3.7) holds, Assumption 3.2, particularly $\lambda _ { N } ^ { \prime } = o ( N ^ { - 1 / 2 } )$ , imposes the (crude) rate requirement

$$
\varepsilon _ { N } = o ( N ^ { - 1 / 4 } ) .
$$

This rate is achievable for many ML methods under structured assumptions on the nuisance parameters. See, among many others, Bickel et al. (2009), Bu¨hlmann and van de Geer (2011), Belloni et al. (2011), Belloni and Chernozhukov (2011), Belloni et al. (2012), and Belloni and Chernozhukov (2013) for $\ell _ { 1 }$ -penalized and related methods in a variety of sparse models; Kozbur (2016) for forward selection in sparse models; Luo and Spindler (2016) for $L _ { 2 }$ -boosting in sparse linear models; Wager and Walther (2016) for concentration results for a class of regression trees and random forests; and Chen and White (1999) for a class of neural nets.

However, the presented conditions allow for more refined statements than (3.8). We note that many important structured problems – such as estimation of parameters in partially linear regression models, estimation of parameters in partially linear structural equation models, and estimation of average treatment effects under unconfoundedness $\mathrm { ~ - ~ }$ are such that some cross-derivatives vanish, allowing more refined requirements than (3.8). This feature allows us to require much finer conditions on the quality of the nuisance parameter estimators than the crude bound (3.8). For example, in many problems

$$
\lambda _ { N } ^ { \prime } = 0 ,
$$

because the second derivatives vanish,

$$
\partial _ { r } ^ { 2 } \mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) ) ] = 0 .
$$

This occurs in the following important examples:

1. the optimal instrument problem; see Belloni et al. (2012).
2. the partially linear regression model when $m _ { 0 } ( X ) = 0$ or is otherwise known; see Section 4.
3. the treatment effect examples when the propensity score is known, which includes randomized control trials as an important special case; see Section 5.

If both (3.7) and (3.9) hold, Assumption 3.2, particularly $r _ { N } = o ( 1 )$ and $r _ { N } ^ { \prime } = o ( 1 )$ , imposes the weakest possible rate requirement:

$$
\varepsilon _ { N } = o ( 1 ) .
$$

We note that similar refined rates have appeared in the context of estimation of treatment effects in high-dimensional settings under sparsity; see Farrell (2015) and Athey et al. (2016) and related discussion in Remark 5.2. Our refined rate results complement this work by applying to a broad class of estimation contexts, including estimation of average treatment effects, and to a broad set of ML estimators.

Theorem 3.1. (Properties of the DML) Suppose that Assumptions 3.1 and 3.2 hold. In addition, suppose that $\delta _ { N } \ \geqslant \ N ^ { - 1 / 2 }$ for all $\textit { N } \geqslant \ 1$ . Then the DML1 and DML2 estimators $\tilde { \theta } _ { 0 }$ concentrate in a $1 / \sqrt { N }$ neighborhood of $\theta _ { 0 }$ and are approximately linear and centered Gaussian:

$$
\sqrt { N } \sigma ^ { - 1 } ( \tilde { \theta } _ { 0 } - \theta _ { 0 } ) = \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \bar { \psi } ( W _ { i } ) + O _ { P } ( \rho _ { N } ) \sim N ( 0 , \mathrm { I } _ { d } ) ,
$$

uniformly over $P \in \mathcal P _ { N }$ , where the size of the remainder term obeys

$$
\rho _ { N } : = N ^ { - 1 / 2 } + r _ { N } + r _ { N } ^ { \prime } + N ^ { 1 / 2 } \lambda _ { N } + N ^ { 1 / 2 } \lambda _ { N } ^ { \prime } \lesssim \delta _ { N } ,
$$

$\bar { \psi } ( \cdot ) : = - \sigma ^ { - 1 } J _ { 0 } ^ { - 1 } \psi ( \cdot , \theta _ { 0 } , \eta _ { 0 } )$ is the influence function, and the approximate variance is

$$
\begin{array} { r } { \sigma ^ { 2 } : = J _ { 0 } ^ { - 1 } \mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ^ { \prime } ] ( J _ { 0 } ^ { - 1 } ) ^ { \prime } . } \end{array}
$$

The result establishes that the estimator based on the orthogonal scores achieves the root- $N$ rate of convergence and is approximately normally distributed. It is noteworthy that this convergence result, both the rate of concentration and the distributional approximation, holds uniformly with respect to $P$ varying over an expanding class of probability measures $\mathcal { P } _ { N }$ . This means that the convergence holds under any sequence of probability distributions $( P _ { N } ) _ { N \geqslant 1 }$ with $P _ { N } \in \mathcal { P } _ { N }$ for each $N$ , which in turn implies that the results are robust with respect to perturbations of a given $P$ along such sequences. The same property can be shown to fail for methods not based on orthogonal scores.

Theorem 3.2. (Variance Estimator for DML) Suppose that Assumptions 3.1 and 3.2 hold. In addition, suppose that $\delta _ { N } \geqslant N ^ { - [ ( 1 - 2 / q ) \wedge 1 / 2 ] }$ for all $\textit { N } \geqslant 1$ . Consider the following estimator of the asymptotic variance matrix of $\sqrt { N } ( \tilde { \theta } _ { 0 } - \theta _ { 0 } )$ :

$$
\widehat { \boldsymbol { \sigma } } ^ { 2 } = \widehat { J } _ { 0 } ^ { - 1 } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } [ \psi ( \boldsymbol { W } ; \widetilde { \boldsymbol { \theta } } _ { 0 } , \widehat { \eta } _ { 0 , k } ) \psi ( \boldsymbol { W } ; \widetilde { \boldsymbol { \theta } } _ { 0 } , \widehat { \eta } _ { 0 , k } ) ^ { \prime } ] ( \widehat { J } _ { 0 } ^ { - 1 } ) ^ { \prime } ,
$$

where

$$
\widehat { J } _ { 0 } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } [ \psi ^ { a } ( W ; \widehat { \eta } _ { 0 , k } ) ] ,
$$

and $\tilde { \theta } _ { 0 }$ is either the DML1 or the DML2 estimator. This estimator concentrates around the true variance matrix $\sigma ^ { 2 }$ ,

$$
\widehat { \sigma } ^ { 2 } = \sigma ^ { 2 } + O _ { P } ( \varrho _ { N } ) , \quad \varrho _ { N } : = N ^ { - [ ( 1 - 2 / q ) \wedge 1 / 2 ] } + r _ { N } + r _ { N } ^ { \prime } \lesssim \delta _ { N } .
$$

Moreover, ${ \widehat { \sigma } } ^ { 2 }$ can replace $\sigma ^ { 2 }$ in the statement of Theorem 3.1 with the size of the remainder term upbdated as $\rho _ { N } = N ^ { - [ ( 1 - 2 / q ) \wedge 1 / 2 ] } + r _ { N } + r _ { N } ^ { \prime } + N ^ { 1 / 2 } \lambda _ { N } + N ^ { 1 / 2 } \lambda _ { N } ^ { \prime }$ .

Theorems 3.1 and 3.2 can be used for standard construction of confidence regions which are uniformly valid over a large, interesting class of models:

Corollary 3.1. (Uniformly Valid Confidence Bands) Under the conditions of Theorem 3.2, suppose we are interested in the scalar parameter $\ell ^ { \prime } \theta _ { 0 }$ for some $d _ { \theta } \times 1$ vector ℓ. Then the confidence interval

$$
\mathrm { C I } : = \Big [ \ell ^ { \prime } \widetilde { \theta } _ { 0 } \pm \Phi ^ { - 1 } ( 1 - \alpha / 2 ) \sqrt { \ell ^ { \prime } \widehat { \sigma } ^ { 2 } \ell / N } \Big ]
$$

obeys

$$
\operatorname* { s u p } _ { P \in \mathcal { P } _ { N } }  \mathrm { P } _ { P } ( \ell ^ { \prime } \theta _ { 0 } \in \mathrm { C I } ) - ( 1 - \alpha )   0 .
$$

Indeed, the above theorem implies that CI obeys $\mathrm { P } _ { P _ { N } } ( \ell ^ { \prime } \theta _ { 0 } \in \mathrm { C I } ) \to ( 1 - \alpha )$ under any sequence $\{ P _ { N } \} \in { \mathcal { P } } _ { N }$ , which implies that these claims hold uniformly in $P \in \mathcal { P } _ { N }$ . For example, one may choose $\{ P _ { N } \}$ such that, for some $\epsilon _ { N }  0$

$$
\operatorname* { s u p } _ { P \in \mathcal { P } _ { N } } | \mathrm { P } _ { P } ( \ell ^ { \prime } \theta _ { 0 } \in \mathrm { C I } ) - ( 1 - \alpha ) | \leqslant | \mathrm { P } _ { P _ { N } } ( \ell ^ { \prime } \theta _ { 0 } \in \mathrm { C I } ) - ( 1 - \alpha ) | + \epsilon _ { N } \to 0 .
$$

Next we note that the estimators need not be semi-parametrically efficient, but under some conditions they can be.

Corollary 3.2. (Cases with Semi-parametric Efficiency) Under the conditions of Theorem 3.1, if the score $\psi$ is efficient for estimating $\theta _ { 0 }$ at a given $P \in \mathcal P \subset \mathcal P _ { N }$ , in the semi-parametric sense as defined in van der Vaart (1998), then the large sample variance $\sigma _ { 0 } ^ { 2 }$ of $\tilde { \theta } _ { 0 }$ reaches the semi-parametric efficiency bound at this $P$ relative to the model $\mathcal { P }$ .

# 3.3. Models with Nonlinear Scores

Let $c _ { 0 } > 0$ , $c _ { 1 } > 0$ , $a > 1$ , $v > 0$ , $s > 0$ , and $q > 2$ be some finite constants, and let $\{ \delta _ { N } \} _ { N \geqslant 1 }$ , $\{ \Delta _ { N } \} _ { N \geqslant 1 }$ , and $\{ \tau _ { N } \} _ { N \geqslant 1 }$ be some sequences of positive constants converging to zero. To derive the properties of the DML estimator, we will use the following assumptions.

Assumption 3.3. (Nonlinear Moment Condition Problem with Approximate Neyman Orthogonality) For all $N \geqslant 3$ and $P \in \mathcal { P } _ { N }$ PN, the following conditions hold. (a) The true parameter value $\theta _ { 0 }$ obeys $( 2 . 1 )$ , and $\Theta$ contains a ball of radius $c _ { 1 } N ^ { - 1 / 2 } \log N$ centered at $\theta _ { 0 }$ . (b) The map $( \theta , \eta ) \mapsto \operatorname { E } _ { P } [ \psi ( W ; \theta , \eta ) ]$ is twice continuously Gateaux-differentiable on $\Theta \times T$ . (c) For all $\theta \in \Theta$ , the identification relation

$$
2 \lVert \mathrm { E } _ { P } [ \psi ( W ; \theta , \eta _ { 0 } ) ] \rVert \geqslant \lVert J _ { 0 } ( \theta - \theta _ { 0 } ) \rVert \wedge c _ { 0 }
$$

is satisfied, for the Jacobian matrix

$$
J _ { 0 } : = \left. \partial _ { \theta ^ { \prime } } \Big \{ \mathrm { E } _ { P } [ \psi ( W ; \theta , \eta _ { 0 } ) ] \Big \} \right| _ { \theta = \theta _ { 0 } }
$$

having singular values between $c _ { 0 }$ and $c _ { 1 }$ . (d) The score $\psi$ obeys the Neyman orthogonality or, more generally the Neyman near-orthogonality with $\lambda _ { N } = \delta _ { N } N ^ { - 1 / 2 }$ for the set $\mathcal { T } _ { N } \subset$ $T$ .

Assumption 3.3 is mild and rather standard in moment condition problems. Assumption 3.3(a) requires $\theta _ { 0 }$ to be sufficiently separated from the boundary of $\Theta$ . Assumption 3.3(b) only requires differentiability of the function $( \theta , \eta ) \mapsto \operatorname { E } _ { P } [ \psi ( W ; \theta , \eta ) ]$ and does not equire differentiability of the function $( \theta , \eta ) \mapsto \psi ( W ; \theta , \eta )$ . Assumption 3.3(c) implie ufficient identifiability of $\theta _ { 0 }$ . Assumption 3.3(d) is the orthogonality condition that ha already been extensively discussed.

Assumption 3.4. (Score Regularity and Requirements on the Quality of Estimation of Nuisance Parameters) Let K be a fixed integer. For all $N \geqslant 3$ and $P \in \mathcal P _ { N }$ , the following conditions hold. (a) Given a random subset $I$ of $\{ 1 , \ldots , N \}$ of size $n = N / K$ , we have that the nuisance parameter estimator $\widehat { \eta } _ { 0 } = \widehat { \eta } _ { 0 } ( ( W _ { i } ) _ { i \in I ^ { c } } )$ belongs to the realization set $\mathcal { T } _ { N }$ with probability at least $1 - \Delta _ { N }$ , whereb $\mathcal { T } _ { N }$ cbontains $\eta _ { 0 }$ and is constrained by conditions given below. (b) The parameter space $\Theta$ is bounded and for each $\eta \in \mathcal { T } _ { N }$ , the function class ${ \mathcal { F } } _ { 1 , \eta } ~ = ~ \{ \psi _ { j } ( \cdot , \theta , \eta ) \colon j ~ = ~ 1 , . . . , d _ { \theta } , \theta ~ \in ~ \Theta \}$ is suitably measurable and its uniform covering entropy obeys

$$
\operatorname* { s u p } _ { Q } \log N ( \epsilon \| F _ { 1 , \eta } \| _ { Q , 2 } , \mathcal { F } _ { 1 , \eta } , \| \cdot \| _ { Q , 2 } ) \leqslant v \log ( a / \epsilon ) , \quad f o r \ a l l \ 0 < \epsilon \leqslant 1 ,
$$

where F1,η is a measurable envelope for $\mathcal { F } _ { 1 , \eta }$ that satisfies $\| F _ { 1 , \eta } \| _ { P , q } \leqslant c _ { 1 }$ . (c) The following conditions on the statistical rates $r _ { N }$ , $r _ { N } ^ { \prime }$ , and $\lambda _ { N } ^ { \prime }$ hold:

$$
\begin{array} { r l } & { r _ { N } : = \underset { \eta \in \mathcal { T } _ { N } , \theta \in \Theta } { \operatorname* { s u p } } \Vert \mathrm { E } _ { P } [ \psi ( W ; \theta , \eta ) - \mathrm { E } _ { P } [ \psi ( W ; \theta , \eta _ { 0 } ) ] \Vert \leqslant \delta _ { N } \tau _ { N } , } \\ & { r _ { N } ^ { \prime } : = \underset { \eta \in \mathcal { T } _ { N } , \Vert \theta - \theta _ { 0 } \Vert \leqslant \tau _ { N } } { \operatorname* { s u p } } ( \mathrm { E } _ { P } [ \Vert \psi ( W ; \theta , \eta ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \Vert ^ { 2 } ] ) ^ { 1 / 2 } \ a n d \ r _ { N } ^ { \prime } \log ^ { 1 / 2 } ( 1 / r _ { N } ^ { \prime } ) \leqslant \delta _ { N } , } \\ & { \lambda _ { N } ^ { \prime } : = \underset { r \in ( 0 , 1 ) , \eta \in \mathcal { T } _ { N } , \Vert \theta - \theta _ { 0 } \Vert \leqslant \tau _ { N } } { \operatorname* { s u p } } \Vert \partial _ { r } ^ { 2 } \mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } + r ( \theta - \theta _ { 0 } ) + r ( \eta - \eta _ { 0 } ) ) ] \Vert \leqslant \delta _ { N } N ^ { - 1 / 2 } . } \end{array}
$$

(d) The variance of the score is non-degenerate: All eigenvalues of the matrix

$$
\mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ^ { \prime } ]
$$

are bounded from below by $c _ { 0 }$ .

Assumptions 3.3(a)-(c) state that the estimator of the nuisance parameter belongs to the realization set $\mathcal { T } _ { N } \subset T$ , which is a shrinking neighborhood of $\eta _ { 0 }$ that contracts at the “statistical” rates $r _ { N }$ and $r _ { N } ^ { \prime }$ and λ′N . These rates are not given in terms of the norm $\| \cdot \| _ { T }$ on $T$ , but rather are intrinsic rates that are connected to the statistical problem at hand. In smooth problems, these conditions translate to the crude requirement that nuisance parameters are estimated at the $o \big ( N ^ { - 1 / 4 } \big )$ rate as discussed previously in the case with linear scores. However, these conditions can be refined as, for example, when $\lambda _ { N } ^ { \prime } = 0$ or when some cross-derivatives vanish in $\lambda _ { N } ^ { \prime }$ ; see the linear case in the previous subsection for further discussion. Suitable measurability and pointwise entropy conditions, required in Assumption 3.4(b), are mild regularity conditions that are satisfied in all practical cases. The assumption of a bounded parameter space $\Theta$ in Assumption 3.4(b) is embedded in the entropy condition, but we state it separately for clarity. This assumption was not needed in the linear case, and it can be removed in the nonlinear case with the imposition of more complicated Huber-like regularity conditions. Assumption 3.4(c) is a set of mild growth conditions.

Remark 3.2. (Rate Requirements on Nuisance Parameter Estimators) Similar to the discussion in the linear case, the conditions in Assumption 3.4 are very flexible and embody refined requirements on the quality of the nuisance parameter estimators. The conditions essentially reduce to the previous conditions in the linear case, with the exception of compactness, which is imposed to make the conditions easy to verify in non-linear cases.

Theorem 3.3. (Properties of the DML for Nonlinear Scores) Suppose that Assumptions 3.3 and 3.4 hold. In addition, suppose that $\delta _ { N } \geqslant N ^ { - 1 / 2 + 1 / q } \log N$ and that $N ^ { - 1 / 2 } \log N \leqslant \tau _ { N } \leqslant \delta _ { N }$ for all $\textit { N } \geqslant \ 1$ . Then the DML1 and DML2 estimators $\tilde { \theta } _ { 0 }$ concentrate in a $1 / \sqrt { N }$ neighborhood of $\theta _ { 0 }$ , and are approximately linear and centered Gaussian:

$$
\sqrt { N } \sigma ^ { - 1 } ( \tilde { \theta } _ { 0 } - \theta _ { 0 } ) = \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \bar { \psi } ( W _ { i } ) + O _ { P } ( \rho _ { N } ) \sim N ( 0 , \mathrm { I } ) ,
$$

uniformly over $P \in \mathcal P _ { N }$ , where the size of the remainder term obeys

$$
\rho _ { N } : = N ^ { - 1 / 2 + 1 / q } \log N + r _ { N } ^ { \prime } \log ^ { 1 / 2 } ( 1 / r _ { N } ^ { \prime } ) + N ^ { 1 / 2 } \lambda _ { N } + N ^ { 1 / 2 } \lambda _ { N } ^ { \prime } \lesssim \delta _ { N } ,
$$

$\bar { \psi } ( \cdot ) : = - \sigma _ { 0 } ^ { - 1 } J _ { 0 } ^ { - 1 } \psi ( \cdot , \theta _ { 0 } , \eta _ { 0 } )$ is the influence function, and the approximate variance is

$$
\boldsymbol { \sigma } ^ { 2 } : = { J _ { 0 } ^ { - 1 } \mathrm { E } _ { P } } [ \psi ( \boldsymbol { W } ; \boldsymbol { \theta } _ { 0 } , \eta _ { 0 } ) \psi ( \boldsymbol { W } ; \boldsymbol { \theta } _ { 0 } , \eta _ { 0 } ) ^ { \prime } ] ( J _ { 0 } ^ { - 1 } ) ^ { \prime } .
$$

Moreover, in the statement above $\sigma ^ { 2 }$ can be replaced by a consistent estimator ${ \widehat { \sigma } } ^ { 2 }$ , obeying ${ \widehat { \sigma } } ^ { 2 } = \sigma ^ { 2 } + o _ { P } ( \varrho _ { N } )$ uniformly in $P \in \mathcal { P } _ { N }$ , with the size of the remainder terbm updated abs $\rho _ { N } = \rho _ { N } + \varrho _ { N }$ . Furthermore, Corollaries 3.1 and 3.2 continue to hold under the conditions of this theorem.

3.4. Finite-Sample Adjustments to Incorporate Uncertainty Induced by Sample Splitting

The estimation technique developed in this paper relies on subsamples obtained by randomly partitioning the sample: an auxiliary sample for estimating the nuisance functions and a main sample for estimating the parameter of interest. Although the specific sample partition has no impact on estimation results asymptotically, the effect of the particular random split on the estimate can be important in finite samples. To make the results more robust with respect to the partitioning, we propose to repeat the DML estimator $S$ times, obtaining the estimates

$$
\begin{array} { r } { \tilde { \theta } _ { 0 } ^ { s } , \quad s = 1 , \dots , S . } \end{array}
$$

Features of these estimates may then provide insight into the sensitivity of results to the sample splitting, and we can report results that incorporate features of this set of estimates that should be less driven by any particular sample-splitting realization.

Definition 3.3. (Incorporating the Impact of Sample Splitting using Mean and Median Methods) For point estimation, we define

$$
\tilde { \theta } _ { 0 } ^ { m e a n } = { \frac { 1 } { S } } \sum _ { s = 1 } ^ { S } \tilde { \theta } _ { 0 } ^ { s } o r \tilde { \theta } _ { 0 } ^ { m e d i a n } = m e d i a n \{ \tilde { \theta _ { 0 } ^ { s } } \} _ { s = 1 } ^ { S } ,
$$

where the median operation is applied coordinatewise. To quantify and incorporate the variation introduced by sample splitting, we consider variance estimators:

$$
\widehat { \sigma } ^ { 2 , m e a n } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \left( \widehat { \sigma } _ { s } ^ { 2 } + ( \widehat { \theta } _ { s } - \widetilde { \theta } ^ { m e a n } ) ( \widehat { \theta } _ { s } - \widetilde { \theta } ^ { m e a n } ) ^ { \prime } \right) ,
$$

and a more robust version,

$$
\widehat { \sigma } ^ { 2 , m e d i a n } = m e d i a n \{ \widehat { \sigma } _ { s } ^ { 2 } + ( ( \widehat { \theta } _ { s } - \widetilde { \theta } ^ { m e d i a n } ) ( \widehat { \theta } _ { s } - \widetilde { \theta } ^ { m e d i a n } ) ^ { \prime } \} _ { s = 1 } ^ { S } ,
$$

where the mediban picks out the matrix with median operator norm, which preserve nonnegative definiteness.

We recommend using medians, reporting θ˜0median and σ2Median, as these quantities are more robust to outliers.

Corollary 3.3. If $S$ is fixed, as $N  \infty$ and maintaining either Assumptions 3.1 and 3.2 or Assumptions 3.3 and 3.4 as appropriate, $\tilde { \theta } _ { 0 } ^ { m e a n }$ and θ˜0median are first-order equivalent to $\tilde { \theta } _ { 0 }$ and obey the conclusions of Theorems 3.1 and 3.2 or of Theorem 3.3. Moreover, σ2,median and ${ \widehat { \sigma } } ^ { 2 , m e a n }$ can replace $\widehat { \sigma }$ in the statement of the appropriate theorems.

It would be interesting to investigate the behavior under the regime where $S \to \infty$ as $N  \infty$ .

# 4. INFERENCE IN PARTIALLY LINEAR MODELS

# 4.1. Inference in Partially Linear Regression Models

Here we revisit the partially linear regression model

$$
\begin{array} { r l } { Y = D \theta _ { 0 } + g _ { 0 } ( X ) + U , } & { \mathrm { \normalfont ~ E } _ { P } [ U \mid X , D ] = 0 , } \\ { D = m _ { 0 } ( X ) + V , } & { \mathrm { \normalfont ~ E } _ { P } [ V \mid X ] = 0 . } \end{array}
$$

The parameter of interest is the regression coefficient $\theta _ { 0 }$ . If $D$ is conditionally exogenous (as good as randomly assigned conditional on covariates), then $\theta _ { 0 }$ measures the average causal/treatment effect of $D$ on potential outcomes.

The first approach to inference on $\theta _ { 0 }$ , which we described in the Introduction, is to employ the DML method using the score function

$$
\psi ( W ; \theta , \eta ) : = \{ Y - D \theta - g ( X ) \} ( D - m ( X ) ) , \quad \eta = ( g , m ) ,
$$

where $W = ( Y , D , X )$ and $g$ and $m$ are $P$ -square-integrable functions mapping the support of $X$ to $\mathbb { R }$ . It is easy to see that $\theta _ { 0 }$ satisfies the moment condition $\mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) = 0$ , and also the orthogonality condition $\partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] = 0$ where $\eta _ { 0 } = ( g _ { 0 } , m _ { 0 } )$ .

A second approach employs the Robinson-style “partialling-out” score function

$$
\cdot \ell ( X ) - \theta ( D - m ( X ) ) \} ( D - m ( X ) ) , \quad \eta = ( \ell , m )
$$

where $W = ( Y , D , X )$ and $\ell$ and $m$ are $P$ -square-integrable functions mapping the support of $X$ to $\mathbb { R }$ . This gives an alternative parameterization of the score function above, and using this score is first-order equivalent to using the previous score. It is easy to see that $\theta _ { 0 }$ satisfies the moment condition $\mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) = 0$ , and also the orthogonality condition $\partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] = 0$ , for $\eta _ { 0 } = ( \ell _ { 0 } , m _ { 0 } )$ , where $\ell _ { 0 } ( X ) = \operatorname { E } _ { P } [ Y | X ]$ .

In the partially linear model, the DML approach complements Belloni et al. (2013), Zhang and Zhang (2014), van de Geer et al. (2014), Javanmard and Montanari (201 4b), and Belloni et al. (2014), Belloni et al. (2014), and Belloni et al. (2015), all of which consider estimation and inference for parameters within the partially linear model using lasso-type methods without cross-fitting. By relying upon cross-fitting, the DML approach allows for the use of a much broader collection of ML methods for estimating the nuisance functions and also allows relaxation of sparsity conditions in the case where lasso or other sparsity-based estimators are used. Both the DML approach and the approaches taken in the aforementioned papers can be seen as heuristically “debiasing” the score function $( Y - D \theta - g ( X ) ) D$ , which does not possess the orthogonality property unless $m _ { 0 } ( X ) = 0$ .

Let $( \delta _ { N } ) _ { n = 1 } ^ { \infty }$ and $( \Delta _ { N } ) _ { n = 1 } ^ { \infty }$ be sequences of positive constants approaching 0 as before. Also, let $c$ , $C$ , and $q$ be fixed strictly positive constants such that $q > 4$ , and let $K \geqslant 2$ be a fixed integer. Moreover, for any $\eta = ( \ell _ { 1 } , \ell _ { 2 } )$ , where $\ell _ { 1 }$ and $\ell _ { 2 }$ are functions mapping the support of $X$ to $\mathbb { R }$ , denote $\| \eta \| _ { P , q } = \| \ell _ { 1 } \| _ { P , q } \vee \| \ell _ { 2 } \| _ { P , q }$ . For simplicity, assume that $N / K$ is an integer.

Assumption 4.1. (Regularity Conditions for Partially Linear Regression Model) Let $\mathcal { P }$ be the collection of probability laws $P$ for the triple $W = ( Y , D , X )$ such that (a) equations (4.1)-(4.2) hold; (b) $\| Y \| _ { P , q } + \| D \| _ { P , q } \leqslant C$ ; (c) $\| U V \| _ { P , 2 } \geqslant c ^ { 2 }$ and $\mathrm { E } _ { P } [ V ^ { 2 } ] \ \geqslant \ c$ ; $( d )$ $\| \mathrm { E } _ { P } [ U ^ { 2 } \mid X ] \| _ { P , \infty } \leqslant C$ and $\| \mathrm { E } _ { P } [ V ^ { 2 } \mid X ] \| _ { P , \infty } \leqslant C$ ; and (e) given a random subset $I$ of $[ N ]$ of size $n = N / K$ , the nuisance parameter estimator $\widehat { \eta } _ { 0 } = \widehat { \eta } _ { 0 } ( ( W _ { i } ) _ { i \in I ^ { c } } )$ obeys the following conditions for all $n \geqslant 1$ : With $P$ -probability no le sbthanb $1 - \Delta _ { N }$ ,

$$
\| \widehat { \eta } _ { 0 } - \eta _ { 0 } \| _ { P , \infty } \leqslant C , \quad \| \widehat { \eta } _ { 0 } - \eta _ { 0 } \| _ { P , 2 } \leqslant \delta _ { N } , \quad a n d ^ { 8 }
$$

(i) for the score $\psi$ in (4.3), where $\widehat { \eta } _ { 0 } = ( \widehat { g } _ { 0 } , \widehat { m } _ { 0 } )$ ,

$$
\lVert \widehat { m } _ { 0 } - m _ { 0 } \rVert _ { P , 2 } \times \lVert \widehat { g } _ { 0 } - g _ { 0 } \rVert _ { P , 2 } \leqslant \delta _ { N } N ^ { - 1 / 2 } ,
$$

(ii) for the score $\psi$ in (4.4b), where $\widehat { \eta } _ { 0 } = ( \widehat { \ell } _ { 0 } , \widehat { m } _ { 0 } )$ ,

$$
\begin{array} { r } { \| \widehat { m } _ { 0 } - m _ { 0 } \| _ { P , 2 } \times \left( \| \widehat { m } _ { 0 } - m _ { 0 } \| _ { P , 2 } + \| \widehat { \ell } _ { 0 } - \ell _ { 0 } \| _ { P , 2 } \right) \leqslant \delta _ { N } N ^ { - 1 / 2 } . } \end{array}
$$

Remark 4.1. (Rate Conditions for Estimators of Nuisance Parameters) The only nonprimitive condition here is the assumption on the rate of estimating the nuisance parameters. These rates of convergence are available for most often used ML methods and are case-specific, so we do not restate conditions that are needed to reach these rates.

The following theorem follows as a corollary to the results in Section 3 by verifying Assumptions 3.1 and 3.2 and will be proven as a special case of Theorem 4.2 below.

Theorem 4.1. (DML Inference on Regression Coefficients in the Partially Linear Regression Model) Suppose that Assumption 4.1 holds. Then the DML1 and DML2 estimators $\tilde { \theta } _ { 0 }$ constructed in Definitions 3.1 and 3.2 above using the score in either (4.3) or (4.4) are first-order equivalent and obey

$$
\sigma ^ { - 1 } \sqrt { N } ( \widetilde { \theta } _ { 0 } - \theta _ { 0 } ) \sim N ( 0 , 1 ) ,
$$

uniformly over $P \in \mathcal { P }$ , where $\sigma ^ { 2 } = [ \mathrm { E } _ { P } V ^ { 2 } ] ^ { - 1 } \mathrm { E } _ { P } [ V ^ { 2 } U ^ { 2 } ] [ \mathrm { E } _ { P } V ^ { 2 } ] ^ { - 1 }$ . Moreover, the result continues to hold if $\sigma ^ { 2 }$ is replaced by ${ \widehat { \sigma } } ^ { 2 }$ defined in Theorem 3.2. Consequently, confidence regions based upon the DML estimatbors $\tilde { \theta } _ { 0 }$ have uniform asymptotic validity:

$$
\operatorname* { l i m } _ { N \to \infty } \operatorname* { s u p } _ { P \in \mathcal { P } } \left| \operatorname* { P } _ { P } \Big ( \theta _ { 0 } \in [ \tilde { \theta } _ { 0 } \pm \Phi ^ { - 1 } ( 1 - \alpha / 2 ) \widehat \sigma / \sqrt { N } ] \Big ) - ( 1 - \alpha ) \right| = 0 .
$$

8We thank Rui Wang from the University of Washington for pointing out a mistake in the published version of the paper: In Assumptions 4.1 and 4.2, the correct condition is $\| \widehat { \eta } _ { 0 } - \eta _ { 0 } \| _ { P , \infty } \leqslant C$ rather than $\| \widehat { \eta } _ { 0 } - \eta _ { 0 } \| _ { P , q } \leqslant C$ appearing in the published version.

Remark 4.2. (Asymptotic Efficiency under Homoscedasticity) Under conditional homoscedasticity, i.e. $\mathrm { E } [ U ^ { 2 } | Z ] = \mathrm { E } [ U ^ { 2 } ]$ , the as ymptotic variance σ2 reduces to $\operatorname { E } [ V ^ { 2 } ] ^ { - 1 } \operatorname { E } [ U ^ { 2 } ]$ which is the semi-parametric efficiency bound for $\theta$ .

Remark 4.3. (Tightness of Conditions under Cross-Fitting) The conditions in Theorem 4.1 are fairly sharp, though they are somewhat simplified for ease of presentation. The sharpness can be understood by examining the case where the regression function $g _ { 0 }$ and the propensity function $m _ { 0 }$ are sparse with sparsity indices $s ^ { g } \ll N$ N and $s ^ { m } \ll N$ N and are estimated by $\ell _ { 1 }$ -penalized estimators $\widehat { g } _ { 0 }$ and $\hat { m } _ { 0 }$ that have sparsity indices of orders $s ^ { g }$ and $s ^ { m }$ and converge to $g _ { 0 }$ and $m _ { 0 }$ at tbhe rat sb $\sqrt { s ^ { g } / N }$ and $\sqrt { { s } ^ { m } / N }$ (ignoring logs). The rate conditions in Assumption 4.1 then require (ignoring logs) that

$$
\sqrt { s ^ { g } / N } \sqrt { s ^ { m } / N } \ll N ^ { - 1 / 2 } \Leftrightarrow s ^ { g } s ^ { m } \ll N ,
$$

which is much weaker than the condition

$$
( s ^ { g } ) ^ { 2 } + ( s ^ { m } ) ^ { 2 } \ll N
$$

(ignoring logs) required without sample splitting. For example, if the propensity function $m _ { 0 }$ is very sparse (low $s _ { m }$ ), then the regression function is allowed to be quite dense (high $s _ { g }$ ), and vice versa. If the propensity function is known ( $s ^ { m } = 0$ ) or can be estimated at the N −1/2 rate, then only consistency for $\widehat { g } _ { 0 }$ is needed. Such comparisons also extend to approximately sparse models.

# 4.2. Inference in Partially Linear IV Models

Here we extend the partially linear regression model studied in Section 4.1 to allow for instrumental variable (IV) identification. Specifically, we consider the model

$$
\begin{array} { r l } { Y = D \theta _ { 0 } + g _ { 0 } ( X ) + U , } & { \mathrm {  { E } } _ { P } [ U \mid X , Z ] = 0 , } \\ { Z = m _ { 0 } ( X ) + V , } & { \mathrm {  { E } } _ { P } [ V \mid X ] = 0 , } \end{array}
$$

where $Z$ is the instrumental variable. As before, the parameter of interest is $\theta$ and its true value is $\theta _ { 0 }$ . If $Z = D$ , the model (4.5)-(4.6) coincides with (4.1)-(4.2) but is otherwise different.

To estimate $\theta _ { 0 }$ and to perform inference on it, we will use the score

$$
\psi ( W ; \theta , \eta ) : = ( Y - D \theta - g ( X ) ) ( Z - m ( X ) ) , \quad \eta = ( g , m ) ,
$$

where $W = ( Y , D , X , Z )$ and $g$ and $m$ are $P$ -square-integrable functions mapping the support of $X$ to $\mathbb { R }$ . Alternatively, we can use the Robinson-style score

$$
\psi ( W ; \theta , \eta ) : = ( Y - \ell ( X ) - \theta ( D - r ( X ) ) ) ( Z - m ( X ) ) , \quad \eta = ( \ell , m , r ) ,
$$

where $W = ( Y , D , X , Z )$ and $\ell$ , $m$ , and $r$ are $P$ -square-integrable functions mapping the support of $X$ to $\mathbb { R }$ . It is straightforward to verify that both scores satisfy the moment condition $\mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) = 0$ and also the orthogonality condition $\partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta -$ $\left. \eta _ { 0 } \right] = 0$ , for $\eta _ { 0 } = \left( g _ { 0 } , m _ { 0 } \right)$ in the former case and $\eta _ { 0 } = ( \ell _ { 0 } , m _ { 0 } , r _ { 0 } )$ for $\ell _ { 0 }$ and $r _ { 0 }$ defined by $\ell _ { 0 } ( X ) = \operatorname { E } _ { P } [ Y \mid X ]$ and $r _ { 0 } ( X ) = \operatorname { E } _ { P } [ D \mid X ]$ , respectively, in the latter case.9

${ } ^ { 9 } \mathrm { I t }$ is interesting to note that the methods for constructing Neyman orthogonal scores described in Section 2 may give scores that are different from those in (4.7) and (4.8). For example, applying the method for conditional moment restriction problems in Section 2.2.4 with $\Omega ( R ) = 1$ gives the score

Note that the score in (4.8) has a minor advantage over the score in (4.7) because all of its nuisance parameters are conditional mean functions, which can be directly estimated by the ML methods. If one prefers to use the score in (4.7), one has to construct an estimator the score in of (4.8), $g _ { 0 }$ first. say To $\tilde { \theta } _ { 0 }$ do . Then, so, one using can the first fact obtain that a $g _ { 0 } ( X ) = \operatorname { E } _ { P } [ Y - D \theta _ { 0 } \mid _ { - } X ]$ DML estimator of $\theta _ { 0 }$ based , one on can construct an estimator $\widehat { g } _ { 0 }$ by applying an ML method to regress $Y - D \ddot { \theta } _ { 0 }$ on $X$ . Alternatively, one can use asbsumption-specific methods to directly estimate $g _ { 0 }$ , without using the score (4.8) first. For example, if $g _ { 0 }$ can be approximated by a sparse linear combination of a large set of transformations of $X$ , one can use the methods of Gautier and Tsybakov (2014) to obtain an estimator of $g _ { 0 }$ .

Let $( \delta _ { N } ) _ { n = 1 } ^ { \infty }$ and $( \Delta _ { N } ) _ { n = 1 } ^ { \infty }$ be sequences of positive constants approaching 0 as before. Also, let $c$ , $C$ , and $q$ be fixed strictly positive constants such that $q > 4$ , and let $K \geqslant 2$ be a fixed integer. Moreover, for any $\boldsymbol { \eta } = ( \ell _ { j } ) _ { j = 1 } ^ { l }$ mapping the support of $X$ to $\mathbb { R } ^ { l }$ , denote $\| \eta \| _ { P , q } = \operatorname* { m a x } _ { 1 \leqslant j \leqslant l } \| \ell _ { j } \| _ { P , q }$ . For simplicity, assume that $N / K$ is an integer.

Assumption 4.2. (Regularity Conditions for Partially Linear IV Model) For all probability laws $P \in \mathcal { P }$ for the quadruple $W = ( Y , D , X , Z )$ the following conditions hold: $( a )$ equations (4.5)-(4.6) hold; (b) $\| Y \| _ { P , q } + \| D \| _ { P , q } + \| Z \| _ { P , q } \leqslant C$ ; (c) $\| U V \| _ { P , 2 } \geqslant c ^ { 2 }$ and $| \mathrm { E } _ { P } [ D V ] | \geqslant c$ ; (d) $\| \mathrm { E } _ { P } [ U ^ { 2 } \ | \ X ] \| _ { P , \infty } \leqslant C$ and $\| \mathrm { E } _ { P } [ V ^ { 2 } \ | \ X ] \| _ { P , \infty } \leqslant C$ ; and (e) given a random subset $I$ of $[ N ]$ of size $n = N / K$ , the nuisance parameter estimator $\widehat { \eta } _ { 0 } = \widehat { \eta } _ { 0 } \big ( ( W _ { i } ) _ { i \in I ^ { c } } \big )$ obeys the following conditions: With $P$ -probability no less than $1 - \Delta _ { N }$ ,

$$
\| \widehat { \eta } _ { 0 } - \eta _ { 0 } \| _ { P , \infty } \leqslant C , \quad \| \widehat { \eta } _ { 0 } - \eta _ { 0 } \| _ { P , 2 } \leqslant \delta _ { N } , \quad a n d
$$

(i) for the score $\psi$ in (4.7), where $\widehat { \eta } _ { 0 } = ( \widehat { g } _ { 0 } , \widehat { m } _ { 0 } )$ ,

$$
\lVert \widehat { m } _ { 0 } - m _ { 0 } \rVert _ { P , 2 } \times \lVert \widehat { g } _ { 0 } - g _ { 0 } \rVert _ { P , 2 } \leqslant \delta _ { N } N ^ { - 1 / 2 } ,
$$

(ii) for the score $\psi$ in (4.8), where $\widehat { \eta } _ { 0 } = ( \widehat { \ell } _ { 0 } , \widehat { m } _ { 0 } , \widehat { r } _ { 0 } )$ ,

$$
\| \widehat { m } _ { 0 } - m _ { 0 } \| _ { P , 2 } \times \left( \| \widehat { r } _ { 0 } - r _ { 0 } \| _ { P , 2 } + \| \widehat { \ell } _ { 0 } - \ell _ { 0 } \| _ { P , 2 } \right) \leqslant \delta _ { N } N ^ { - 1 / 2 } .
$$

The following theorem follows as a corollary to the results in Section 3 by verifying Assumptions 3.1 and 3.2.

Theorem 4.2. (DML Inference on Regression Coefficients in the Partially Linear IV Model) Suppose that Assumption 4.2 holds. Then the DML1 and DML2 estimators $\ddot { \theta } _ { 0 }$ constructed in Definitions 3.1 and 3.2 above using the score in either (4.7) or (4.8) are first-order equivalent and obey

$$
\sigma ^ { - 1 } \sqrt { N } ( \widetilde { \theta } _ { 0 } - \theta _ { 0 } ) \sim N ( 0 , 1 ) ,
$$

uniformly over $P \in \mathcal { P }$ , where $\sigma ^ { 2 } = [ \mathrm { E } _ { P } D V ] ^ { - 1 } \mathrm { E } _ { P } [ V ^ { 2 } U ^ { 2 } ] [ \mathrm { E } _ { P } D V ] ^ { - 1 }$ . Moreover, the result continues to hold if $\sigma ^ { 2 }$ is replaced by ${ \widehat { \sigma } } ^ { 2 }$ defined in Theorem 3.2. Consequently, confidence regions based upon the DML estimatbors $\tilde { \theta } _ { 0 }$ have uniform asymptotic validity:

$$
\operatorname* { l i m } _ { N \to \infty } \operatorname* { s u p } _ { P \in \mathcal { P } } \left| \operatorname* { P } _ { P } \left( \theta _ { 0 } \in [ \tilde { \theta } _ { 0 } \pm \Phi ^ { - 1 } ( 1 - \alpha / 2 ) \widehat \sigma / \sqrt { N } ] \right) - ( 1 - \alpha ) \right| = 0 .
$$

$\psi ( W ; \theta , \eta ) = ( Y - D \theta - g ( X ) ) ( r ( Z , X ) - f ( X ) )$ , where the true values of $r ( Z , X )$ and $f ( X )$ are $r _ { 0 } ( Z , X ) =$ $\operatorname { E } _ { P } [ D \mid Z , X ]$ and $f _ { 0 } ( X ) = \operatorname { E } _ { P } [ D \mid X ]$ , respectively. It may be interesting to compare properties of the DML estimators $ { \widetilde { \theta } } _ { 0 }$ based on this score with those based on (4.7) and (4.8) in future work.

# 5. INFERENCE ON TREATMENT EFFECTS IN THE INTERACTIVE MODEL

# 5.1. Inference on ATE and ATTE

In this section, we specialize the results of Section 3 to estimating treatment effects under the unconfoundedness assumption of Rosenbaum and Rubin (1983). Within this setting, there is a large classical literature focused on low-dimensional settings that provides methods for adjusting for confounding variables including regression methods, propensity score adjustment methods, matching methods, and “doubly-robust” combinations of these methods; see, for example, Robins and Rotnitzky (1995), Hahn (1998), Hirano et al. (2003), and Abadie and Imbens (2006) as well as the textbook overview provided in Imbens and Rubin (2015). In this section, we present results that complement this important classic work as well as the rapidly expanding body of work on estimation under unconfoundedness using ML methods; see, among others, Athey et al. (2016), Belloni et al. (2017), Belloni et al. (2014), Farrell (2015), and Imai and Ratkovic (2013).

We specifically consider estimation of average treatment effects when treatment effects are fully heterogeneous and the treatment variable is binary, $D \in \{ 0 , 1 \}$ . We consider vectors $( Y , D , X )$ such that

$$
\begin{array} { r l } { Y = g _ { 0 } ( D , X ) + U , } & { \mathrm { \normalfont ~ E } _ { P } [ U \mid X , D ] = 0 , } \\ { D = m _ { 0 } ( X ) + V , } & { \mathrm { \normalfont ~ E } _ { P } [ V \mid X ] = 0 . } \end{array}
$$

Since $D$ is not additively separable, this model is more general than the partially linear model for the case of binary $D$ . A common target parameter of interest in this model is the average treatment effect (ATE),

$$
\theta _ { 0 } = \mathrm { E } _ { P } [ g _ { 0 } ( 1 , X ) - g _ { 0 } ( 0 , X ) ] . ^ { 1 0 }
$$

Another common target parameter is the average treatment effect for the treated (ATTE),

$$
\theta _ { 0 } = \mathrm { E } _ { P } [ g _ { 0 } ( 1 , X ) - g _ { 0 } ( 0 , X ) | D = 1 ] .
$$

The confounding factors $X$ affect the policy variable via the propensity score $m _ { 0 } ( X )$ and the outcome variable via the function $g _ { 0 } ( D , X )$ . Both of these functions are unknown and potentially complicated, and we can employ ML methods to learn them.

We proceed to set up moment conditions with scores obeying orthogonality conditions. For estimation of the ATE, we employ

$$
\psi ( W ; \theta , \eta ) : = \left( g ( 1 , X ) - g ( 0 , X ) \right) + \frac { D ( Y - g ( 1 , X ) ) } { m ( X ) } - \frac { ( 1 - D ) ( Y - g ( 0 , X ) ) } { 1 - m ( X ) } - \theta ,
$$

where the nuisance parameter $\eta = ( g , m )$ consists of $P$ -square-integrable functions $g$ and $m$ mapping the support of $( D , X )$ to $\mathbb { R }$ and the support of $X$ to $( \varepsilon , 1 - \varepsilon )$ , respectively, for some $\varepsilon \in ( 0 , 1 / 2 )$ . The true value of $\eta$ is ${ \eta _ { 0 } } = ( g _ { 0 } , m _ { 0 } )$ . This orthogonal moment condition is based on the influence function for the mean for missing data from Robins and Rotnitzky (1995).

For estimation of the ATTE, we use the score

$$
\psi ( W ; \theta , \eta ) = \frac { D ( Y - \overline { { { g } } } ( X ) ) } { p } - \frac { m ( X ) ( 1 - D ) ( Y - \overline { { { g } } } ( X ) ) } { p ( 1 - m ( X ) ) } - \frac { D \theta } { p } ,
$$

10Without unconfoundedness/conditional exogeneity, these quantities measure association, and could be referred to as average predictive effect (APE) and average predictive effect for the exposed (APEX). Inferential results for these objects would follow immediately from Theorem 5.1.

where the nuisance parameter $\eta \ : = \ : ( \overline { { { g } } } , m , p )$ consists of $P$ -square-integrable functions $\overline { { g } }$ and $m$ mapping the support of $X$ to $\mathbb { R }$ and to $( \varepsilon , 1 - \varepsilon )$ , respectively, and a constant $p \in ( \varepsilon , 1 - \varepsilon )$ , for some $\varepsilon \in ( 0 , 1 / 2 )$ . The true value of $\eta$ is $\eta _ { 0 } = ( \overline { { g } } _ { 0 } , m _ { 0 } , p _ { 0 } )$ , where $\overline { { { g } } } _ { 0 } ( X ) = g _ { 0 } ( 0 , X )$ and $p _ { 0 } = \mathrm { E } _ { P } [ D ]$ . Note that estimating ATTE does not require estimating $g _ { 0 } ( 1 , X )$ . Note also that since $p$ is a constant, it does not affect the DML estimators $\tilde { \theta } _ { 0 }$ based on the score $\psi$ in (5.4) but having $p$ simplifies the formula for the variance of $\ddot { \theta } _ { 0 }$ .

Using their respective scores, it can be easily seen that true parameter values $\theta _ { 0 }$ for ATE and ATTE obey the moment condition $\mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) = 0$ , and also that the orthogonality condition $\partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] = 0$ holds.

Let $( \delta _ { N } ) _ { n = 1 } ^ { \infty }$ and $( \Delta _ { N } ) _ { n = 1 } ^ { \infty }$ be sequences of positive constants approaching 0. Also, let $c , \varepsilon , C$ and $q$ be fixed strictly positive constants such that $q > 2$ , and let $\textit { K } \geqslant \ 2$ be a fixed integer. Moreover, for any $\boldsymbol { \eta } = ( \ell _ { 1 } , \dots , \ell _ { l } )$ , denote $\| \eta \| _ { P , q } = \operatorname* { m a x } _ { 1 \leqslant j \leqslant l } \| \ell _ { j } \| _ { P , q }$ . For simplicity, assume that $N / K$ is an integer.

Assumption 5.1. (Regularity Conditions for ATE and ATTE Estimation) For all probability laws $P \in \mathcal { P }$ for the triple $( Y , D , X )$ the following conditions hold: (a) equations (5.1)-(5.2) hold, with $D \in \{ 0 , 1 \}$ , (b) $\| Y \| _ { P , q } \leqslant C$ , (c) $\mathrm { P } _ { P } ( \varepsilon \leqslant m _ { 0 } ( X ) \leqslant 1 - \varepsilon ) = 1$ , (d) $\| U \| _ { P , 2 } \geqslant c$ , (e) $\| \mathrm { E } _ { P } [ U ^ { 2 } \mid X ] \| _ { P , \infty } \leqslant C$ , and $( f )$ given a random subset $I$ of $[ N ]$ of size $n = N / K$ , the nuisance parameter estimator $\widehat { \eta } _ { 0 } = \widehat { \eta } _ { 0 } ( ( W _ { i } ) _ { i \in I ^ { c } } )$ obeys the following conditions: with $P$ -probability no less than $1 - \Delta _ { N }$ ∆N:b

$$
\| \widehat { \eta } _ { 0 } - \eta _ { 0 } \| _ { P , q } \leqslant C , \quad \| \widehat { \eta } _ { 0 } - \eta _ { 0 } \| _ { P , 2 } \leqslant \delta _ { N } , \quad \| \widehat { m } _ { 0 } - 1 / 2 \| _ { P , \infty } \leqslant 1 / 2 - \varepsilon , \quad a n d
$$

(i) for the score $\psi$ in (5.3), where $\widehat { \eta } _ { 0 } = ( \widehat { g } _ { 0 } , \widehat { m } _ { 0 } )$ and the target parameter is $A T E$ ,

$$
\lVert \widehat { m } _ { 0 } - m _ { 0 } \rVert _ { P , 2 } \times \lVert \widehat { g } _ { 0 } - g _ { 0 } \rVert _ { P , 2 } \leqslant \delta _ { N } N ^ { - 1 / 2 } ,
$$

(ii) for the score $\psi$ in (5.4)b, where $\widehat { \eta } _ { 0 } = ( \widehat { \overline { { g } } } _ { 0 } , \widehat { m } _ { 0 } , \widehat { p } _ { 0 } )$ and the target parameter is ATTE,

$$
\lVert \widehat { m } _ { 0 } - m _ { 0 } \rVert _ { P , 2 } \times \lVert \widehat { \overline { { g } } } _ { 0 } - \overline { { g } } _ { 0 } \rVert _ { P , 2 } \leqslant \delta _ { N } N ^ { - 1 / 2 } .
$$

Remark 5.1. The only non-primitive condition here is the assumption on the rate of estimating the nuisance parameters. These rates of convergence are available for most often used ML methods and are case-specific, so we do not restate conditions that are needed to reach these rates. The conditions are not the tightest possible, but offer a set of simple conditions under which Theorem 5.1 follows as a special case of the general theorem provided in Section 3. One could obtain more refined conditions by doing customized proofs.

The following theorem follows as a corollary to the results in Section 3 by verifying Assumptions 3.1 and 3.2.

Theorem 5.1. (DML Inference on ATE and ATTE) Suppose that either (a) the target parameter is ATE, $\theta _ { 0 } = \mathrm { E } _ { P } [ g _ { 0 } ( 1 , X ) - g _ { 0 } ( 0 , X ) ]$ , and the score $\psi$ in (5.3) is used, or (b) the target parameter is ATTE, $\theta _ { 0 } = \mathrm { E } _ { P } [ g _ { 0 } ( 1 , X ) - g _ { 0 } ( 0 , X ) \mid D = 1 ]$ , and the score $\psi$ in (5.4) is used. In addition, suppose that Assumption 5.1 holds. Then the DML1 and DML2 estimators $\tilde { \theta } _ { 0 }$ , constructed in Definitions 3.1 and 3.2, are first-order equivalent and obey

$$
\sigma ^ { - 1 } \sqrt { N } ( \widetilde { \theta } _ { 0 } - \theta _ { 0 } )  N ( 0 , 1 ) ,
$$

uniformly over $P \in \mathcal { P }$ , where $\sigma ^ { 2 } = \mathrm { E } _ { P } [ \psi ^ { 2 } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ]$ . Moreover, the result continue o hold if $\sigma ^ { 2 }$ is replaced by ${ \widehat { \sigma } } ^ { 2 }$ defined in Theorem 3.2. Consequently, confidence region based upon the DML estimabtors $\tilde { \theta } _ { 0 }$ have uniform asymptotic validity:

$$
\operatorname* { l i m } _ { N \to \infty } \operatorname* { s u p } _ { P \in \mathcal { P } } \left| \operatorname* { P } _ { P } \left( \theta _ { 0 } \in [ \tilde { \theta } _ { 0 } \pm \Phi ^ { - 1 } ( 1 - \alpha / 2 ) \widehat \sigma / \sqrt { N } ] \right) - ( 1 - \alpha ) \right| = 0 .
$$

The scores $\psi$ in (5.3) and (5.4) are efficient, so both estimators are asymptotically efficient, reaching the semi-parametric efficiency bound of Hahn (1998).

Remark 5.2. (Tightness of Conditions) The conditions in Assumption 5.1 are fairly sharp though somewhat simplified for ease of presentation. The sharpness can be understood by examining the case where the regression function $g _ { 0 }$ and the propensity function $m _ { 0 }$ are sparse with sparsity indices $s ^ { g } \ll N$ and $s ^ { m } \ll N$ and are estimated by $\ell _ { 1 }$ -penalized estimators $\widehat { g } _ { 0 }$ and $\hat { m } _ { 0 }$ that have sparsity indices of orders $s ^ { y }$ and $s ^ { m }$ and converge to $g _ { 0 }$ and $m _ { 0 }$ at tbhe rat sb $\sqrt { s ^ { g } / N }$ and $\sqrt { { s } ^ { m } / N }$ (ignoring logs). Then the rate conditions in Assumption 5.1 require

$$
\sqrt { s ^ { g } / N } \sqrt { s ^ { m } / N } \ll N ^ { - 1 / 2 } \Leftrightarrow s ^ { g } s ^ { m } \ll N
$$

(ignoring logs) which is much weaker than the condition $( s ^ { g } ) ^ { 2 } + ( s ^ { m } ) ^ { 2 } \ll N$ (ignoring logs) required without sample splitting. For example, if the propensity score $m _ { 0 }$ is very sparse, then the regression function is allowed to be quite dense with $s ^ { g } > \sqrt { N }$ , and vice versa. If the propensity score is known ( $s ^ { m } = 0$ ), then only consistency for $\widehat { g } _ { 0 }$ is needed. Such comparisons also extend to approximately sparse models. We note bthat similar refined rates appeared in Farrell (2015) who considers estimation of treatment effects in a setting where an approximately sparse model holds for both the regression and propensity score functions. In interesting related work, Athey et al. (2016) show that $\sqrt { N }$ consistent estimation of an average treatment effect is possible under very weak conditions on the propensity score - allowing for the possibility that the propensity score may not be consistently estimated under strong sparsity of the regression function such that $s _ { g } \ll \sqrt { N }$ . Thus, the approach taken in this context and Athey et al. (2016) are complementary and one may prefer either depending on whether or not the regression function can be estimated extremely well based on a sparse method.

# 5.2. Inference on Local Average Treatment Effects

In this section, we consider estimation of local average treatment effects (LATE) with a binary treatment variable, $D \in \{ 0 , 1 \}$ , and a binary instrument, $Z \in \{ 0 , 1 \}$ .11 As before, $Y$ denotes the outcome variable, and $X$ is the vector of covariates.

Consider the functions $\mu _ { 0 }$ , $m _ { 0 }$ , and $p _ { 0 }$ , where $\mu _ { 0 }$ maps the support of $( Z , X )$ to $\mathbb { R }$ and $m _ { 0 }$ and $p _ { 0 }$ respectively map the support of $( Z , X )$ and $X$ to $( \varepsilon , 1 - \varepsilon )$ for some $\varepsilon \in ( 0 , 1 / 2 )$ , such that

$$
\begin{array} { r l } { Y = \mu _ { 0 } ( Z , X ) + U , } & { \mathrm { ~ E } _ { P } [ U \mid Z , X ] = 0 , } \\ { D = m _ { 0 } ( Z , X ) + V , } & { \mathrm { ~ E } _ { P } [ V \mid Z , X ] = 0 , } \\ { Z = p _ { 0 } ( X ) + \zeta , } & { \mathrm { ~ E } _ { P } [ \zeta \mid X ] = 0 . } \end{array}
$$

We are interested in estimating

$$
\theta _ { 0 } = \frac { \mathrm { E } _ { P } [ \mu ( 1 , X ) ] - \mathrm { E } _ { P } [ \mu ( 0 , X ) ] } { \mathrm { E } _ { P } [ m ( 1 , X ) ] - \mathrm { E } _ { P } [ m ( 0 , X ) ] } .
$$

Under the assumptions of Imbens and Angrist (1994) and Fr¨olich (2007), $\theta _ { 0 }$ is the LATE - the average treatment effect for compliers which are observations that would have $D = 1$ if $Z$ were $^ { 1 }$ and would have $D = 0$ if $Z$ were 0. To estimate $\theta _ { 0 }$ , we will use the score

$$
\begin{array} { l } { \displaystyle \psi ( W ; \theta , \eta ) : = \mu ( 1 , X ) - \mu ( 0 , X ) + \frac { Z ( Y - \mu ( 1 , X ) ) } { p ( X ) } - \frac { ( 1 - Z ) ( Y - \mu ( 1 , X ) ) } { 1 - p ( X ) ) } \qquad \quad } \\ { \displaystyle - \left( m ( 1 , X ) - m ( 0 , X ) + \frac { Z ( D - m ( 1 , X ) ) } { p ( X ) } - \frac { ( 1 - Z ) ( D - m ( 0 , X ) ) } { 1 - p ( X ) } \right) \times \theta , } \end{array}
$$

where $W = ( Y , D , X , Z )$ and the nuisance parameter $\boldsymbol { \eta } = ( \mu , m , p )$ consists of P -squareintegrable functions $\mu$ , $m$ , and $p$ , with $\mu$ mapping the support of $( Z , X )$ to $\mathbb { R }$ and $m$ and $p$ respectively mapping the support of $( Z , X )$ and $X$ to $( \varepsilon , 1 - \varepsilon )$ for some $\varepsilon \in ( 0 , 1 / 2 )$ . It is easy to verify that this score satisfies the moment condition $\mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) = 0$ and also the orthogonality condition $\partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] = 0$ for $\eta _ { 0 } = ( \mu _ { 0 } , m _ { 0 } , p _ { 0 } )$ .

Let $( \delta _ { N } ) _ { n = 1 } ^ { \infty }$ and $( \Delta _ { N } ) _ { n = 1 } ^ { \infty }$ be sequences of positive constants approaching 0. Also, let $c$ , $C$ , and $q$ be fixed strictly positive constants such that $q > 4$ , and let $K \geqslant 2$ be a fixed integer. Moreover, for any $\eta = ( \ell _ { 1 } , \ell _ { 2 } , \ell _ { 3 } )$ , where $\ell _ { 1 }$ is a function mapping the support of $( Z , X )$ to $\mathbb { R }$ and $\ell _ { 2 }$ and $\ell _ { 3 }$ are functions respectively mapping the support of $( Z , X )$ and X to $( \varepsilon , 1 - \varepsilon )$ for some $\varepsilon \in ( 0 , 1 / 2 )$ , denote $\| \eta \| _ { P , q } = \| \ell _ { 1 } \| _ { P , q } \vee \| \ell _ { 2 } \| _ { P , 2 } \vee \| \ell _ { 3 } \| _ { P , q }$ . For simplicity, assume that $N / K$ is an integer.

Assumption 5.2. (Regularity Conditions for LATE Estimation) For all probability laws $P \in \mathcal { P }$ for the quadruple $W = ( Y , D , X , Z )$ the following conditions hold: (a) equations $( 5 . 6 )$ -(5.8) hold, with $D \in \{ 0 , 1 \}$ and $Z \in \{ 0 , 1 \}$ ; (b) $\| Y \| _ { P , q } \leqslant C$ ; (c) $\mathrm { P } _ { P } ( \varepsilon \leqslant p _ { 0 } ( X ) \leqslant$ $1 - \varepsilon ) = 1$ , (d) $\mathrm { E } _ { P } [ m _ { 0 } ( 1 , X ) - m _ { 0 } ( 0 , X ) ] \ \geqslant \ c$ , (e) $\lVert U - \theta _ { 0 } V \rVert _ { P , 2 } \geqslant c$ ; (f ) $\Vert \mathrm { E } _ { P } [ U ^ { 2 } \ ]$ $X ] \| _ { P , \infty } \leqslant C$ ; and (g) given a random subset $I$ of $[ N ]$ of size $n = N / K$ , the nuisance parameter estimator $\widehat { \eta } _ { 0 } = \widehat { \eta } _ { 0 } \big ( ( W _ { i } ) _ { i \in I ^ { c } } \big )$ obeys the following conditions: with $P$ -probability no less than $1 - \Delta _ { N }$ b

$$
\begin{array} { r l } & { \| \widehat { \eta } _ { 0 } - \eta _ { 0 } \| _ { P , q } \leqslant C , \quad \| \widehat { \eta } _ { 0 } - \eta _ { 0 } \| _ { P , 2 } \leqslant \delta _ { N } , \quad \| \widehat { p } _ { 0 } - 1 / 2 \| _ { P , \infty } \leqslant 1 / 2 - \varepsilon , \quad a n d } \\ & { \qquad \| \widehat { p } _ { 0 } - p _ { 0 } \| _ { P , 2 } \times \Big ( \| \widehat { \mu } _ { 0 } - \mu _ { 0 } \| _ { P , 2 } + \| \widehat { m } _ { 0 } - m _ { 0 } \| _ { P , 2 } \Big ) \leqslant \delta _ { N } N ^ { - 1 / 2 } . } \end{array}
$$

The following theorem follows as a corollary to the results in Section 3 by verifying Assumptions 3.1 and 3.2.

Theorem 5.2. (DML Inference on LATE) Suppose that Assumption 5.2 holds. Then the DML1 and DML2 estimators $\tilde { \theta } _ { 0 }$ constructed in Definitions 3.1 and 3.2 and based on the score $\psi$ above are first-order equivalent and obey

$$
\sigma ^ { - 1 } \sqrt { N } ( \widetilde { \theta } _ { 0 } - \theta _ { 0 } )  N ( 0 , 1 ) ,
$$

uniformly over $P \in \mathcal { P }$ , where $\sigma ^ { 2 } = ( \mathrm { E } _ { P } [ m ( 1 , X ) - m ( 0 , X ) ] ) ^ { - 2 } \mathrm { E } _ { P } [ \psi ^ { 2 } ( W ; \theta _ { 0 } , \eta _ { 0 } )$ )]. Moreover, the result continues to hold if $\sigma ^ { 2 }$ is replaced by ${ \widehat { \sigma } } ^ { 2 }$ defined in Theorem 3.2. Consequently, confidence regions based upon the DML estimbators $\tilde { \theta } _ { 0 }$ have uniform asymptotic

validity:

$$
\operatorname* { l i m } _ { N \to \infty } \operatorname* { s u p } _ { P \in \mathcal { P } } \left| \operatorname* { P } _ { P } \left( \theta _ { 0 } \in [ \tilde { \theta } _ { 0 } \pm \Phi ^ { - 1 } ( 1 - \alpha / 2 ) \widehat \sigma / \sqrt { N } ] \right) - ( 1 - \alpha ) \right| = 0 .
$$

# 6. EMPIRICAL EXAMPLES

To illustrate the methods developed in the preceding sections, we consider three empirical examples. The first example reexamines the Pennsylvania Reemployment Bonus experiment which used a randomized control trial to investigate the incentive effect of unemployment insurance. In the second, we use the DML method to estimate the effect of 401(k) eligibility, the treatment variable, and 401(k) participation, a self-selected decision to receive the treatment that we instrument for with assignment to the treatment state, on accumulated assets. In this example, the treatment variable is not randomly assigned and we aim to eliminate the potential biases due to the lack of random assignment by flexibly controlling for a rich set of variables. In the third, we revisit Acemoglu et al. (2001) IV estimation of the effects of institutions on economic growth by estimating a partially linear IV model.

# 6.1. The effect of Unemployment Insurance Bonus on Unemployment Duration

In this example, we re-analyze the Pennsylvania Reemployment Bonus experiment which was conducted by the US Department of Labor in the 1980s to test the incentive effects of alternative compensation schemes for unemployment insurance (UI). This experiment has been previously studied by Bilias (2000) and Bilias and Koenker (2002). In these experiments, UI claimants were randomly assigned either to a control group or one of five treatment groups.12 In the control group, the standard rules of the UI system applied. Individuals in the treatment groups were offered a cash bonus if they found a job within some pre-specified period of time (qualification period), provided that the job was retained for a specified duration. The treatments differed in the level of the bonus, the length of the qualification period, and whether the bonus was declining over time in the qualification period; see Bilias and Koenker (2002) for further details.

In our empirical example, we focus only on the most generous compensation scheme, treatment 4, and drop all individuals who received other treatments. In this treatment, the bonus amount is high and the qualification period is long compared to other treatments, and claimants are eligible to enroll in a workshop. Our treatment variable, D, is an indicator variable for being assigned treatment 4, and the outcome variable, Y, is the log of duration of unemployment for the UI claimants. The vector of covariates, X, consists of age group dummies, gender, race, the number of dependents, quarter of the experiment, location within the state, existence of recall expectations, and type of occupation.

We report results based on five simple methods for estimating the nuisance functions used in forming the orthogonal estimating equations. We consider three tree-based methods, labeled “Random Forest” “Reg. Tree”, and “Boosting”, one $\ell _ { 1 }$ -penalization based method, labeled “Lasso”, and a neural network method, labeled “Neural Net”. For “Reg. Tree,” we fit a single CART tree to estimate each nuisance function with penalty parameter chosen by 10-fold cross-validation. The results in the “Random Forest” column are obtained by estimating each nuisance function with a random forest which averages over 1000 trees. The results in “Boosting” are obtained using boosted regression trees with regularization parameters chosen by 10-fold cross-validation. To estimate the nuisance functions using the neural networks, we use 2 neurons and a decay parameter of 0.02, and we set activation function as logistic for classification problems and as linear for regression problems. 13 “Lasso” estimates an $\ell _ { 1 }$ -penalized linear regression model using the data-driven penalty parameter selection rule developed in Belloni et al. (2012). For “Lasso”, we use a set of 96 potential control variables formed from the raw set of covariates and all second order terms, i.e. all squares and first-order interactions. For the remaining methods, we use the raw set of covariates as features.

We also consider two hybrid methods labeled “Ensemble” and “Best”. “Ensemble” optimally combines four of the ML methods listed above by estimating the nuisance functions as weighted averages of estimates from “Lasso,” “Boosting,” “Random Forest,” and “Neural Net”. The weights are restricted to sum to one and are chosen so that the weighted average of these methods gives the lowest average mean squared out-of-sample prediction error estimated using 5-fold cross-validation. The final column in Table 1 (“Best”) reports results that combine the methods in a different way. After obtaining estimates from the five simple methods and “Ensemble” we select the best methods for estimating each nuisance functions based on the average out-of-sample prediction performance for the target variable associated with each nuisance function obtained from each of the previously described approaches. As a result, the reported estimate in the last column uses different ML methods to estimate different nuisance functions. Note that if a single method outperformed all the others in terms of prediction accuracy for all nuisance functions, the estimate in the “Best” column would be identical to the estimate reported under that method.

Table 1 presents DML2 estimates of the ATE on unemployment duration using the median method described in Section 3.4. We report results for heterogeneous effect model in Panel A and for the partially linear model in Panel B. Because the treatment is randomly assigned, we use the fraction of treated as the estimator of the propensity score in forming the orthogonal estimating equations.14 For both the partially linear model and the interactive model, we report estimates obtained using 2-fold cross-fitting and 5-fold cross-fitting. All results are based on taking 100 different sample splits. We summarize results across the sample splits using the median method. For comparison, we report two different standard errors. In brackets, we report the median standard error from across the 100 splits; and we report standard errors adjusted for variability across the sample splits using the median method in parentheses.

The estimation results are consistent with the findings of previous studies which have analyzed the Pennsylvania Bonus Experiment. The ATE on unemployment duration is negative and significant across all estimation methods at the 5% level regardless of the standard error estimator used. Interestingly, we see that there is no practical difference across the two different standard errors in this example.

Table 1. Estimated Effect of Cash Bonus on Unemployment Duration
![](/tmp/mineru_hhuwcznw/images/064132254e38c9b375a55ced8ea67ea0c8ceb1491c17924a75c00ceedc6b523b.jpg)

Note: Estimated ATE and standard errors from a linear model (Panel B) and heterogeneous effect model (Panel A) based on orthogonal estimating equations. Column labels denote the method used to estimate nuisance functions. Results are based on 100 splits with point estimates calculated the median method. The median standard error across the splits are reported in brackets and standard errors calculated using the median method to adjust for variation across splits are provided in parentheses. Further details about the methods are provided in the main text.

# 6.2. The effect of 401(k) Eligibility and Participation on Net Financial Assets

The key problem in determining the effect of 401(k) eligibility is that working for a firm that offers access to a 401(k) plan is not randomly assigned. To overcome the lack of random assignment, we follow the strategy developed in Poterba et al. (1994a) and Poterba et al. (1994b). In these papers, the authors use data from the 1991 Survey of Income and Program Participation and argue that eligibility for enrolling in a 401(k) plan in this data can be taken as exogenous after conditioning on a few observables of which the most important for their argument is income. The basic idea of their argument is that, at least around the time 401(k) initially became available, people were unlikely to be basing their employment decisions on whether an employer offered a 401(k) but would instead focus on income and other aspects of the job. Following this argument, whether one is eligible for a 401(k) may then be taken as exogenous after appropriately conditioning on income and other control variables related to job choice.

A key component of the argument underlying the exogeneity of $4 0 1 ( \mathrm { k } )$ eligibility is that eligibility may only be taken as exogenous after conditioning on income and other variables related to job choice that may correlate with whether a firm offers a $4 0 1 ( \mathrm { k } )$ . Poterba et al. (1994a) and Poterba et al. (1994b) and many subsequent papers adopt this argument but control only linearly for a small number of terms. One might wonder whether such specifications are able to adequately control for income and other related confounds. At the same time, the power to learn about treatment effects decreases as one allows more flexible models. The principled use of flexible ML tools offers one resolution to this tension. The results presented below thus complement previous results which rely

Table 2. Estimated Effect of 401(k) Eligibility on Net Financial Assets
![](/tmp/mineru_hhuwcznw/images/e8cbfdd34ff68fa21756337e4ee4adf7adcd363c83913f72b73fdd34b3f41719.jpg)

Note: Estimated ATE and standard errors from a linear model (Panel B) and heterogeneous effect model (Panel A) based on orthogonal estimating equations. Column labels denote the method used to estimate nuisance functions. Results are based on 100 splits with point estimates calculated the median method. The median standard error across the splits are reported in brackets and standard errors calculated using the median method to adjust for variation across splits are provided in parentheses. Further details about the methods are provided in the main text.

on the assumption that confounding effects can adequately be controlled for by a small number of variables chosen ex ante by the researcher.

In the example in this paper, we use the same data as in Chernozhukov and Hansen (2004). We use net financial assets - defined as the sum of IRA balances, 401(k) balances, checking accounts, U.S. saving bonds, other interest-earning accounts in banks and other financial institutions, other interest-earning assets (such as bonds held personally), stocks, and mutual funds less non-mortgage debt - as the outcome variable, $Y$ , in our analysis. Our treatment variable, $D$ , is an indicator for being eligible to enroll in a 401(k) plan. The vector of raw covariates, $X$ , consists of age, income, family size, years of education, a married indicator, a two-earner status indicator, a defined benefit pension status indicator, an IRA participation indicator, and a home ownership indicator.

In Table 2, we report DML2 estimates of ATE of 401(k) eligibility on net financial assets both in the partially linear model as in (1.1) and allowing for heterogeneous treatment effects using the interactive model outlined in Section 5.1. To reduce the disproportionate impact of extreme propensity score weights in the interactive model, we trim the propensity scores at 0.01 and 0.99. We present two sets of results based on sample-splitting as discussed in Section 3 using 2-fold cross -fitting and 5-fold cross-fitting. As in the previous section, we consider 100 different sample partitions and summarize the results across different sample splits using the median method. For comparison, we report two different standard errors. In brackets, we report the median standard error from across the 100 splits; and we report standard errors adjusted for variability across the sample splits using the median method in parentheses. We consider the same methods with the same tuning choices for estimating the nuisance functions as in the previous example, with one exception, and so do not repeat details for brevity. The one exception is that

Table 3. Estimated Effect of 401(k) Participation on Net Financial Assets
![](/tmp/mineru_hhuwcznw/images/124147fd8ef76dea44b89dc73de958e1f4d6323685dcf47ce58e4015f7964faa.jpg)

Note: Estimated LATE based on orthogonal estimating equations. Column labels denote the method used to estimate nuisance functions. Results are based on 100 splits with point estimates calculated the median method. The median standard error across the splits are reported in brackets and standard errors calculated using the median method to adjust for variation across splits are provided in parentheses. Further details about the methods are provided in the main text.

we implement neural networks with 8 neurons and a decay parameter of 0.01 in this example.

Turning to the results, it is first worth noting that the estimated ATE of 401(k) eligibility on net financial assets is \$19,559 with an estimated standard error of 1413 when no control variables are used. Of course, this number is not a valid estimate of the causal effect of $4 0 1 ( \mathrm { k } )$ eligibility on financial assets if there are neglected confounding variables as suggested by Poterba et al. (1994a) and Poterba et al. (1994b). When we turn to the estimates that flexibly account for confounding reported in Table 2, we see that they are substantially attenuated relative to this baseline that does not account for confounding, suggesting much smaller causal effects of 401(k) eligibility on financial asset holdings. It is interesting and reassuring that the results obtained from the different flexible methods are broadly consistent with each other. This similarity is consistent with the theory that suggests that results obtained through the use of orthogonal estimating equations and any sensible method of estimating the necessary nuisance functions should be similar. Finally, it is interesting that these results are also broadly consistent with those reported in the original work of Poterba et al. (1994a) and Poterba et al. (1994b) which used a simple intuitively motivated functional form, suggesting that this intuitive choice was sufficiently flexible to capture much of the confounding variation in this example.

As a further illustration, we also report the LATE in this example where we take the endogenous treatment variable to be participating in a 401(k) plan. Even after controlling for features related to job choice, it seems likely that the actual choice of whether to participate in an offered plan would be endogenous. Of course, we can use eligibility for a 401(k) plan as an instrument for participation in a 401(k) plan under the conditions that were used to justify the exogeneity of eligibility for a 401(k) plan provided above in the discussion of estimation of the ATE of 401(k) eligibility.

We report DML2 results of estimating the LATE of 401(k) participation using 401(k) eligibility as an instrument in Table 3. We employ the procedure outlined in Section 5.2 using the same ML estimators to estimate the quantities used to form the orthogonal estimating equation as we employed to estimate the ATE of 401(k) eligibility outlined previously, so we omit the details for brevity. Looking at the results, we see that the estimated causal effect of 401(k) participation on net financial assets is uniformly positive and statistically significant across all of the considered methods. As when looking at the ATE of 401(k) eligibility, it is reassuring that the results obtained from the different flexible methods are broadly consistent with each other. It is also interesting that the results based on flexible ML methods are broadly consistent with, though somewhat attenuated relative to, those obtained by applying the same specification for controls as used in Poterba et al. (1994a) and Poterba et al. (1994b) and using a linear IV model which returns an estimated effect of participation of $\ S$ 13,102 with estimated standard error of (1922). The mild attenuation may suggest that the simple intuitive control specification used in the original baseline specification is somewhat too simplistic.

Looking at Tables 2 and 3, there are other interesting observations that can provide useful insights into understanding the finite sample properties of the DML estimation method. First, the standard errors of the estimates obtained using 5-fold cross-fitting are lower than those obtained from 2-fold cross-fitting for all methods across all cases. This fact suggests that having more observations in the auxiliary sample may be desirable. Specifically, the 5-fold cross-fitting estimates use more observations to learn the nuisance functions than 2-fold cross-fitting and thus likely learn them more precisely. This increase in precision in learning the nuisance functions may then translate into more precisely estimated parameters of interest. While intuitive, we note that this statement does not seem to be generalizable in that there does not appear to be a general relationship between the number of folds in cross-fitting and the precision of the estimate of the parameter of interest; see the next example. Second, we also see that the standard errors of the Lasso estimates after adjusting for variation due to sample splitting are noticeably larger than the standard errors coming from the other ML methods. We believe that this is due to the fact that the out-of-sample prediction errors from a linear model tend to be larger when there is a need to extrapolate. In our framework, if the main sample includes observations that are outside of the range of the observations in the auxiliary sample, the model has to extrapolate to those observations. The fact that the standard errors are lower in 5-fold cross-fitting than in 2-fold cross-fitting for the “Lasso” estimations also supports this hypothesis, because the higher number of observations in the auxiliary sample reduces the degree of extrapolation. We also see that there is a noticeable increase in the standard errors that account for variability due to sample splitting relative to the simple unadjusted standard errors in this case, though these differences do not qualitatively change the results.

# 6.3. The Effect of Institutions on Economic Growth

To demonstrate DML estimation of partially linear structural equation models with instrumental variables, we consider estimation of the effect of institutions on aggregate output following the work of Acemoglu et al. (2001) (AJR). Estimating the effect of institutions on output is complicated by the clear potential for simultaneity between institutions and output: Specifically, better institutions may lead to higher incomes, but higher incomes may also lead to the development of better institutions. To help overcome this simultaneity, AJR use mortality rates for early European settlers as an instrument for institution quality. The validity of this instrument hinges on the argument that settlers set up better institutions in places where they are more likely to establish long-term settlements; that where they are likely to settle for the long term is related to settler mortality at the time of initial colonization; and that institutions are highly persistent. The exclusion restriction for the instrumental variable is then motivated by the argument that GDP, while persistent, is unlikely to be strongly influenced by mortality in the previous century, or earlier, except through institutions.

Table 4. Estimated Effect of Institutions on Output
![](/tmp/mineru_hhuwcznw/images/f0dfd265728b35a486a5753bd52293a1e72030b7aa1cb2f792f21900855a0b49.jpg)

Note: Estimated coefficient from a linear instrumental variables model based on orthogonal estimating equations. Column labels denote the method used to estimate nuisance functions. Results are based on 100 splits with point estimates calculated the median method. The median standard error across the splits are reported in brackets and standard errors calculated using the median method to adjust for variation across splits are provided in parentheses. Further details about the methods are provided in the main text.

In their paper, AJR note that their instrumental variable strategy will be invalidated if other factors are also highly persistent and related to the development of institutions within a country and to the country’s GDP. A leading candidate for such a factor, as they discuss, is geography. AJR address this by assuming that the confounding effect of geography is adequately captured by a linear term in distance from the equator and a set of continent dummy variables. Using DML allows us to relax this assumption and replace it by a weaker assumption that geography can be sufficiently controlled by an unknown function of distance from the equator and continent dummies which can be learned by ML methods.

We use the same set of 64 country-level observations as AJR. The data set contains measurements of GDP, settler morality, an index which measures protection against expropriation risk and geographic information. The outcome variable, Y, is the logarithm of GDP per capita and the endogenous explanatory variable, D, is a measure of the strength of individual property rights that is used as a proxy for the strength of institutions. To deal with endogeneity, we use an instrumental variable Z, which is mortality rates for early European settlers. Our raw set of control variables, X, include distance from the equator and dummy variables for Africa, Asia, North America, and South America.

We report results from applying DML2 following the procedure outlined in Section 4.2 in Table 4. The considered ML methods and tuning parameters are the same as the previous examples except for the Ensemble method, from which we exclude Neural Network since the small sample size causes stability problems in training the Neural Network. We use the raw set of covariates and all second order terms when doing lasso estimation, and we simply use the raw set of covariates in the remaining methods. As in the previous examples, we consider 100 different sample splits and report the “Median” estimates of the coefficient and two different standard error estimates. In brackets, we report the median standard error from across the 100 splits; and we report standard errors adjusted for variability across the sample splits using the median method in parentheses. Finally, we report results from both 2-fold cross -fitting and 5-fold cross -fitting as in the other examples.

In this example, we see uniformly large and positive point estimates across all procedures considered, and estimated effects are statistically significant at the 5% level. As in the second example, we see that adjusting for variability across sample splits leads to noticeable increases in estimated standard errors but does not result in qualitatively different conclusions. Interestingly, we see that the estimated standard errors based on 5-fold cross- fitting are larger than those on twofold cross-fitting in all procedures except lasso, which differs from the finding in the 401(k) example. Further understanding these differences and the impact of the number of folds on inference for objects of interest seems like an interesting question for future research. Finally, although the estimates are somewhat smaller than the baseline estimates reported in AJR an estimated coefficient of 1.10 with estimated standard error of 0.46 (Acemoglu et al. (2001), Table 4, Panel A, column 7) - the results are qualitatively similar, indicating a strong and positive effect of institutions on output.

# 6.4. Comments on Empirical Results

Before closing this section we want to emphasize some important conclusions that can be drawn from these empirical examples. First, the choice of the ML method used in estimating nuisance functions does not substantively change the conclusion in any of the examples, and we obtained broadly consistent results regardless of which method we employ. The robustness of the results to the different methods is implied by the theory assuming that all of the employed methods are able to deliver sufficiently highquality approximations to the underlying nuisance functions. Second, the incorporation of uncertainty due to sample-splitting using the median method increases the standard errors relative to a baseline that does not account for this uncertainty, though these differences do not alter the main results in any of the examples. This lack of variation suggests that the parameter estimates are robust to the particular sample split used in the estimation in these examples.

# ACKNOWLEDGEMENTS

We would like to acknowledge research support from the National Science Foundation. We also thank participants of the MIT Stochastics and Statistics seminar, the Kansas Econometrics conference, the Royal Economic Society Annual Conference, The Hannan Lecture at the Australasian Econometric Society meeting, The Econometric Theory lecture at the $E C ^ { 2 }$ meetings 2016 in Toulouse, The CORE 50th Anniversary Conference, The Becker-Friedman Institute Conference on Machine Learning and Economics, The INET conferences at USC on Big Data, the World Congress of Probability and Statistics 2016, the Joint Statistical Meetings 2016, the New England Day of Statistics Conference, CEMMAP’s Masterclass on Causal Machine Learning, and St. Gallen’s summer school on “Big Data”, for many useful comments and questions. We would like to thank Susan Athey, Peter Aronow, Jin Hahn, Guido Imbens, Mark van der Laan, Matt Taddy, and Rui Wang for constructive comments. We thank Peter Aronow for pointing us to the literature on targeted learning on which, along with prior works of Neyman, Bickel, and the many other contributions to semiparametric learning theory, we build.

# REFERENCES

Abadie, A. and G. W. Imbens (2006). Large sample properties of matching estimators for average treatment effects. Econometrica 74, 235–267. Acemoglu, D., S. Johnson, and J. A. Robinson (2001). The colonial origins of comparative development: An empirical investigation. American Economic Review 91, 1369–1401.

Ai, C. and X. Chen (2012). The semiparametric efficiency bound for models of sequential moment restrictions containing unknown functions. Journal of Econometrics 170, 442– 457.
Andrews, D. W. K. (1994a). Asymptotics for semiparametric econometric models via stochastic equicontinuity. Econometrica 62, 43–72.
Andrews, D. W. K. (1994b). Empirical process methods in econometrics. Handbook of Econometrics, Volume IV, Chapter 37 , 2247–2294.
Angrist, J. D. and A. B. Krueger (1995). Split-sample instrumental variables estimates of the return to schooling. Journal of Business and Economic Statistics 13, 225–235.
Athey, S., G. Imbens, and S. Wager (2016). Approximate residual balancing: De-biased inference of average treatment effects in high-dimensions. arXiv:1604.07125v3 . arXiv, 2016.
Ayyagari, R. (2010). Applications of influence functions to semiparametric regression models. Ph.D. Thesis, Harvard School of Public Health, Harvard University.
Belloni, A., D. Chen, V. Chernozhukov, and C. Hansen (2012). Sparse models and methods for optimal instruments with an application to eminent domain. Econometrica 80, 2369–2429. arXiv, 2010.
Belloni, A. and V. Chernozhukov (2011). $\ell _ { 1 }$ -penalized quantile regression for high dimensional sparse models. Annals of Statistics 39, 82–130. arXiv, 2009.
Belloni, A. and V. Chernozhukov (2013). Least squares after model selection in highdimensional sparse models. Bernoulli 19, 521–547. arXiv, 2009.
Belloni, A., V. Chernozhukov, I. Fern´andez-Val, and C. Hansen (2017). Program evaluation with high-dimensional data. Econometrica 85, 233— -298. arXiv, 2013.
Belloni, A., V. Chernozhukov, and C. Hansen (2010). Lasso methods for gaussian instrumental variables models. arXiv:1012.1297 .
Belloni, A., V. Chernozhukov, and C. Hansen (2013). Inference for high-dimensional sparse econometric models. Advances in Economics and Econometrics. 10th World Congress of Econometric Society. August 2010 , III:245– 295. arXiv, 2011.
Belloni, A., V. Chernozhukov, and C. Hansen (2014). Inference on treatment effects after selection amongst high-dimensional controls. Review of Economic Studies 81, 608– 650. arXiv, 2011.
Belloni, A., V. Chernozhukov, and K. Kato (2015). Uniform post selection inference for lad regression models and other z-estimators. Biometrika 102, 77–94. arXiv, 2013.
Belloni, A., V. Chernozhukov, and L. Wang (2011). Square-root-lasso: Pivotal recovery of sparse signals via conic programming. Biometrika 98, 791–806. arXiv, 2010.
Belloni, A., V. Chernozhukov, and L. Wang (2014). Pivotal estimation via square-root lasso in nonparametric regression. Annals of Statistics 42, 757– 788. arXiv, 2011.
Belloni, A., V. Chernozhukov, and Y. Wei (2016). Post-selection inference for generalized linear models with many controls. Journal of Business and Economic Statistics 34, 606–619. arXiv, 2013.
Bera, A., G. Montes-Rojas, and W. Sosa-Escudero (2010). General specification testing with locally misspecified models. Econometric Theory 26, 1838–1845.
Bickel, P. and Y. Ritov (1988). Estimating integrated squared density derivatives. Sankhya A-50, 381– 393.
Bickel, P. J. (1982). On adaptive estimation. Annals of Statistics 10, 647–671.
Bickel, P. J., C. A. J. Klaassen, Y. Ritov, and J. A. Wellner (1998). Efficient and Adaptive Estimation for Semiparametric Models. Springer.
Bickel, P. J., Y. Ritov, and A. Tsybakov (2009). Simultaneous analysis of Lasso and Dantzig selector. Annals of Statistics 37, 1705–1732. arXiv, 2008.
Bilias, (2000). Sequential testing of duration data: The case of the Pennsylvania ‘reemployment bonus’ experiment. Journal of Applied Econometrics 15, 575–594.
Bilias, Y. and R. Koenker (2002). Quantile regression for duration data: A reappraisal of the pennsylvania reemployment bonus experiments. In B. Fitzenberger, R. Koenker, and J. A. Machado (Eds.), Studies in Empirical Economics: Economic Applications of Quantile Regression, pp. 199– 220. Physica-Verlag Heidelberg.
Bu¨hlmann, P. and S. van de Geer (2011). Statistics for High-Dimensional Data. Springer Series in Statistics.
Chamberlain, G. (1987). Asymptotic efficiency in estimation with conditional moment restrictions. Journal of Econometrics 34, 305–334.
Chamberlain, G. (1992). Efficiency bounds for semiparametric regression. Econometrica 60, 567–596.
Chen, X., O. Linton, and I. van Keilegom (2003). Estimation of semiparametric models when the criterion function is not smooth. Econometrica 71, 1591– 1608.
Chen, X. and H. White (1999). Improved rates and asymptotic normality for nonparametric neural network estimators. IEEE Transactions on Information Theory 45, 682– 691.
Chernozhukov, V., D. Chetverikov, and K. Kato (2014). Gaussian approximation of suprema of empirical processes. The Annals of Statistics 42, 1564–1597. arXiv, 2012.
Chernozhukov, V., J. Escanciano, H. Ichimura, W. Newey, and J. Robins (2016). Locally robust semiparametric estimation. arXiv:1608.00033 . arXiv, 2016.
Chernozhukov, V. and C. Hansen (2004). The effects of 401 (k) participation on the wealth distribution: an instrumental quantile regression analysis. Review of Economics and statistics 86, 735–751.
Chernozhukov, V., C. Hansen, and M. Spindler (2015a). Post-selection and postregularization inference in linear models with very many controls and instruments. Americal Economic Review: Papers and Proceedings 105, 486– 490.
Chernozhukov, V., C. Hansen, and M. Spindler (2015b). Valid post-selection and postregularization inference: An elementary, general approach. Annual Review of Economics 7, 649–688.
DasGupta, A. (2008). Asymptotic Theory of Statistics and Probability. Springer Texts in Statistics.
Fan, J., S. Guo, and K. Yu (2012). Variance estimation using refitted cross-validation in ultrahigh dimensional regression. Journal of the Royal Statistical Society, Series B 74, 37–65.
Farrell, M. (2015). Robust inference on average treatment effects with possibly more covariates than observations. Journal of Econometrics 174, 1–23.
Ferguson, T. (1967). Mathematical Statistics: A Decision Theoretic Approach. Academic Press.
Fr¨olich, M. (2007). Nonparametric IV estimation of local average treatment effects with covariates. Journal of Econometrics 139, 35– 75.
Gautier, E. and A. Tsybakov (2014). High-dimensional instrumental variables regression and confidence sets. arXiv:1105.2454 . arXiv, 2011.
Hahn, J. (1998). On the role of the propensity score in efficient semiparametric estimation of average treatment effects. Econometrica 66, 315– 331.
Hansen, L. (1982). Large sample properties of generalized method of moments estimators. Econometrica 50, 1029– 1054.
Hasminskii, R. and I. Ibragimov (1978). On the nonparametric estimation of functionals. Proceedings of the 2nd Prague Symposium on Asymptotic Statistics, 41– 51.
Hirano, K., G. W. Imbens, and G. Ridder (2003). Efficient estimation of average treatment effects using the estimated propensity score. Econometrica 71, 1161– 1189.
Hubbard, A. E., S. Kherad-Pajouh, and M. J. van der Laan (2016). Statistical inference for data adaptive target parameters. International Journal of Biostatistics 12, 3–19.
Ibragimov, I. A. and R. Z. Hasminskii (1981). Statistical Estimation: Asymptotic Theory. Springer-Verlag, New York.
Ichimura, H. and W. Newey (2015). The influence function of semiparametric estimators. arXiv:1508.01378 . arXiv, 2015.
Imai, K. and M. Ratkovic (2013). Estimating treatment effect heterogeneity in randomized program evaluation. Annals of Applied Statistics 7, 443–470.
Imbens, G. and J. Angrist (1994). Identification and estimation of local average treatment effects. Econometrica, 467– 475.
Imbens, G. W. and D. B. Rubin (2015). Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction. Cambridge University Press.
Javanmard, A. and A. Montanari (2014a). Confidence intervals and hypothesis testing for high-dimensional regression. Journal of Machine Learning Research 15, 2869–2909.
Javanmard, A. and A. Montanari (2014b). Hypothesis testing in high-dimensional regression under the gaussian random design model: Asymptotic theory. IEEE Transactions on Information Theory 60, 6522–6554. arXiv, 2013.
Kozbur, D. (2016). Testing-based forward model selection. arXiv:1512.02666 . arXiv, 2015.
Lee, L. (2005). A $c ( \alpha )$ -type gradient test in the GMM approach. Working paper, The Ohio State University.
Levit, B. Y. (1975). On the efficiency of a class of nonparametric estimates. Theory of Probability and Its Applications 20, 723–740.
Linton, O. (1996). Edgeworth approximation for MINPIN estimators in semiparametric regression models. Econometric Theory 12, 30–60.
Luedtke, A. R. and M. J. van der Laan (2016). Optimal individualized treatments in resource-limited settings. The International Journal of Biostatistics 12, 283– 303.
Luo, Y. and M. Spindler (2016). High-dimensional $l _ { 2 }$ boosting: Rate of convergence. arXiv:1602.08927 . arXiv, 2016.
Nevelson, M. (1977). On one informational lower bound. Problemy Peredachi Informatsii 13, 26– 31.
Newey, W. (1990). Semiparametric efficiency bounds. Journal of Applied Econometrics 5, 99–135.
Newey, W. (1994). The asymptotic variance of semiparametric estimators. Econometrica 62, 1349– 1382.
Newey, W. K., F. Hsieh, and J. Robins (1998). Undersmoothing and bias corrected functional estimation. Working paper, MIT Economics Dept., http://economics.mit.edu/files/11219.
Newey, W. K., F. Hsieh, and J. M. Robins (2004). Twicing kernels and a small bias property of semiparametric estimators. Econometrica 72, 947–962.
Neyman, J. (1959). Optimal asymptotic tests of composite statistical hypotheses. In U. Grenander (Ed.), Probability and Statistics, pp. 416— -444. New York, John Wiley.
Neyman, J. (1979). $c ( \alpha )$ tests and their use. Sankhya, 1–21.
Poterba, J. M., S. F. Venti, and D. A. Wise (1994a). 401(k) plans and tax-deferred savings. In D. Wise (Ed.), Studies in the Economics of Aging, pp. 105–142. Chicago: University of Chicago Press.
Poterba, J. M., S. F. Venti, and D. A. Wise (1994b). Do 401(k) cont ibutions crowd out other personal saving? Journal of Public Economics 58, 1–32.
Robins, J., L. Li, R. Mukherjee, E. Tchetgen, and A. van der Vaart (2017). Minimax estimation of a functional on a structured high dimensional model. Annals of Statistics, forthcoming.
Robins, J., L. Li, E. Tchetgen, and A. van der Vaart (2008). Higher order influence functions and minimax estimation of nonlinear functionals. In D. Nolan and T. Speed (Eds.), Probability and Statistics: Essays in Honor of David A. Freedman, pp. 335–421. Institute of Mathematical Statistics.
Robins, J. and A. Rotnitzky (1995). Semiparametric efficiency in multivariate regression models with missing data. Journal of the American Statistical Association 90, 122– 129.
Robins, J., P. Zhang, R. Ayyagari, R. Logan, E. Tchetgen, L. Li, A. Lumley, and A. van der Vaart (2013). New statistical approaches to semiparametric regression with application to air pollution research. Research Report 175, Health Effects Institute.
Robinson, P. M. (1988). Root- $N$ -consistent semiparametric regression. Econometrica 56, 931– 954.
Rosenbaum, P. R. and D. B. Rubin (1983). The central role of the propensity score in observational studies for causal effects. Biometrika 70, 41–55.
Scharfstein, D. O., A. Rotnitzky, and J. M. Robins (1999). Rejoinder to “adjusting for non-ignorable drop-out using semiparametric non-response models”. Journal of the American Statistical Association 94, 1135–1146.
Schick, A. (1986). On asymptotically efficient estimation in semiparametric models. Annals of Statistics 14, 1139– 1151.
Severini, T. A. and W. H. Wong (1992). Profile likelihood and conditionally parametric models. The Annals of Statistics 20, 1768–1802.
Toth, B. and M. J. van der Laan (2016). TMLE for marginal structural models based on an instrument. Working Paper 350, U.C. Berkeley Division of Biostatistics Working Paper Series.
van de Geer, S., P. B¨uhlmann, Y. Ritov, and R. Dezeure (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. Annals of Statistics 42, 1166–1202. arXiv, 2013.
van der Laan, M. and D. Rubin (2006). Targeted maximum likelihood learning. Working Paper 213, UC Berkeley Division of Biostatistics Working Paper Series.
van der Laan, M. J. (2015). A generally efficient targeted minimum loss based estimator. Working Paper 343, U.C. Berkeley Division of Biostatistics Working Paper Series.
van der Laan, M. J., E. C. Polley, and A. E. Hubbard (2007). Super learner. Statistical Applications in Genetics and Molecular Biology 6. Retrieved 24 Feb. 2017, from doi:10.2202/1544-6115.1309.
van der Laan, M. J. and S. Rose (2011). Targeted Learning: Causal Inference for Observational and Experimental Data. Springer.
van der Vaart, A. W. (1991). On differentiable functionals. Annals of Statistics 19, 178–204.
van der Vaart, A. W. (1998). Asymptotic Statistics. Cambridge University Press.
Wager, S. and G. Walther (2016). Adaptive concentration of regression trees, with application to random forests. arXiv:1503.06388 . arXiv, 2015.
Wooldridge, J. (1991). Specification testing and quasi-maximum-likelihood estimation. Journal of Econometrics 48, 29– 55.
Zhang, C. and S. Zhang (2014). Confidence intervals for low-dimensional parameters with high-dimensional data. Journal of the Royal Statistical Society, Series B 76, 217–242. arXiv, 2012.
Zheng, W., Z. Luo, and M. J. van der Laan (2016). Marginal structural models with counterfactual effect modifiers. Working Paper 348, U.C. Berkeley Division of Biostatistics Working Paper Series.
Zheng, W. and M. J. van der Laan (2011). Cross-validated targeted minimum-loss-based estimation. In Targeted Learning, pp. 459–474. Springer.

# APPENDIX: PROOFS OF RESULTS

In this appendix, we use $C$ to denote a strictly positive constant that is independent of $n$ and $P \in \mathcal { P } _ { N }$ . The value of $C$ may change at each appearance. Also, the notation $a _ { N } \lesssim b _ { N }$ means that $a _ { N } \leqslant C b _ { N }$ for all $n$ and some $C$ . The notation $a _ { N } \gtrsim b _ { N }$ means that $b _ { N } \lesssim a _ { N }$ . Moreover, the notation $a _ { N } = o ( 1 )$ means that there exists a sequence $\left( b _ { N } \right) _ { n \geqslant 1 }$ of positive numbers such that (a) $| a _ { N } | \leqslant b _ { N }$ for all $n$ , (b) $b _ { N }$ is independent of $P \in \mathcal { P } _ { N }$ for all $n$ , and (c) $b _ { N }  0$ as $n \to \infty$ . Finally, the notation $a _ { N } = O _ { P } ( b _ { N } )$ means that for all $\epsilon > 0$ , there exists $C$ such that $\mathrm { P } _ { P } ( a _ { N } > C b _ { N } ) \leqslant 1 - \epsilon$ for all $n$ . Using this notation allows us to avoid repeating “uniformly over $P \in \mathcal P _ { N }$ ” many times in the proofs.

Define the empirical process $\mathbb { G } _ { n } ( \psi ( W ) )$ as a linear operator acting on measurable functions $\psi : \mathcal { W } \to \mathbb { R }$ such that $\| \psi \| _ { P , 2 } < \infty$ via,

$$
\mathbb { G } _ { n } \big ( \psi ( W ) \big ) : = \mathbb { G } _ { n , I } \big ( \psi ( W ) \big ) : = \frac { 1 } { \sqrt { n } } \sum _ { i \in I } \psi ( W _ { i } ) - \int \psi ( w ) d P ( w ) .
$$

Analogously, we defined the empirical expectation as:

$$
\mathbb { E } _ { n } ( \psi ( W ) ) : = \mathbb { E } _ { n , I } ( \psi ( W ) ) : = \frac { 1 } { n } \sum _ { i \in I } \psi ( W _ { i } ) .
$$

# A.5. Useful Lemmas

The following lemma is useful particularly in the sample-splitting contexts.

Lemma 6.1. (Conditional Convergence Implies Unconditional) Let $\{ X _ { m } \}$ and $\{ Y _ { m } \}$ be sequences of random vectors. (a) If for $\epsilon _ { m } \to 0$ , $\operatorname { P } ( \| X _ { m } \| > \epsilon _ { m } \mid Y _ { m } )  _ { \mathrm { P } } 0$ , then $\mathrm { P } ( \| X _ { m } \| > \epsilon _ { m } )  0$ . In particular, this occurs if $\operatorname { E } [ \| X _ { m } \| ^ { q } / \epsilon _ { m } ^ { q } \ \mid \ Y _ { m } ] \ \to _ { \mathrm { P } } \ 0$ for some $q \geqslant 1$ , by Markov’s inequality. (b) Let $\{ A _ { m } \}$ be a sequence of positive constants. If $\| X _ { m } \| = O _ { P } ( A _ { m } )$ conditional on $Y _ { m }$ , namely, that for any $\ell _ { m } \to \infty$ , $\operatorname { P } ( \| X _ { m } \| > \ell _ { m } A _ { m } \ |$ $Y _ { m } ) \  _ { \mathrm { { P } } } 0$ , then $\| X _ { m } \| = O _ { P } ( A _ { m } )$ unconditionally, namely, that for any $\ell _ { m } \to \infty$ , $\operatorname { P } ( \| X _ { m } \| > \ell _ { m } A _ { m } )  0$ .

Proof. Part (a). For any $\epsilon > 0$ $\begin{array} { r } { \mathrm { ~ P ~ } ( \| X _ { m } \| > \epsilon _ { m } ) \leqslant \mathrm { E } [ \mathrm { P } ( \| X _ { m } \| > \epsilon _ { m } \mid Y _ { m } ) ]  0 } \end{array}$ , since the sequence $\{ \operatorname { P } ( \| X _ { m } \| > \epsilon _ { m } \mid Y _ { m } ) \}$ is uniformly integrable. To show the second part note that $\mathrm { P } ( \| X _ { m } \| > \epsilon _ { m } \mid Y _ { m } ) \leqslant \mathrm { E } [ \| X _ { m } \| ^ { q } / \epsilon _ { m } ^ { q } \mid Y _ { m } ] \vee 1  _ { P } 0$ by Markov’s inequality. Part (b). This follows from Part (a).

Let $( W _ { i } ) _ { i = 1 } ^ { n }$ be a sequence of independent copies of a random element $W$ taking values in a measurable space $( \mathcal { W } , A _ { \mathcal { W } } )$ according to a probability law $P$ . Let $\mathcal { F }$ be a set of suitably measurable functions $f \colon \mathcal { W } \to \mathbb { R }$ , equipped with a measurable envelope $F \colon \mathcal { W } $ $\mathbb { R }$ .

Lemma 6.2. (Maximal Inequality, Chernozhukov et al. (2014)) Work with the setup above. Suppose that $F \geqslant \operatorname* { s u p } _ { f \in { \mathcal { F } } } | f |$ is a measurable envelope for $\mathcal { F }$ with $\| F \| _ { P , q } <$ $\infty$ for some $q \geqslant 2$ . Let $M = \operatorname* { m a x } _ { i \leqslant n } F ( W _ { i } )$ and $\sigma ^ { 2 } > 0$ be any positive constant such that $\begin{array} { r } { \operatorname* { s u p } _ { f \in \mathcal { F } } \| f \| _ { P , 2 } ^ { 2 } \leqslant \sigma ^ { 2 } \leqslant \| F \| _ { P , 2 } ^ { 2 } } \end{array}$ . Suppose that there exist constants $a \geqslant e$ and $v \geqslant 1$ such that

$$
\log \operatorname* { s u p } _ { Q } N ( \epsilon \| F \| _ { Q , 2 } , \mathcal { F } , \| \cdot \| _ { Q , 2 } ) \leqslant v \log ( a / \epsilon ) , \ 0 < \epsilon \leqslant 1 .
$$

Then

$$
\mathrm { E } _ { P } [ \left. \mathbb { G } _ { n } \right. _ { \mathcal { F } } ] \leqslant K \left( \sqrt { v \sigma ^ { 2 } \log \left( \frac { a \| F \| _ { P , 2 } } { \sigma } \right) } + \frac { v \| M \| _ { P , 2 } } { \sqrt { n } } \log \left( \frac { a \| F \| _ { P , 2 } } { \sigma } \right) \right) ,
$$

where $K$ is an absolute constant. Moreover, for every $t \geqslant 1$ , with probability $> 1 - t ^ { - q / 2 }$

$\begin{array} { r } { \| \mathbb { G } _ { n } \| _ { \mathcal { F } } \leqslant ( 1 + \alpha ) \mathbf { E } _ { P } [ \| \mathbb { G } _ { n } \| _ { \mathcal { F } } ] + K ( q ) \left[ ( \sigma + n ^ { - 1 / 2 } \| M \| _ { P , q } ) \sqrt { t } + \alpha ^ { - 1 } n ^ { - 1 / 2 } \| M \| _ { P , 2 } t \right] , \forall \alpha > 0 , } \end{array}$ where $K ( q ) > 0$ is a constant depending only on $q$ . In particular, setting $a \geqslant n$ and $t = \log n$ , with probability $> 1 - c ( \log n ) ^ { - 1 }$

$$
\| { \mathbb { G } } _ { n } \| _ { \mathcal { F } } \leqslant K ( q , c ) \left( \sigma \sqrt { v \log \left( \frac { a \| F \| _ { P , 2 } } { \sigma } \right) } + \frac { v \| M \| _ { P , q } } { \sqrt { n } } \log \left( \frac { a \| F \| _ { P , 2 } } { \sigma } \right) \right) ,
$$

where $\| M \| _ { P , q } \leqslant n ^ { 1 / q } \| F \| _ { P , q }$ and $K ( q , c ) > 0$ is a constant depending only on $q$ and $c$ .

# A.6. Proof of Lemma 2.1

Proof. Since $J$ exists and $J _ { \beta \beta }$ is invertible, (2.8) has the unique solution $\mu _ { 0 }$ given in (2.10), and so we have by (2.6) that $\mathrm { E } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] = 0$ for $\eta _ { 0 }$ given in (2.9). Moreover,

$$
\begin{array} { r } { \partial _ { \eta ^ { \prime } } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) = \left( \left[ J _ { \theta \beta } - \mu _ { 0 } J _ { \beta \beta } \right] , \mathrm { E } [ \partial _ { \beta ^ { \prime } } \ell ( W ; \theta _ { 0 } , \beta _ { 0 } ) ] \otimes I _ { d _ { \theta } \times d _ { \theta } } \right) = 0 , } \end{array}
$$

where Idθ×dθ is the $d _ { \theta } \times d _ { \theta }$ identity matrix and $\otimes$ is the Kronecker product. Hence, the asserted claim holds by the remark after Definition 2.1.

# A.7. Proof of Lemma 2.2

The proof follows similarly to that of Lemma 2.1, except that now we have to verify (2.4) intead of (2.3). To do so, take any $\beta \in B$ such that $\| \beta - \beta _ { 0 } \| _ { q } ^ { * } \leqslant \lambda _ { N } / r _ { N }$ and any $d _ { \theta } \times d _ { \beta }$ matrix $\mu$ . Denote $\eta = ( \beta ^ { \prime } , \mathrm { v e c } ( \mu ) ^ { \prime } ) ^ { \prime }$ . Then

$$
\begin{array} { r l } & { \| \partial _ { \eta } \mathrm { E } _ { P } \psi ( W , \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] \| = \| ( J _ { \theta \beta } - \mu _ { 0 } J _ { \beta \beta } ) ( \beta - \beta _ { 0 } ) \| } \\ & { \qquad \leqslant \| J _ { \theta \beta } - \mu _ { 0 } J _ { \beta \beta } \| _ { q } \times \| \beta - \beta _ { 0 } \| _ { q } ^ { * } \leqslant r _ { n } \times ( \lambda _ { N } / r _ { N } ) = \lambda _ { N } . } \end{array}
$$

This completes the proof of the lemma.

# A.8. Proof of Lemma 2.3

The proof is similar to that of Lemma 2.1, except that now we have

$$
\boldsymbol { \lvert } W , \theta _ { 0 } , \eta _ { 0 } ) = \lvert \mu _ { 0 } G _ { \beta } , \operatorname { E } _ { P } m ( W , \theta _ { 0 } , \beta _ { 0 } ) ^ { \prime } \otimes I _ { d _ { \theta } \times d _ { \theta } } \rvert = 0
$$

where $I _ { d _ { \theta } \times d _ { \theta } }$ is the $d _ { \theta } \times d _ { \theta }$ identity matrix and $\otimes$ is the Kronecker product.

# A.9. Proof of Lemma 2.4

The proof follows similarly to that of Lemma 2.2, except that now for any $\beta \in B$ such that $\| \beta - \beta _ { 0 } \| _ { 1 } \leqslant \lambda _ { N } / r _ { N }$ , any $d _ { \theta } \times k$ matrix $\mu$ , and $\eta = ( \beta ^ { \prime } , \mathrm { v e c } ( \mu ) ^ { \prime } ) ^ { \prime }$ , we have

$$
\begin{array} { r l } & { \left\| \partial _ { \eta } \mathrm { E } _ { P } \psi ( W , \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] \right\| = \| \mu _ { 0 } G _ { \beta } ( \beta - \beta _ { 0 } ) \| } \\ & { \qquad \leqslant \| A ^ { \prime } \Omega ^ { - 1 / 2 } L - \gamma _ { 0 } L ^ { \prime } L \| _ { \infty } \times \| \beta - \beta _ { 0 } \| _ { 1 } } \\ & { \qquad \leqslant r _ { n } \times ( \lambda _ { N } / r _ { N } ) = \lambda _ { N } . } \end{array}
$$

This completes the proof of the lemma.

# A.10. Proof of Lemma 2.5

Take any $\eta \in T$ , and consider the function

$$
Q ( W ; \theta , r ) : = \ell ( W ; \theta , \eta _ { 0 } ( \theta ) + r ( \eta ( \theta ) - \eta _ { 0 } ( \theta ) ) ) , \quad \theta \in \Theta , \ r \in [ 0 , 1 ] .
$$

Then

$$
\psi ( W ; \theta , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) ) = \partial _ { \theta } Q ( W ; \theta , r ) ,
$$

and so

$$
\begin{array} { r l } { \partial _ { r } \mathrm { E } _ { P } [ \psi ( W ; \theta , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) ) ] = \partial _ { r } \mathrm { E } _ { P } [ \partial _ { \theta } Q ( W ; \theta , r ) ] } & { } \\ { = \partial _ { r } \partial _ { \theta } \mathrm { E } _ { P } [ Q ( W ; \theta , r ) ] = \partial _ { \theta } \partial _ { r } \mathrm { E } _ { P } [ Q ( W ; \theta , r ) ] } & { } \\ { = \partial _ { \theta } \partial _ { r } \mathrm { E } _ { P } [ \ell ( W ; \theta , \eta _ { 0 } ( \theta ) + r ( \eta ( \theta ) - \eta _ { 0 } ( \theta ) ) ) ] . } \end{array}
$$

Hence,

$$
\left. \partial _ { r } \mathrm { E } _ { P } [ \psi ( W ; \theta , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) ) ] \right| _ { r = 0 } = 0
$$

since

$$
\left. \partial _ { r } \mathrm { E } _ { P } [ \ell ( W ; \theta , \eta _ { 0 } ( \theta ) + r ( \eta ( \theta ) - \eta _ { 0 } ( \theta ) ) ) ] \right| _ { r = 0 } = 0 , \quad \mathrm { f o r ~ a l l ~ } \theta \in \Theta ,
$$

as $\eta _ { 0 } ( \theta ) = \beta _ { \theta }$ solves the optimization problem

$$
\operatorname* { m a x } _ { \beta \in \mathcal { B } } \mathrm { E } _ { P } [ \ell ( W ; \theta , \beta ) ] , \quad \mathrm { f o r ~ a l l ~ } \theta \in \Theta .
$$

Here the regularity conditions are needed to make sure that we can interchange $\mathrm { E } _ { P }$ and $\partial _ { \theta }$ and also $\partial _ { \theta }$ and $\partial _ { r }$ in (A.2). This completes the proof of the lemma.

First, we demonstrate that $\mu _ { 0 } \in \mathcal { L } ^ { 1 } ( \mathcal { R } ; \mathbb { R } ^ { d _ { \theta } \times d _ { m } } )$ . Indeed,

$$
\begin{array} { r l } & { \mathrm { E } _ { P } [ \| \mu _ { 0 } ( R ) \| ] \leqslant \mathrm { E } _ { P } \Big [ \| A ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \| \Big ] + \mathrm { E } _ { P } \Big [ \| G ( Z ) \Gamma ( R ) \Omega ( R ) ^ { - 1 } \| \Big ] } \\ & { \qquad \leqslant \mathrm { E } _ { P } \Big [ \| A ( R ) \| \times \| \Omega ( R ) \| ^ { - 1 } \Big ] + \mathrm { E } _ { P } \Big [ \| G ( Z ) \| \times \| \Gamma ( R ) \| \times \| \Omega ( R ) \| ^ { - 1 } \Big ] } \\ & { \qquad \leqslant \Big ( \mathrm { E } _ { P } [ \| A ( R ) \| ^ { 2 } ] \times \mathrm { E } _ { P } [ \| \Omega ( R ) \| ^ { - 2 } ] \Big ) ^ { 1 / 2 } } \\ & { \qquad + \Big ( \mathrm { E } _ { P } \Big [ \| G ( Z ) \| ^ { 2 } \times \| \Gamma ( R ) \| ^ { 2 } \Big ] \times \mathrm { E } _ { P } [ \| \Omega ( R ) \| ^ { - 2 } ] \Big ) ^ { 1 / 2 } , } \end{array}
$$

which is finite by assumptions of the lemma since

$$
\begin{array} { r } { \mathrm { E } _ { P } \Big [ \| G ( Z ) \| ^ { 2 } \times \| \Gamma ( R ) \| ^ { 2 } \Big ] \leqslant \Big ( \mathrm { E } _ { P } [ \| G ( Z ) \| ^ { 4 } ] \times \mathrm { E } _ { P } [ \Gamma ( R ) \| ^ { 4 } ] \Big ) ^ { 1 / 2 } < \infty . } \end{array}
$$

Next, we demonstrate that

$$
\operatorname { E } _ { P } [ \left| | \psi ( W , \theta _ { 0 } , \eta ) | \right| ] < \infty \quad { \mathrm { f o r ~ a l l ~ } } \eta \in T .
$$

Indeed, for all $\eta \in T$ , there exist $\mu \in \mathcal { L } ^ { 1 } ( \mathcal { R } ; \mathbb { R } ^ { d _ { \theta } \times d _ { m } } )$ and $h \in \mathcal { H }$ such that $\boldsymbol { \eta } = \left( \mu , h \right)$ , and so

$$
\begin{array} { r l } & { \mathrm { E } _ { P } [ \| \psi ( W , \theta _ { 0 } , \eta ) \| ] = \mathrm { E } _ { P } [ \| \mu ( X ) m ( W , \theta _ { 0 } , h ( Z ) ) \| ] } \\ & { \qquad \leqslant \mathrm { E } _ { P } \Big [ \| \mu ( R ) \| \times \| m ( W , \theta _ { 0 } , h ( Z ) ) \| \Big ] } \\ & { \qquad = \mathrm { E } _ { P } \Big [ \| \mu ( R ) \| \times \mathrm { E } _ { P } [ \| m ( W , \theta _ { 0 } , h ( Z ) ) \mid R ] \Big ] \leqslant C _ { h } \mathrm { E } [ \| \mu ( R ) \| ] , } \end{array}
$$

which is finite by assumptions of the lemma. Further, (2.1) holds because

$$
\begin{array} { r l } & { \mathrm { E } _ { P } [ \psi ( W , \theta _ { 0 } , \eta _ { 0 } ) ] = \mathrm { E } _ { P } \Big [ \mu _ { 0 } ( R ) m ( W , \theta _ { 0 } , h _ { 0 } ( Z ) ) \Big ] } \\ & { \quad \quad \quad = \mathrm { E } _ { P } \Big [ \mu _ { 0 } ( R ) \mathrm { E } _ { P } [ m ( W , \theta _ { 0 } , h _ { 0 } ( Z ) ) \mid R ] \Big ] = 0 , } \end{array}
$$

where the last equality follows from (2.22).

Finally, we demonstrate that (2.3) holds. To do so, take any $\eta = ( \mu , h ) \in \mathcal { T } _ { N } = T$ . Then

$$
\begin{array} { r l } & { \mathrm { E } _ { P } [ \psi ( W , \theta _ { 0 } , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) ] } \\ & { \qquad = \mathrm { E } _ { P } \Big [ ( \mu _ { 0 } ( R ) + r ( \mu ( R ) - \mu _ { 0 } ( R ) ) ) m ( W , \theta _ { 0 } , h _ { 0 } ( Z ) + r ( h ( Z ) - h _ { 0 } ( Z ) ) ) \Big ] , } \end{array}
$$

and so

$$
\partial _ { \eta } \mathrm { E } _ { P } \psi ( W , \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] = \mathcal { T } _ { 1 } + \mathcal { T } _ { 2 } ,
$$

where

$$
\begin{array} { r l } & { \mathcal { T } _ { 1 } = \mathrm { E } _ { P } \Big [ ( \mu ( R ) - \mu _ { 0 } ( R ) ) m ( W , \theta _ { 0 } , h _ { 0 } ( Z ) \Big ] , } \\ & { \mathcal { T } _ { 2 } = \mathrm { E } _ { P } \Big [ \mu _ { 0 } ( R ) \partial _ { v ^ { \prime } } m ( W , \theta _ { 0 } , v ) | _ { v = h _ { 0 } ( Z ) } ( h ( Z ) - h _ { 0 } ( Z ) ) \Big ] . } \end{array}
$$

Here $\mathcal { I } _ { 1 } = 0$ by the same argument as that in (A.3) and $\mathcal { I } _ { 2 } = 0$ because

$$
\begin{array} { r l } & { \mathcal { Z } _ { 2 } = \mathrm { E } _ { P } \Big [ \mu _ { 0 } ( R ) \mathrm { E } _ { P } [ \partial _ { \nu ^ { \prime } } m ( W , \theta _ { 0 } , v ) | _ { v = h _ { 0 } ( Z ) } \mid X ] ( h ( Z ) - h _ { 0 } ( Z ) ) \Big ] } \\ & { \quad = \mathrm { E } _ { P } \Big [ \mu _ { 0 } ( R ) \Gamma ( X ) ( h ( Z ) - h _ { 0 } ( Z ) ) \Big ] = \mathrm { E } _ { P } \Big [ \mathrm { E } _ { P } [ \mu _ { 0 } ( R ) \Gamma ( R ) \mid Z ] ( h ( Z ) - h _ { 0 } ( Z ) ) \Big ] = 0 } \end{array}
$$

since

$$
\begin{array} { r l } & { \mathrm { E } _ { P } [ \mu _ { 0 } ( R ) \Gamma ( X ) \mid Z ] = \mathrm { E } _ { P } [ A ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] - \mathrm { E } _ { P } [ G ( Z ) \Gamma ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] } \\ & { \phantom { \qquad \quad } = \mathrm { E } _ { P } [ A ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] - G ( Z ) \mathrm { E } _ { P } [ \Gamma ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] } \\ & { \phantom { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } = \mathrm { E } _ { P } [ A ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] - \mathrm { E } _ { P } [ A ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] } \\ & { \phantom { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times } \left( \mathrm { E } _ { P } [ \Gamma ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] \right) ^ { - 1 } \times \mathrm { E } _ { P } [ \Gamma ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] } \\ &  \phantom { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = \mathrm { E } _ { P } [ A ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] - \mathrm { E } _ { P } [ A ( R ) ^ { \prime } \Omega ( R ) ^ { - 1 } \Gamma ( R ) \mid Z ] = 0 . } \end{array}
$$

This completes the proof of the lemma.

# A.12. Proof of Theorem 3.1 (DML2 case)

To start with, note that (3.11) follows immediately from the assumptions. Hence, it suffices to show that (3.10) holds uniformly over $P \in \mathcal P _ { N }$ .

Fix any sequence $\{ P _ { N } \} _ { N \geqslant 1 }$ such that $P _ { N } \in \mathcal { P } _ { N }$ for all $N \geqslant 1$ . Since this sequence is chosen arbitrarily, to show that (3.10) holds uniformly over $P \in \mathcal P _ { N }$ , it suffices to show that

$$
\sqrt { N } \sigma ^ { - 1 } ( \tilde { \theta } _ { 0 } - \theta _ { 0 } ) = \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \bar { \psi } ( W _ { i } ) + O _ { P _ { N } } ( \rho _ { N } ) \sim N ( 0 , \mathrm { I } _ { d } ) .
$$

To do so, we proceed in 5 steps. Step 1 shows the main argument, and Steps 2–5 present auxiliary calculations. In the proof, it will be convenient to denote by $\mathcal { E } _ { N }$ the event that $\widehat { \eta _ { 0 , k } } \in \mathcal { T } _ { N }$ for all $k \in [ K ]$ . Note that by Assumption 3.2 and the union bound, $\mathrm { P } _ { P _ { N } } ( \mathcal { E } _ { N } ) \geqslant 1 - K \Delta _ { n } = 1 - o ( 1 )$ since $\Delta _ { n } = o ( 1 )$ .

Step 1. Denote

$$
\begin{array} { r l } & { \widehat { J } _ { 0 } : = \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } [ \psi ^ { a } ( W ; \widehat { \eta } _ { 0 , k } ) ] , \quad R _ { N , 1 } : = \widehat { J } _ { 0 } - J _ { 0 } , } \\ & { R _ { N , 2 } : = \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } [ \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) ] - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) . } \end{array}
$$

In Steps 2, 3, 4, and 5 below, we will show that

$$
\begin{array} { r l } & { \| R _ { N , 1 } \| = O _ { P _ { N } } ( N ^ { - 1 / 2 } + r _ { N } ) , } \\ & { \| R _ { N , 2 } \| = O _ { P _ { N } } ( N ^ { - 1 / 2 } r _ { N } ^ { \prime } + \lambda _ { N } + \lambda _ { N } ^ { \prime } ) , } \\ & { \| N ^ { - 1 / 2 } \sum _ { i = 1 } ^ { N } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) \| = O _ { P _ { N } } ( 1 ) , } \\ & { \| \sigma ^ { - 1 } \| = O _ { P _ { N } } ( 1 ) , } \end{array}
$$

respectively. Since $N ^ { - 1 / 2 } + r _ { N } \leqslant \rho _ { N } = o ( 1 )$ and all singular values of $J _ { 0 }$ are bounded below from zero by Assumption 3.1, it follows from (A.5) that with PN -probability $1 -$

$o ( 1 )$ , all singular values of $\widehat { J } _ { 0 }$ are bounded below from zero as well. Therefore, with the same $P _ { N }$ -probability,

$$
\tilde { \theta } _ { 0 } = - \widehat { J } _ { 0 } ^ { - 1 } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } [ \psi ^ { b } ( W ; \widehat { \eta } _ { 0 , k } ) ]
$$

and

$$
\begin{array} { l } { \sqrt { N } ( \widetilde { \theta } _ { 0 } - \theta _ { 0 } ) = - \sqrt { N } \widehat { J } _ { 0 } ^ { - 1 } \Big ( \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } \big [ \psi ^ { b } ( W ; \widehat { \eta } _ { 0 , k } ) \big ] + \widehat { J } _ { 0 } \theta _ { 0 } \Big ) } \\ { = - \sqrt { N } \widehat { J } _ { 0 } ^ { - 1 } \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } \big [ \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) \big ] } \\ { = - \Big ( J _ { 0 } + R _ { N , 1 } \Big ) ^ { - 1 } \times \Big ( \displaystyle \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) + \sqrt { N } R _ { N , 2 } \Big ) . } \end{array}
$$

In addition, given that

$$
\begin{array} { r l } & { ( J _ { 0 } + R _ { N , 1 } ) ^ { - 1 } - J _ { 0 } ^ { - 1 } = ( J _ { 0 } + R _ { N , 1 } ) ^ { - 1 } ( J _ { 0 } - ( J _ { 0 } + R _ { N , 1 } ) ) J _ { 0 } ^ { - 1 } } \\ & { \qquad = - ( J _ { 0 } + R _ { N , 1 } ) ^ { - 1 } R _ { N , 1 } J _ { 0 } ^ { - 1 } , } \end{array}
$$

it follows from (A.5) that

$$
\begin{array} { r l } & { \| ( J _ { 0 } + R _ { N , 1 } ) ^ { - 1 } - J _ { 0 } ^ { - 1 } \| \leqslant \| ( J _ { 0 } + R _ { N , 1 } ) ^ { - 1 } \| \times \| R _ { N , 1 } \| \times \| J _ { 0 } ^ { - 1 } \| } \\ & { \qquad = O _ { P _ { N } } ( 1 ) O _ { P _ { N } } ( N ^ { - 1 / 2 } + r _ { N } ) O _ { P _ { N } } ( 1 ) = O _ { P _ { N } } ( N ^ { - 1 / 2 } + r _ { N } ) . } \end{array}
$$

Moreover, since $r _ { N } ^ { \prime } + \sqrt { N } ( \lambda _ { N } + \lambda _ { N } ^ { \prime } ) \leqslant \rho _ { N } = o ( 1 )$ , it follows from (A.6) and (A.7) that

$$
\begin{array} { r l r } {  { \Big \| \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) + \sqrt { N } R _ { N , 2 } \Big \| \leqslant \Big \| \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) \Big \| + \Big \| \sqrt { N } R _ { N , 2 } \Big \| \Big | } } \\ & { } & { \quad = O _ { P _ { N } } ( 1 ) + o _ { P _ { N } } ( 1 ) = O _ { P _ { N } } ( 1 ) . \qquad ( \mathrm { ~ a ~ n ~ d ~ } ~ \mathrm { ~ a ~ n ~ d ~ } ~ ) } \end{array}
$$

Combining (A.10) and (A.11) gives

$$
\begin{array} { r l } & { \left\| \Big ( ( J _ { 0 } + R _ { N , 1 } ) ^ { - 1 } - J _ { 0 } ^ { - 1 } \Big ) \times \Big ( \displaystyle \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) + \sqrt { N } R _ { N , 2 } \Big ) \right\| } \\ & { \leqslant \left\| ( J _ { 0 } + R _ { N , 1 } ) ^ { - 1 } - J _ { 0 } ^ { - 1 } \right\| \times \left\| \displaystyle \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) + \sqrt { N } R _ { N , 2 } \right\| = O _ { P _ { N } } ( N ^ { - 1 / 2 } + r _ { N } ) . } \end{array}
$$

Now, substituting the last bound into (A.9) yields

$$
\begin{array} { l } { { \sqrt { N } ( \displaystyle \tilde { \theta } _ { 0 } - \theta _ { 0 } ) = - J _ { 0 } ^ { - 1 } \times \Big ( \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) + \sqrt { N } R _ { N , 2 } \Big ) + O _ { P _ { N } } ( N ^ { - 1 / 2 } + r _ { N } ) } } \\ { { \displaystyle \qquad = - J _ { 0 } ^ { - 1 } \times \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) + O _ { P _ { N } } ( \rho _ { N } ) , } } \end{array}
$$

where in the second line we used (A.6) and the definition of $\rho _ { N }$ . Combining this with

(A.8) gives

$$
\sqrt { N } \sigma ^ { - 1 } ( \tilde { \theta } _ { 0 } - \theta _ { 0 } ) = \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \bar { \psi } ( W _ { i } ) + O _ { P _ { N } } ( \rho _ { N } )
$$

by the definition of $\psi$ given in the statement of the theorem. In turn, since $\rho _ { N } = o ( 1 )$ , combining (A.12) with the Lindeberg-Feller CLT and the Cramer-Wold device yields (A.4). To complete the proof of the theorem, it remains to establish the bounds (A.5)– (A.8). We do so in four steps below.

Step 2. Here we establish (A.5). Since $K$ is a fixed integer, which is independent of $N$ , it suffices to show that for any $k \in [ K ]$ ,

$$
\begin{array} { r } { \Big \| \mathbb { E } _ { n , k } [ \psi ^ { a } ( W ; \widehat { \eta } _ { 0 , k } ) ] - \mathrm { E } _ { P _ { N } } [ \psi ^ { a } ( W ; \eta _ { 0 } ) ] \Big \| = O _ { P _ { N } } ( N ^ { - 1 / 2 } + r _ { N } ) . } \end{array}
$$

To do so, fix any $k \in [ K ]$ and observe that by the triangle inequality,

$$
\begin{array} { r l } & { \left\| \mathbb { E } _ { n , k } [ \psi ^ { a } ( W ; \widehat { \eta } _ { 0 , k } ) ] - \mathrm { E } _ { P _ { N } } [ \psi ^ { a } ( W ; \eta _ { 0 } ) ] \right\| \leqslant \mathcal { I } _ { 1 , k } + \mathcal { I } _ { 2 , k } , } \end{array}
$$

where

$$
\begin{array} { r l } & { \mathcal { I } _ { 1 , k } : = \Big \| \mathbb { E } _ { n , k } [ \psi ^ { a } ( W ; \widehat { \eta } _ { 0 , k } ) ] - \mathrm { E } _ { P _ { N } } [ \psi ^ { a } ( W ; \widehat { \eta } _ { 0 , k } ) \mid ( W _ { i } ) _ { i \in I _ { k } ^ { c } } ] \Big \| , } \\ & { \mathcal { I } _ { 2 , k } : = \Big \| \mathrm { E } _ { P _ { N } } [ \psi ^ { a } ( W ; \widehat { \eta } _ { 0 , k } ) \mid ( W _ { i } ) _ { i \in I _ { k } ^ { c } } ] - \mathrm { E } _ { P _ { N } } [ \psi ^ { a } ( W ; \eta _ { 0 } ) ] \Big \| . } \end{array}
$$

To bound $\mathcal { I } _ { 2 , k }$ , note that on the event $\mathcal { E } _ { N }$ , which holds with $P _ { N }$ -probability $1 - o ( 1 )$ ,

$$
\mathcal { Z } _ { 2 , k } \leqslant \operatorname* { s u p } _ { \eta \in \mathcal { T } _ { N } } \left\| \mathrm { E } _ { P _ { N } } [ \psi ^ { a } ( W ; \eta ) ] - \mathrm { E } _ { P _ { N } } [ \psi ^ { a } ( W ; \eta _ { 0 } ) ] \right\| = r _ { N } ,
$$

and so $\mathcal { T } _ { 2 , k } = O _ { P _ { N } } ( r _ { N } )$ . To bound I1,k, note that conditional on (Wi)i∈Ikc , the estimator $\widehat { \eta } _ { 0 , k }$ is non-stochastic, and so on the event EN,

$$
\begin{array} { r } { \mathrm { E } _ { P _ { N } } [ \mathcal { T } _ { 1 , k } ^ { 2 } \ | \ ( W _ { i } ) _ { i \in I _ { k } ^ { c } } ] \leqslant n ^ { - 1 } \mathrm { E } _ { P _ { N } } [ \| \psi ^ { a } ( W ; \widehat { \eta } _ { 0 , k } ) \| ^ { 2 } \ | \ ( W _ { i } ) _ { i \in I _ { k } ^ { c } } ] } \\ { \leqslant \underset { \eta \in \mathcal { T } _ { N } } { \operatorname* { s u p } } \ n ^ { - 1 } \mathrm { E } _ { P _ { N } } [ \| \psi ^ { a } ( W ; \eta ) \| ^ { 2 } ] \leqslant c _ { 1 } ^ { 2 } / n , } \end{array}
$$

where the last inequality holds by Assumption 3.2. Hence, $\mathcal { T } _ { 1 , k } = O _ { P _ { N } } ( N ^ { - 1 / 2 } )$ by Lemma 6.1 in the Appendix. Combining the bounds $\mathcal { T } _ { 1 , k } = O _ { P _ { N } } ( N ^ { - 1 / 2 } )$ and $\mathcal { T } _ { 2 , k } = O _ { P _ { N } } ( r _ { N } )$ with (A.14) gives (A.13).

Step 3. Here we establish (A.6). This is the step where we invoke the Neyman orthogonality (or near-orthogonality) condition. Again, since $K$ is a fixed integer, which is independent of $N$ , it suffices to show that for any $k \in [ K ]$ ,

$$
\mathbb { E } _ { n , k } [ \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) ] - \frac { 1 } { n } \sum _ { i \in I _ { k } } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) = O _ { P _ { N } } ( N ^ { - 1 / 2 } r _ { N } ^ { \prime } + \lambda _ { N } + \lambda _ { N } ^ { \prime } ) .
$$

To do so, fix any $k \in [ K ]$ and introduce the following additional empirical process notation:

$$
\mathbb { G } _ { n , k } [ \phi ( W ) ] = \frac { 1 } { \sqrt { n } } \sum _ { i \in I _ { k } } \Big ( \phi ( W _ { i } ) - \int \phi ( w ) d P _ { N } \Big ) ,
$$

where $\phi$ is any $P _ { N }$ -integrable function on $\mathcal { W }$ . Then observe that by the triangle inequality,

$$
\left. \left. \mathbb { E } _ { n , k } [ \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) ] - \frac { 1 } { n } \sum _ { i \in I _ { k } } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) \right. \right. \leqslant \frac { \mathcal { Z } _ { 3 , k } + \mathcal { Z } _ { 4 , k } } { \sqrt { n } } ,
$$

where

$$
\begin{array} { r l } & { \mathcal { T } _ { 3 , k } : = \Big \| \mathbb { G } _ { n , k } [ \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) ] - \mathbb { G } _ { n , k } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] \Big \| , } \\ & { \mathcal { T } _ { 4 , k } : = \sqrt { n } \Big \| \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) \mid ( W _ { i } ) _ { i \in I _ { k } ^ { c } } ] - \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] \Big \| . } \end{array}
$$

To bound I3,k, note that, as above, conditional on $( W _ { i } ) _ { i \in I _ { k } ^ { c } }$ , the estimator $\widehat { \eta } _ { 0 , k }$ is nonstochastic, and so on the event $\mathcal { E } _ { N }$ ,

$$
\begin{array} { r l } { \operatorname { E } _ { P _ { N } } [ \mathcal { T } _ { 3 , k } ^ { 2 } \mid ( W _ { i } ) _ { i \in I _ { k } ^ { c } } ] = \operatorname { E } _ { P _ { N } } \Big [ | | \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) | | ^ { 2 } \mid ( W _ { i } ) _ { i \in I _ { k } ^ { c } } \Big ] } & { } \\ { \leqslant \underset { \eta \in \mathcal { T } _ { N } } { \operatorname* { s u p } } \operatorname { E } _ { P _ { N } } \Big [ | | \psi ( W ; \theta _ { 0 } , \eta ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) | | ^ { 2 } \mid ( W _ { i } ) _ { i \in I _ { k } ^ { c } } \Big ] } & { } \\ { \leqslant \underset { \eta \in \mathcal { T } _ { N } } { \operatorname* { s u p } } \operatorname { E } _ { P _ { N } } \Big [ | | \psi ( W ; \theta _ { 0 } , \eta ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) | | ^ { 2 } \Big ] = ( r _ { N } ^ { \prime } ) ^ { 2 } } & { } \end{array}
$$

by the definition of $r _ { N } ^ { \prime }$ in Assumption 3.2. Hence, $\mathcal { T } _ { 3 , k } = O _ { P _ { N } } ( r _ { N } ^ { \prime } )$ by Lemma 6.1 in the Appendix. To bound I4,k, introduce the function

$$
f _ { k } ( r ) : = \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } + r ( \widehat { \eta } _ { 0 , k } - \eta _ { 0 } ) ) \mid ( W _ { i } ) _ { i \in I _ { k } ^ { c } } ] - \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] , \quad r \in [ 0 , 1 ] .
$$

Then, by Taylor’s expansion,

$$
f _ { k } ( 1 ) = f _ { k } ( 0 ) + f _ { k } ^ { \prime } ( 0 ) + f _ { k } ^ { \prime \prime } ( \tilde { r } ) / 2 , \quad \mathrm { f o r ~ s o m e ~ } \tilde { r } \in ( 0 , 1 ) .
$$

But $\| f _ { k } ( 0 ) \| = 0$ since

$$
\operatorname { E } _ { P _ { N } } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \mid ( W _ { i } ) _ { i \in I _ { k } ^ { c } } ] = \operatorname { E } _ { P _ { N } } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] .
$$

In addition, on the event $\mathcal { E } _ { N }$ , by the Neyman $\lambda _ { N }$ near-orthogonality condition imposed in Assumption 3.1,

$$
\begin{array} { r } { \| f _ { k } ^ { \prime } ( 0 ) \| = \left\| \partial _ { \eta } \mathrm { E } _ { P _ { N } } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \widehat { \eta } _ { 0 , k } - \eta _ { 0 } ] \right\| \leqslant \lambda _ { N } . } \end{array}
$$

Moreover, on the event $\mathcal { E } _ { N }$ ,

$$
\| f _ { k } ^ { \prime \prime } ( \tilde { r } ) \| \leqslant \operatorname* { s u p } _ { r \in ( 0 , 1 ) } \| f _ { k } ^ { \prime \prime } ( r ) \| \leqslant \lambda _ { N } ^ { \prime }
$$

by the definition $\lambda _ { N } ^ { \prime }$ in Assumption 3.2. Hence,

$$
\mathcal { T } _ { 4 , k } = \sqrt { n } \| f _ { k } ( 1 ) \| = O _ { P _ { N } } \big ( \sqrt { n } ( \lambda _ { N } + \lambda _ { N } ^ { \prime } ) \big ) .
$$

Combining the bounds on I3,k and I4,k with (A.16) and using the fact that $n ^ { - 1 } =$ $O ( N ^ { - 1 } )$ gives (A.15).

Step 4. To establish (A.7), note that

$$
\mathrm { E } _ { P _ { N } } \Big [ \Big \lVert \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) \Big \rVert ^ { 2 } \Big ] = \mathrm { E } _ { P _ { N } } \Big [ \lVert \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \rVert ^ { 2 } \Big ] \leqslant c _ { 1 } ^ { 2 }
$$

by Assumption 3.2. Combining this with Markov’s inequality gives (A.7).

Step 5. Here we establish (A.8). Note that all eigenvalues of the matrix

$$
\sigma ^ { 2 } = J _ { 0 } ^ { - 1 } \mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ^ { \prime } ] ( J _ { 0 } ^ { - 1 } ) ^ { \prime }
$$

are bounded from below by $c _ { 0 } / c _ { 1 } ^ { 2 }$ since all singular values of $J _ { 0 }$ are bounded from above by $c _ { 1 }$ by Assumption 3.1 and all eigenvalues of $\operatorname { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ^ { \prime } ]$ are bounded from below by $c _ { 0 }$ by Assumption 3.2. Hence, given that $\| \sigma ^ { - 1 } \|$ is the largest eigenvalue of $\sigma ^ { - 1 }$ , it follows that $\| \sigma ^ { - 1 } \| = c _ { 1 } / \sqrt { c _ { 0 } }$ . This gives (A.8) and completes the proof of the theorem.

# A.13. Proof of Theorem 3.1 (DML1 case)

As in the case of the DML2 version, note that (3.11) follows immediately from the assumptions, and so it suffices to show that (3.10) holds uniformly over $P \in \mathcal P _ { N }$ .

Fix any sequence $\{ P _ { N } \} _ { N \geqslant 1 }$ such that $P _ { N } \in \mathcal { P } _ { N }$ for all $N \geqslant 1$ . Since this sequence is chosen arbitrarily, to show that (3.10) holds uniformly over $P \in \mathcal P _ { N }$ P , it suffices to show that

$$
\sqrt { N } \sigma ^ { - 1 } ( \tilde { \theta } _ { 0 } - \theta _ { 0 } ) = \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \bar { \psi } ( W _ { i } ) + O _ { P _ { N } } ( \rho _ { N } ) \sim N ( 0 , \mathrm { I } _ { d } ) .
$$

To do so, for all $k \in [ K ]$ , denote

$$
\begin{array} { r l } & { \widehat { J } _ { 0 , k } : = \mathbb { E } _ { n , k } [ \psi ^ { a } ( W ; \widehat { \eta } _ { 0 , k } ) ] , \quad R _ { N , 1 , k } : = \widehat { J } _ { 0 , k } - J _ { 0 } , } \\ & { R _ { N , 2 , k } : = \mathbb { E } _ { n , k } [ \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) ] - \displaystyle \frac { 1 } { n } \sum _ { i \in I _ { k } } \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) . } \end{array}
$$

Since $K$ is a fixed integer, which is independent of $n$ , it follows by the same arguments as those in Steps 2–5 in Section A.12 that

$$
\begin{array} { l } { \displaystyle \operatorname* { m a x } _ { k \in [ K ] } \| R _ { N , 1 , k } \| = O _ { P _ { N } } \big ( N ^ { - 1 / 2 } + r _ { N } \big ) , } \\ { \displaystyle \operatorname* { m a x } _ { k \in [ K ] } \| R _ { N , 2 , k } \| = O _ { P _ { N } } \big ( N ^ { - 1 / 2 } r _ { N } ^ { \prime } + \lambda _ { N } + \lambda _ { N } ^ { \prime } \big ) , } \\ { \displaystyle \operatorname* { m a x } _ { k \in [ K ] } \| n ^ { - 1 / 2 } \sum _ { i \in I _ { k } } \psi \big ( { \cal W } _ { i } ; \theta _ { 0 } , \eta _ { 0 } \big ) \| = O _ { P _ { N } } \big ( 1 \big ) , } \\ { \| \sigma ^ { - 1 } \| = O _ { P _ { N } } \big ( 1 \big ) . } \end{array}
$$

Since $N ^ { - 1 / 2 } + r _ { N } \leqslant \rho _ { N } = o ( 1 )$ and all singular values of $J _ { 0 }$ are bounded below from zero by Assumption 3.1, it follows from (A.18) that for all $k \in [ K ]$ , with $P _ { N }$ -probability $1 - o ( 1 )$ , all singular values of J0,k are bounded below from zero, and so with the same $P _ { N }$ -probability,

$$
\begin{array} { r } { \check { \theta } _ { 0 , k } = - \widehat { J } _ { 0 , k } ^ { - 1 } \mathbb { E } _ { n , k } [ \psi ^ { b } ( W ; \widehat { \eta } _ { 0 , k } ) ] . } \end{array}
$$

Hence, by the same arguments as those in Step 1 inb Section A.12, it follows from the bounds (A.18)–(A.21) that for all $k \in [ K ]$ ,

$$
\sqrt { n } \sigma ^ { - 1 } ( \check { \theta } _ { 0 , k } - \theta _ { 0 } ) = \frac { 1 } { \sqrt { n } } \sum _ { i \in I _ { k } } \bar { \psi } ( W _ { i } ) + O _ { P _ { N } } ( \rho _ { N } ) .
$$

Therefore,

$$
\sqrt { N } \sigma ^ { - 1 } ( \tilde { \theta } _ { 0 } - \theta _ { 0 } ) = \sqrt { N } \sigma ^ { - 1 } \Bigl ( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \tilde { \theta } _ { 0 , k } - \theta _ { 0 } \Bigr ) = \frac { 1 } { \sqrt { N } } \sum _ { i = 1 } ^ { N } \bar { \psi } ( W _ { i } ) + O _ { P _ { N } } ( \rho _ { N } ) .
$$

In turn, since $\rho _ { N } ~ = ~ o ( 1 )$ , combining (A.22) with the Lindeberg-Feller CLT and the Cramer-Wold device yields (A.17) and completes the proof of the theorem.

# A.14. Proof of Theorem 3.2.

In this proof, all bounds hold uniformly in $P \in \mathcal P _ { N }$ for $N \geqslant 3$ , and we do not repeat this qua lification throughout. Also, the second asserted claim follows immediately from the first one and Theorem 3.1. Hence, it suffices to prove the first asserted claim.

In the proof of Theorem 3.1 in Section A.12, we established that $\| \widehat { J } _ { 0 } - J _ { 0 } \| = O _ { P } ( r _ { N } +$ $N ^ { - 1 / 2 }$ ). Hence, since $\| J _ { 0 } ^ { - 1 } \| \leqslant c _ { 0 } ^ { - 1 }$ by Assumption 3.1 and

$$
\Bigl \| \mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ^ { \prime } ] \Bigr \| \leqslant \mathrm { E } _ { P } [ \| \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } ] \leqslant c _ { 1 } ^ { 2 }
$$

by Assumption 3.2, it suffices to show that

$$
\Big \lVert \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { n , k } [ \psi ( W ; \tilde { \theta } _ { 0 } , \widehat { \eta } _ { 0 , k } ) \psi ( W ; \tilde { \theta } _ { 0 } , \widehat { \eta } _ { 0 , k } ) ^ { \prime } ] - \mathbb { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ^ { \prime } ] \Big \rVert = O _ { P } ( \varrho _ { N } ) .
$$

Moreover, since both $K$ and $d _ { \theta }$ , the dimension of $\psi$ , are fixed integers, which are independent of $N$ , the last bound will follow if we show that for all $k \in [ K ]$ and all $j , k \in [ d _ { \theta } ]$ ,

$$
\mathcal { T } _ { k j l } : = \left| \mathbb { E } _ { n , k } [ \psi _ { j } ( W ; \tilde { \theta } _ { 0 } , \widehat { \eta } _ { 0 , k } ) \psi _ { l } ( W ; \tilde { \theta } _ { 0 } , \widehat { \eta } _ { 0 , k } ) ] - \mathrm { E } _ { P } [ \psi _ { j } ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi _ { l } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] \right|
$$

satisfies

$$
\mathcal { T } _ { k j l } = O _ { P } ( \varrho _ { N } ) .
$$

To do so, observe that by the triangle inequality,

$$
\mathcal { T } _ { k j l } \leqslant \mathcal { T } _ { k j l , 1 } + \mathcal { T } _ { k j l , 2 } ,
$$

where

$$
\begin{array} { r l } & { \mathcal { T } _ { k j l , 1 } : = \Big | \mathbb { E } _ { n , k } [ \psi _ { j } ( W ; \tilde { \theta } _ { 0 } , \widehat { \eta } _ { 0 , k } ) \psi _ { l } ( W ; \tilde { \theta } _ { 0 } , \widehat { \eta } _ { 0 , k } ) ] - \mathbb { E } _ { n , k } [ \psi _ { j } ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi _ { l } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] \Big | , } \\ & { \mathcal { T } _ { k j l , 2 } : = \Big | \mathbb { E } _ { n , k } [ \psi _ { j } ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi _ { l } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] - \mathbb { E } _ { P } [ \psi _ { j } ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi _ { l } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] \Big | . } \end{array}
$$

We bound $\mathcal { I } _ { k j l , 2 }$ first. If $q \geqslant 4$ , then

$$
\begin{array} { r l } & { \mathrm { E } _ { P } [ \mathcal { Z } _ { k j l , 2 } ^ { 2 } ] \leqslant n ^ { - 1 } \mathrm { E } _ { P } \Big [ ( \psi _ { j } ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi _ { l } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ) ^ { 2 } \Big ] } \\ & { \qquad \leqslant n ^ { - 1 } \Big ( \mathrm { E } _ { P } [ \psi _ { j } ^ { 4 } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] \times \mathrm { E } _ { P } [ \psi _ { l } ^ { 4 } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] \Big ) ^ { 1 / 2 } } \\ & { \qquad \leqslant n ^ { - 1 } \mathrm { E } _ { P } [ \| \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 4 } ] \leqslant c _ { 1 } ^ { 4 } , } \end{array}
$$

where the second line holds by H¨older’s inequality, and the third one by Assumption 3.2. Hence, $\mathcal { T } _ { k j l , 2 } = O _ { P } ( N ^ { - 1 / 2 } )$ . If $q \in ( 2 , 4 )$ , we apply the following von Bahr- Esseen inequality with $p = q / 2$ : if $X _ { 1 } , \ldots , X _ { n }$ are independent random variables with mean zero,

then for any $p \in [ 1 , 2 ]$ ,

$$
\operatorname { E } \left[ \left| \sum _ { i = 1 } ^ { n } X _ { i } \right| ^ { p } \right] \leqslant { \Big ( } 2 - { \frac { 1 } { n } } { \Big ) } \sum _ { i = 1 } ^ { n } \operatorname { E } [ | X _ { i } | ^ { p } ] ;
$$

see DasGupta (2008), p. 650. This gives

$$
\begin{array} { r l } & { { \mathrm { E } } _ { P } [ \mathcal { Z } _ { k j l , 2 } ^ { q / 2 } ] \lesssim n ^ { - q / 2 + 1 } { \mathrm { E } } _ { P } \Big [ ( \psi _ { j } ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi _ { l } ( W ; \theta _ { 0 } , \eta _ { 0 } ) ) ^ { q / 2 } \Big ] } \\ & { \qquad \leqslant n ^ { - q / 2 + 1 } { \mathrm { E } } _ { P } [ \| \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { q } ] \lesssim n ^ { - q / 2 + 1 } } \end{array}
$$

by Assumption 3.2. Hence, $\mathcal { T } _ { k j l , 2 } = O _ { P } \big ( N ^ { 2 / q - 1 } \big )$ . Conclude that

$$
\begin{array} { r } { \mathcal { T } _ { k j l , 2 } = O _ { P } \Big ( N ^ { - [ ( 1 - 2 / q ) \wedge ( 1 / 2 ) ] } \Big ) . } \end{array}
$$

Next, we bound $\mathcal { T } _ { k j l , 1 }$ . To do so, observe that for any numbers $a$ , $b$ , $\delta \boldsymbol { a }$ , and $\delta b$ such that $| a | \vee | b | \leqslant c$ and $| \delta \boldsymbol { a } | \vee | \delta \boldsymbol { b } | \leqslant r$ , we have

$$
\left| ( a + \delta a ) ( b + \delta b ) - a b \right| \leqslant 2 r ( c + r ) .
$$

Denoting

$\psi _ { h i } : = \psi _ { h } ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) ~ \mathrm { a n d } ~ \widehat { \psi } _ { h i } : = \psi _ { h } ( W _ { i } ; \widetilde { \theta } _ { 0 } , \widehat { \eta } _ { 0 , k } ) , ~ \mathrm { f o r } ~ ( h , i )$ $( h , i ) \in \{ j , l \} \times I _ { k }$ , and applying the inequality above with $a : = \psi _ { j i }$ , $b : = \psi _ { l i }$ , $a + \delta a : = \widehat { \psi } _ { j i }$ , $b + \delta b : = \widehat { \psi } _ { l i }$ , $r : = | \widehat { \psi } _ { j i } - \psi _ { j i } | \vee | \widehat { \psi } _ { l i } - \psi _ { l i } |$ , and $c : = | \psi _ { j i } | \vee | \psi _ { l i } |$ gives

$$
\begin{array} { r l } {  { \mathcal { Z } _ { k j l , 1 } = \Big | \frac { 1 } { n } \sum _ { i \in I _ { k } } \widehat { \psi } _ { j i } \widehat { \psi } _ { l i } - \psi _ { j i } \psi _ { k i } \Big | \leqslant \frac { 1 } { n } \sum _ { i \in I _ { k } } | \widehat { \psi } _ { j i } \widehat { \psi } _ { l i } - \psi _ { j i } \psi _ { l i } | } } \\ & { \leqslant \frac { 2 } { n } \sum _ { i \in I _ { k } } \Big ( | \widehat { \psi } _ { j i } - \psi _ { j i } | \vee | \widehat { \psi } _ { l i } - \psi _ { l i } | \Big ) \times \Big ( | \psi _ { j i } | \vee | \psi _ { l i } | + | \widehat { \psi } _ { j i } - \psi _ { j i } | \vee | \widehat { \psi } _ { l i } - \psi _ { l i } | \Big ) } \\ & { \leqslant \Big ( \frac { 2 } { n } \sum _ { i \in I _ { k } } \Big ( | \widehat { \psi } _ { j i } - \psi _ { j i } | ^ { 2 } \vee | \widehat { \psi } _ { l i } - \psi _ { l i } | ^ { 2 } \Big ) \Big ) ^ { 1 / 2 } } \\ & { \quad \times \Big ( \frac { 2 } { n } \sum _ { i \in I _ { k } } \Big ( | \psi _ { j i } | \vee | \psi _ { l i } | + | \widehat { \psi } _ { j i } - \psi _ { j i } | \vee | \widehat { \psi } _ { l i } - \psi _ { l i } | \Big ) ^ { 2 } \Big ) ^ { 1 / 2 } . } \end{array}
$$

In addition, the expression in the last line above is bounded by

$$
\biggl ( \frac { 2 } { n } \sum _ { i \in I _ { k } } | \psi _ { j i } | ^ { 2 } \vee | \psi _ { l i } | ^ { 2 } \biggr ) ^ { 1 / 2 } + \biggl ( \frac { 2 } { n } \sum _ { i \in I _ { k } } | \widehat { \psi } _ { j i } - \psi _ { j i } | ^ { 2 } \vee | \widehat { \psi } _ { l i } - \psi _ { l i } | ^ { 2 } \biggr ) ^ { 1 / 2 } ,
$$

and so

$$
\mathcal { T } _ { k j l , 1 } ^ { 2 } \lesssim R _ { N } \times \Big ( \frac { 1 } { n } \sum _ { i \in I _ { k } } \| \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } + R _ { N } \Big ) ,
$$

where

$$
R _ { N } : = \frac { 1 } { n } \sum _ { i \in I _ { k } } \| \psi ( W _ { i } ; \widetilde { \theta } _ { 0 } , \widehat { \eta } _ { 0 , k } ) - \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } .
$$

Moreover,

$$
\frac { 1 } { n } \sum _ { i \in I _ { k } } \| \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } = O _ { P } ( 1 ) ,
$$

by Markov’s inequality since

$$
\mathrm { E } _ { P } \big [ \frac { 1 } { n } \sum _ { i \in I _ { k } } \| \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } \big ] = \mathrm { E } _ { P } [ \| \psi ( W ; \theta _ { 0 } , \eta _ { 0 } \| ^ { 2 } ] \leqslant c _ { 1 } ^ { 2 }
$$

by Assumption 3.2. It remains to bound $R _ { N }$ . We have

$$
R _ { N } \lesssim \frac { 1 } { n } \sum _ { i \in I _ { k } } \Big \Vert \psi ^ { a } ( W _ { i } ; \widehat { \eta } _ { 0 , k } ) ( \widetilde { \theta } _ { 0 } - \theta _ { 0 } ) \Big \Vert ^ { 2 } + \frac { 1 } { n } \sum _ { i \in I _ { k } } \Big \Vert \psi ( W _ { i } ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) - \psi ( W _ { i } ; \theta _ { 0 } , \eta _ { 0 } ) \Big \Vert ^ { 2 } .
$$

The first term on the right-hand side of (A.26) is bounded from above by

$$
\Big ( \frac { 1 } { n } \sum _ { i \in I _ { k } } \| \psi ^ { a } ( W _ { i } ; \widehat { \eta } _ { 0 , k } ) \| ^ { 2 } \Big ) \times \| \tilde { \theta } _ { 0 } - \theta _ { 0 } \| ^ { 2 } = O _ { P } ( 1 ) \times O _ { P } ( N ^ { - 1 } ) = O _ { P } ( N ^ { - 1 } ) ,
$$

and the conditional expectation of the second term given $( W _ { i } ) _ { i \in I _ { k } ^ { c } }$ on the event that $\widehat { \eta _ { 0 , k } } \in \mathcal { T } _ { N }$ is equal to

$$
\begin{array} { r l } & { \mathrm { E } _ { P } \Big [ \| \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 , k } ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } \mid ( W _ { i } ) _ { i \in I _ { k } ^ { c } } \Big ] } \\ & { \qquad \leqslant \underset { \eta \in { \mathcal T } _ { N } } { \operatorname* { s u p } } \mathrm { E } _ { P } \Big [ \| \psi ( W ; \theta _ { 0 } , \eta ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } \mid ( W _ { i } ) _ { i \in I _ { k } ^ { c } } \Big ] = ( r _ { N } ^ { \prime } ) ^ { 2 } . } \end{array}
$$

Since the event that $\widehat { \eta } _ { 0 , k } \in \mathcal { T } _ { N }$ holds with probability $1 - \Delta _ { N } = 1 - o ( 1 )$ , it follows that $R _ { N } = O _ { P } ( N ^ { - 1 } + ( r _ { N } ^ { \prime } ) ^ { 2 } )$ , and so

$$
\begin{array} { r } { \mathcal { T } _ { k j l , 1 } = O _ { P } \Big ( N ^ { - 1 / 2 } + r _ { N } ^ { \prime } \Big ) . } \end{array}
$$

Combining the bounds (A.25) and (A.27) with (A.24) gives (A.23) and completes the proof of the theorem.

# Proof of Theorem 3.3.

We only consider the case of the DML1 estimator and note that the DML2 estimator can be treated similarly.

The main part of the proof is the same as that in the linear case (Theorem 3.1, DML1 case, presented in Section A.13), once we have the following lemma that establishes approximate linearity of the subsample DML estimators θˇ0,k.

Lemma 6.3. (Linearization for Subsample DML in Nonlinear Problems) Under the conditions of Theorem 3.3, for any $k = 1 , \ldots , K$ , the estimator $\check { \theta } _ { 0 } = \check { \theta } _ { 0 , k }$ defined by equation (3.2) obeys

$$
\sqrt { n } \sigma _ { 0 } ^ { - 1 } ( \check { \theta } _ { 0 } - \theta _ { 0 } ) = \frac { 1 } { \sqrt { n } } \sum _ { i \in I } \bar { \psi } ( W _ { i } ) + O _ { P } ( \rho _ { n } ^ { \prime } )
$$

uniformly over $P \in \mathcal { P } _ { N }$ , where $\rho _ { n } ^ { \prime } = n ^ { - 1 / 2 } + r _ { N } + r _ { N } ^ { \prime } + n ^ { 1 / 2 } \lambda _ { N } + n ^ { 1 / 2 } \lambda _ { N } ^ { \prime } \lesssim \delta _ { N }$ and where $\bar { \psi } ( \cdot ) : = - \sigma _ { 0 } ^ { - 1 } J _ { 0 } ^ { - 1 } \psi ( \cdot , \theta _ { 0 } , \eta _ { 0 } )$ .

Proof of Lemma 6.3. Fix any $k = 1 , \ldots , K$ and any sequence $\{ P _ { N } \} _ { N \geqslant 1 }$ such that $P _ { N } ~ \in ~ \mathcal { P } _ { N }$ for all $\textit { N } \geqslant \ 1$ . To prove the asserted claim, it suffices to show that the estimator $\check { \theta } _ { 0 } = \check { \theta } _ { 0 , k }$ satisfies (A.28) with $P$ replaced by $P _ { N }$ . To do so, we split the proof

into four steps. In the proof, we will use En, Gn, $I$ , and $\widehat { \eta } _ { 0 }$ instead of En,k, Gn,k, $\boldsymbol { I } _ { k }$ , and $\widehat { \eta } _ { 0 , k }$ , respectively.

Step 1. (Preliminary Rate Result). We claim that with $P _ { N }$ -probability $1 - o ( 1 )$ ,

$$
\lVert \check { \theta } _ { 0 } - \theta _ { 0 } \rVert \leqslant \tau _ { N } .
$$

To show this claim, note that the definition of $\dot { \theta } _ { 0 }$ implies that

$$
\begin{array} { r } { \left\| \mathbb { E } _ { n } [ \psi ( W ; \check { \theta } _ { 0 } , \widehat { \eta } _ { 0 } ) ] \right\| \leqslant \left\| \mathbb { E } _ { n } [ \psi ( W ; \theta _ { 0 } , \widehat { \eta } _ { 0 } ) ] \right\| + \epsilon _ { N } , } \end{array}
$$

which in turn implies via the triangle inequality that, with $P _ { N }$ -probability $1 - o ( 1 )$ ,

$$
\Bigl \| \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta , \eta _ { 0 } ) ] \vert _ { \theta = \check { \theta } _ { 0 } } \Bigr \| \leqslant \epsilon _ { N } + 2  { \mathbb { T } } _ { 1 } + 2  { \mathbb { T } } _ { 2 } ,
$$

where

$$
\begin{array} { r l } & { \mathcal { T } _ { 1 } : = \underset { \theta \in \Theta , \eta \in \mathcal { T } _ { N } } { \operatorname* { s u p } } \left. \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta , \eta ) ] - \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta , \eta _ { 0 } ) ] \right. , } \\ & { \mathcal { T } _ { 2 } : = \underset { \eta \in \{ \eta _ { 0 } , \hat { \eta } _ { 0 } \} } { \operatorname* { m a x } } \underset { \theta \in \Theta } { \operatorname* { s u p } } \left. \mathbb { E } _ { n } [ \psi ( W ; \theta , \eta ) ] - \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta , \eta ) ] \right. . } \end{array}
$$

Here $\epsilon _ { N } = o ( \tau _ { N } )$ because $\epsilon _ { N } = o ( \delta _ { N } N ^ { - 1 / 2 } )$ , $\delta _ { N } = o ( 1 )$ , and $\tau _ { N } \geqslant c _ { 0 } N ^ { - 1 / 2 } \log n$ . Also, $\mathcal { T } _ { 1 } = r _ { N } \leqslant \delta _ { N } \tau _ { N } = o ( \tau _ { N } )$ by Assumption 3.4(c). Moreover, applying Lemma 6.2 to the function class F1,η for $\eta = \eta _ { 0 }$ and $\eta = \hat { \eta } _ { 0 }$ defined in Assumption 3.4, conditional on $( W _ { i } ) _ { i \in I ^ { c } }$ and $I ^ { c }$ , so that $\widehat { \eta } _ { 0 }$ is fixed afte bconditioning, shows that with $P _ { N }$ -probability $1 - o ( 1 )$ ,

$$
I _ { 2 } \lesssim N ^ { - 1 / 2 } ( 1 + N ^ { - 1 / 2 + 1 / q } \log n ) \lesssim N ^ { - 1 / 2 } = o ( \tau _ { N } ) .
$$

Hence, it follows from (A.30) and Assumption 3.3 that with $P _ { N }$ -probability $1 - o ( 1 )$

$$
\begin{array} { r } { \| J _ { 0 } ( \check { \theta } _ { 0 } - \theta _ { 0 } ) \| \wedge c _ { 0 } \leqslant \Big \| \operatorname { E } _ { P _ { N } } [ \psi ( W ; \theta , \eta _ { 0 } ) ] | _ { \theta = \check { \theta } _ { 0 } } \Big \| = o ( \tau _ { N } ) . } \end{array}
$$

Combining this bound with the fact that the singular values of $J _ { 0 }$ are bounded away from zero, which holds by Assumption 3.3, gives the claim of this step.

Step 2. (Linearization) Here we prove the claim of the lemma. First, by definition of $\dot { \theta } _ { 0 }$ we have

$$
\sqrt { n } \Big \| \mathbb { E } _ { n } [ \psi ( W ; \check { \theta } _ { 0 } , \widehat { \eta } _ { 0 } ) ] \Big \| \leqslant \operatorname* { i n f } _ { \theta \in \Theta } \sqrt { n } \Big \| \mathbb { E } _ { n } [ \psi ( W ; \theta , \widehat { \eta } _ { 0 } ) ] \Big \| + \epsilon _ { N } \sqrt { n } .
$$

Also, it will be shown in Step 4 that

$$
\begin{array} { r l } {  { \mathcal { Z } _ { 3 } : = \operatorname* { i n f } _ { \theta \in \Theta } \sqrt { n } \| \mathbb { E } _ { n } [ \psi ( W ; \theta , \widehat { \eta } _ { 0 } ) ] \| } } \\ & { \quad = O _ { P _ { N } } ( n ^ { - 1 / 2 + 1 / q } \log n + r _ { N } ^ { \prime } \log ^ { 1 / 2 } ( 1 / r _ { N } ^ { \prime } ) + \lambda _ { N } \sqrt { n } + \lambda _ { N } ^ { \prime } \sqrt { n } ) . } \end{array}
$$

Moreover, for any $\theta \in \Theta$ and $\eta \in \mathcal { T } _ { N }$ , we have

$$
\begin{array} { r l } & { \sqrt { n } \mathbb { E } _ { n } [ \psi ( W ; \theta , \eta ) ] = \sqrt { n } \mathbb { E } _ { n } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] + \mathbb { G } _ { n } [ \psi ( W ; \theta , \eta ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] } \\ & { \phantom { \sqrt { n } \mathbb { E } _ { n } } + \sqrt { n } \Big ( \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta , \eta ) \Big ) , } \end{array}
$$

where we are using that $\mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] = 0$ . Finally, by Taylor’s expansion of the

function $r \mapsto \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta _ { 0 } + r ( \theta - \theta _ { 0 } ) , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) ) ]$ , which vanishes at $r = 0$ ,

$$
\begin{array} { l } { { \displaystyle { \bf E } _ { P _ { N } } [ \psi ( W ; \theta , \eta ) ] = J _ { 0 } ( \theta - \theta _ { 0 } ) + \partial _ { \eta } { \bf E } _ { P _ { N } } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] } } \\ { ~ + \displaystyle \int _ { 0 } ^ { 1 } 2 ^ { - 1 } \partial _ { r } ^ { 2 } { \bf E } _ { P _ { N } } [ W ; \theta _ { 0 } + r ( \theta - \theta _ { 0 } ) , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) ] d r . } \end{array}
$$

Therefore, since $\lVert \check { \theta } _ { 0 } - \theta _ { 0 } \rVert \leqslant \tau _ { N }$ and $\eta \in \mathcal { T } _ { N }$ with PN -probability $1 - o ( 1 )$ , and since by Neyman $\lambda _ { N }$ -near orthogonality,

$$
\lVert \partial _ { \eta } \mathrm { E } _ { P _ { N } } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] [ \widehat { \eta } _ { 0 } - \eta _ { 0 } ] \rVert \leqslant \lambda _ { N } \mathrm { , }
$$

applying (A.34) with $\theta = \dot { \theta } _ { 0 }$ and $\eta = \hat { \eta } _ { 0 }$ , we have with $P _ { N }$ -probability $1 - o ( 1 )$ ,

$$
\begin{array} { r } { \sqrt { n } \Big \lvert | \mathbb { E } _ { n } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] + J _ { 0 } ( \check { \theta } _ { 0 } - \theta _ { 0 } ) \Big \rvert \Big \rvert \leqslant \lambda _ { N } \sqrt { n } + \epsilon _ { N } \sqrt { n } + \mathcal { T } _ { 3 } + \mathcal { T } _ { 4 } + \mathcal { T } _ { 5 } , } \end{array}
$$

where by Assumption 3.4,

$$
\mathcal { Z } _ { 4 } : = \sqrt { n } \operatorname* { s u p } _ { \| \theta - \theta _ { 0 } \| \leqslant \tau _ { N } , \eta \in \mathcal { T } _ { N } } \Big \| \int _ { 0 } ^ { 1 } 2 ^ { - 1 } \partial _ { r } ^ { 2 } \mathsf { E } _ { P _ { N } } \big [ W ; \theta _ { 0 } + r ( \theta - \theta _ { 0 } ) , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) \big ] d r \Big \| \leqslant \lambda _ { N } ^ { \prime } \sqrt { n } ,
$$

and by Step 3 below, with $P _ { N }$ -probability $1 - o ( 1 )$ ,

$$
\begin{array} { r l } & { \mathcal { T } _ { 5 } : = \underset { \| \theta - \theta _ { 0 } \| \leqslant \tau _ { N } } { \operatorname* { s u p } } \left\| \mathbb { G } _ { n } \Big ( \psi ( W ; \theta , \widehat { \eta } _ { 0 } ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \Big ) \right\| } \\ & { \quad \leqslant r _ { N } ^ { \prime } \log ^ { 1 / 2 } ( 1 / r _ { N } ^ { \prime } ) + n ^ { - 1 / 2 + 1 / q } \log n . } \end{array}
$$

Therefore, since all singular values of $J _ { 0 }$ are bounded below from zero by Assumption 3.3(d), it follows that

$$
\begin{array} { r l } & { \left\| J _ { 0 } ^ { - 1 } \sqrt { n } \mathbb { E } _ { n } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] + \sqrt { n } ( \check { \theta } _ { 0 } - \theta _ { 0 } ) \right\| } \\ & { \qquad = O _ { P _ { N } } ( n ^ { - 1 / 2 + 1 / q } \log n + r _ { N } ^ { \prime } \log ^ { 1 / 2 } ( 1 / r _ { N } ^ { \prime } ) + ( \epsilon _ { N } + \lambda _ { N } + \lambda _ { N } ^ { \prime } ) \sqrt { n } . } \end{array}
$$

The asserted claim now follows by multiplying both parts of the display by $\Sigma _ { 0 } ^ { - 1 / 2 }$ (under the norm on the left-hand side) and noting that singular values of $\Sigma _ { 0 }$ are bounded below from zero by Assumptions 3.3 and 3.4.

Step 3. Here we derive a bound on $\mathcal { I } _ { 5 }$ in (A.37). We have

$$
\mathcal { T } _ { 5 } \lesssim \operatorname* { s u p } _ { f \in \mathcal { F } _ { 2 } } \vert \mathbb { G } _ { n } ( f ) \vert , \quad \mathcal { F } _ { 2 } = \big \{ \psi _ { j } ( \cdot , \theta , \widehat { \eta } _ { 0 } ) - \psi _ { j } ( \cdot , \theta _ { 0 } , \eta _ { 0 } ) \colon ~ j = 1 , . . . , d _ { \theta } , ~ \| \theta - \theta _ { 0 } \| \leqslant \tau _ { n } \big \} .
$$

To bound $\operatorname* { s u p } _ { f \in { \mathcal { F } } _ { 2 } } | \mathbb { G } _ { n } ( f ) |$ , we apply Lemma 6.2 conditional on $( W _ { i } ) _ { i \in I ^ { c } }$ and $I ^ { c }$ so that $\widehat { \eta _ { 0 } }$ can be treated as fixed. Observe that with $P _ { N }$ -probability $1 - o ( 1 )$ , $\mathrm { s u p } _ { f \in \mathcal { F } _ { 2 } } \| f \| _ { P _ { N } , 2 } \lesssim r _ { N } ^ { \prime }$ where we used Assumption 3.4. Thus, an application of Lemma 6.2 to the empirical process $\{ \mathbb { G } _ { n } ( f ) , f \in \mathcal { F } _ { 2 } \}$ with an envelope $F _ { 2 } = F _ { 1 , \widehat { \eta } \mathrm { 0 } } + F _ { 1 , \eta \mathrm { 0 } }$ and $\sigma = C r _ { N } ^ { \prime }$ for sufficiently large constant $C$ conditional on $( W _ { i } ) _ { i \in I ^ { c } }$ and $I ^ { c }$ yieblds that with $P _ { N }$ -probability $1 - o ( 1 )$ ,

$$
\operatorname* { s u p } _ { f \in \mathcal { F } _ { 2 } } | \mathbb { G } _ { n } ( f ) | \lesssim r _ { N } ^ { \prime } \log ^ { 1 / 2 } ( 1 / r _ { N } ^ { \prime } ) + n ^ { - 1 / 2 + 1 / q } \log n .
$$

This follows since $\| F _ { 2 } \| _ { P , q } = \| F _ { 1 , \widehat { \eta } _ { 0 } } + F _ { 1 , \eta _ { 0 } } \| _ { P , q } \leqslant 2 C _ { 1 }$ by Assumption $3 . 4 ( \mathrm { b } )$ and the triangle inequality, and

$$
\begin{array} { r } { \log \underset { Q } { \operatorname* { s u p } } N ( \epsilon \| F _ { 2 } \| _ { Q , 2 } , \mathcal { F } _ { 2 } , \| \cdot \| _ { Q , 2 } ) \leqslant 2 v \log ( 2 a / \epsilon ) , \quad \mathrm { f o r ~ a l l ~ } 0 < \epsilon \leqslant 1 , } \end{array}
$$

because $\mathcal { F } _ { 2 } \subset \mathcal { F } _ { 1 , \widehat { \eta } _ { 0 } } - \mathcal { F } _ { 1 , \eta _ { 0 } }$ for $\mathcal { F } _ { 1 , \eta }$ defined in Assumption 3.4(b), and

$$
\begin{array} { r l } & { \log \underset { Q } { \operatorname* { s u p } } N ( \epsilon \| F _ { 1 , \hat { \eta } _ { 0 } } + F _ { 1 , \eta _ { 0 } } \| _ { Q , 2 } , \mathcal { F } _ { 1 , \hat { \eta } _ { 0 } } - \mathcal { F } _ { 1 , \eta _ { 0 } } , \| \cdot \| _ { Q , 2 } ) } \\ & { \quad \leqslant \log \underset { Q } { \operatorname* { s u p } } N ( ( \epsilon / 2 ) \| F _ { 1 , \hat { \eta } _ { 0 } } \| _ { Q , 2 } , \mathcal { F } _ { 1 , \hat { \eta } _ { 0 } } , \| \cdot \| _ { Q , 2 } ) + \log \underset { Q } { \operatorname* { s u p } } N ( ( \epsilon / 2 ) \| F _ { 1 , \eta _ { 0 } } \| _ { Q , 2 } , \mathcal { F } _ { 1 , \eta _ { 0 } } , \| \cdot \| _ { Q , 2 } ) } \end{array}
$$

by the proof of Theorem 3 in Andrews (1994b). The claim of this step follows.

Step 4. Here we derive a bound on $\mathcal { I } _ { 3 }$ in (A.33). Let $\bar { \theta } _ { 0 } = \theta _ { 0 } - J _ { 0 } ^ { - 1 } \mathbb { E } _ { n } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ]$ . Then $\| \theta _ { 0 } - \theta _ { 0 } \| = O _ { P _ { N } } ( 1 / \sqrt { n } ) = o _ { P _ { N } } ( \tau _ { n } )$ since $\operatorname { E } _ { P _ { N } } [ \| \sqrt { n \mathbb { E } _ { n } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] } \| ]$ is bounded and the singular values of $J _ { 0 }$ are bounded below from zero by Assumption 3.3(d). Therefore, $\theta _ { 0 } \in$ $\Theta$ with $P _ { N }$ -probability $1 - o ( 1 )$ by Assumption 3.3(a). Hence, with the same probability,

$$
\operatorname* { i n f } _ { \theta \in \Theta } \sqrt { n } \Big \vert \Big \vert \mathbb { E } _ { n } [ \psi ( W ; \theta , \widehat { \eta } _ { 0 } ) ] \Big \vert \Big \vert \leqslant \sqrt { n } \Big \vert \Big \vert \mathbb { E } _ { n } [ \psi ( W ; \bar { \theta } _ { 0 } , \widehat { \eta } _ { 0 } ) ] \Big \vert \Big \vert ,
$$

and so it suffices to show that with $P _ { N }$ -probability $1 - o ( 1 )$ ,

$$
\sqrt { n } \Big \lvert \Big \lvert \mathbb { E } _ { n } \big [ \psi \big ( W ; \bar { \theta } _ { 0 } , \widehat { \eta } _ { 0 } \big ) \big ] \Big \rvert \Big | = O \big ( n ^ { - 1 / 2 + 1 / q } \log n + r _ { N } ^ { \prime } \log ^ { 1 / 2 } ( 1 / r _ { N } ^ { \prime } ) + \lambda _ { N } \sqrt { n } + \lambda _ { N } ^ { \prime } \sqrt { n } \big ) .
$$

To prove it, substitute $\theta = \theta _ { 0 }$ and $\eta = \hat { \eta } _ { 0 }$ into (A.34) and use Taylor’s expansion in (A.35). This shows that with $P _ { N }$ -probab ibty $1 - o ( 1 )$ ,

$$
\begin{array} { r l } & { \sqrt { n } \Big \lvert \Big \lvert \mathbb { E } _ { n } [ \psi ( W ; \bar { \theta } _ { 0 } , \widehat { \eta } _ { 0 } ) ] \Big \rvert \Big \rvert \leqslant { \sqrt { n } } \Big \lvert \Big \lvert \mathbb { E } _ { n } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ] + J _ { 0 } ( \bar { \theta } _ { 0 } - \theta _ { 0 } ) \Big \rvert \Big \rvert + \lambda _ { N } \sqrt { n } + \mathcal { Z } _ { 4 } + \mathcal { Z } _ { 5 } } \\ & { \qquad = \lambda _ { N } \sqrt { n } + \mathcal { Z } _ { 4 } + \mathcal { Z } _ { 5 } , } \end{array}
$$

Combining this with the bounds on $\mathcal { T } _ { 4 }$ and $\mathcal { I } _ { 5 }$ derived above gives the claim of this step and completes the proof of the theorem.

# Proof of Theorems 4.1 and 4.2

Since Theorem 4.1 is a special case of Theorem 4.2 (with $Z = D$ ), it suffices to prove the latter. Also, we only consider the DML estimators based on the score (4.7) and note that the estimators based on the score (4.8) can be treated similarly.

Observe that the score $\psi$ in (4.7) is linear in $\theta$ :

$$
\psi ( W ; \theta , \eta ) = ( Y - D \theta - g ( X ) ) ( Z - m ( X ) ) = \psi ^ { a } ( W ; \eta ) \theta + \psi ^ { b } ( W ; \eta )
$$

$$
\psi ^ { a } ( W ; \eta ) = D ( m ( X ) - Z ) , \quad \psi ^ { b } ( W ; \eta ) = ( Y - g ( X ) ) ( Z - m ( X ) ) .
$$

Therefore, all asserted claims of Theorem 4.2 follow from Theorems 3.1 and 3.2 and Corollary 3.1 as long as we can verify Assumptions 3.1 and 3.2, which we do here. We do so with $\mathcal { T } _ { N }$ being the set of all $\boldsymbol { \eta } = ( g , m )$ consisting of $P$ -square-integrable functions $g$ and $m$ such that

$$
\begin{array} { r l } & { \| \eta - \eta _ { 0 } \| _ { P , \infty } \leqslant C , \quad \| \eta - \eta _ { 0 } \| _ { P , 2 } \leqslant \delta _ { N } , } \\ & { \| m - m _ { 0 } \| _ { P , 2 } \times \| g - g _ { 0 } \| _ { P , 2 } \leqslant \delta _ { N } N ^ { - 1 / 2 } . } \end{array}
$$

Also, we replace the constant $q / 2$ and $\left( \delta _ { N } ^ { \prime } \right) _ { N \geqslant 1 }$ with $\delta _ { N } ^ { \prime } = ( C + 2 \sqrt { C } + 2 ) ( \delta _ { N } \vee N ^ { - [ ( 1 - 4 / q ) \wedge ( 1 / 2 ) ] } )$ $q$ and the sequence $\left( \delta _ { N } \right) _ { N \geqslant 1 }$ in Assumptions 3.1 and 3.2 by for all $N$ (recall that we assume that $q > 4$ , and the analysis in Section 3 only requires that $q > 2$ ; also, $\delta _ { N } ^ { \prime }$ satisfies $\delta _ { N } ^ { \prime } \geqslant N ^ { - [ ( 1 - 4 / q ) \wedge ( 1 / 2 ) ] }$ , which is required in Theorems 3.1 and 3.2). We proceed

in five steps. All bounds in the proof hold uniformly over $P \in \mathcal { P }$ but we omit this qualifier for brevity).

Step 1. We first verify Neyman orthogonality. We have that $\mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) = 0$ by definition of $\theta _ { 0 }$ of $\eta _ { 0 }$ . Also, for any $\eta \ = \ ( g , m ) \in \mathcal { T } _ { N }$ , the Gateaux derivative in the direction $\eta - \eta _ { 0 } = \left( g - g _ { 0 } , m - m _ { 0 } \right)$ is given by

$$
\begin{array} { r l } & { \partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] = \mathrm { E } _ { P } \Big [ ( g ( X ) - g _ { 0 } ( X ) ) ( m _ { 0 } ( X ) - Z ) \Big ] } \\ & { \qquad + \mathrm { E } _ { P } \Big [ ( m _ { 0 } ( X ) - m ( X ) ) ( Y - D \theta _ { 0 } - g _ { 0 } ( X ) ) \Big ] = 0 , } \end{array}
$$

by the law of iterated expectations, since $V = Z - m _ { 0 } ( X )$ and $U = ( Y - D \theta _ { 0 } - g _ { 0 } ( X ) )$ obey $\operatorname { E } _ { P } [ V | X ] = 0$ and $\mathrm { E } _ { P } [ U | Z , X ] = 0$ . This gives Assumption 3.1(d) with $\lambda _ { N } = 0$ .

Step 2. Note that

$$
| J _ { 0 } | = | \mathrm { E } _ { P } [ \psi ^ { a } ( W ; \eta _ { 0 } ) ] | = | \mathrm { E } _ { P } [ D ( m _ { 0 } ( X ) - Z ) ] | = | \mathrm { E } _ { P } [ D V ] | \geqslant c > 0
$$

by Assumption 4.2(c). In addition,

$$
\begin{array} { r l } & { | \mathrm { E } _ { P } [ \psi ^ { a } ( W ; \eta _ { 0 } ) ] | = | \mathrm { E } _ { P } [ D ( m _ { 0 } ( X ) - Z ) ] | \leqslant \| D \| _ { P , 2 } \| m _ { 0 } ( X ) \| _ { P , 2 } + \| D \| _ { P , 2 } \| Z \| _ { P , 2 } } \\ & { \qquad \leqslant 2 \| D \| _ { P , 2 } \| Z \| _ { P , 2 } \leqslant 2 \| D \| _ { P , q } \| Z \| _ { P , q } \leqslant 2 C ^ { 2 } } \end{array}
$$

by the triangle inequality, H¨older’s inequality, Jensen’s inequality, and Assumption 4.2(b). This gives Assumption 3.1(e). Hence, given that Assumptions 3.1(i,ii,iii) hold trivially, Steps 1 and 2 together show that all conditions of Assumption 3.1 hold.

Step 3. Note that Assumption 3.2(a) holds by construction of the set $\mathcal { T } _ { N }$ and Assumption 4.2(e). Also, note that $\psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) = U V$ , and so

$$
{ } \cdot [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ^ { \prime } ] = \mathrm { E } _ { P } [ U ^ { 2 } V ^ { 2 } ] \geqslant c ^ { 4 } > 0
$$

by Assumption $4 . 2 ( \mathrm { c } )$ , which gives Assumption 3.2(d).

Step 4. Here we verify Assumption 3.2(b). For any $\eta = ( g , m ) \in \mathcal { T } _ { N }$ , we have

$$
\begin{array} { r l } {  { ( \mathrm { E } _ { P } [ \| \psi ^ { a } ( W ; \eta ) \| ^ { q / 2 } ] ) ^ { 2 / q } = \| \psi ^ { a } ( W ; \eta ) \| _ { P , q / 2 } = \| D ( m ( X ) - Z ) \| _ { P , q / 2 } } } \\ & { \leqslant \| D ( m ( X ) - m _ { 0 } ( X ) ) \| _ { P , q / 2 } + \| D m _ { 0 } ( X ) \| _ { P , q / 2 } + \| D Z \| _ { P , q / 2 } } \\ & { \leqslant \| D \| _ { P , q } \| m ( X ) - m _ { 0 } ( X ) \| _ { P , q } + \| D \| _ { P , q } \| m _ { 0 } ( X ) \| _ { P , q } + \| D \| _ { P , q } \| Z \| _ { P , q } } \\ & { \leqslant C \| D \| _ { P , q } + 2 \| D \| _ { P , q } \| Z \| _ { P , q } \leqslant 3 C ^ { 2 } } \end{array}
$$

by Assumption 4.2(b), which gives the bound on $m _ { N } ^ { \prime }$ in Assumption 3.2(b). Also, since

$$
| \mathrm { E } _ { P } [ ( D - r _ { 0 } ( X ) ) ( Z - m _ { 0 } ( X ) ) ] | = | \mathrm { E } _ { P } [ D V ] | \geqslant c
$$

by Assumption $4 . 2 ( \mathrm { c } )$ , it follows that $\theta _ { 0 }$ satisfies

$$
\begin{array} { r l } & { | \theta _ { 0 } | = \frac { \left| \operatorname { E } _ { P } \left[ ( Y - \ell _ { 0 } ( X ) ) ( Z - m _ { 0 } ( X ) ) \right] \right| } { \left| \operatorname { E } _ { P } \left[ ( D - r _ { 0 } ( X ) ) ( Z - m _ { 0 } ( X ) ) \right] \right| } } \\ & { \quad \leqslant c ^ { - 1 } \Big ( \| Y \| _ { P , 2 } + \| \ell _ { 0 } ( X ) \| _ { P , 2 } \Big ) \Big ( \| Z \| _ { P , 2 } + \| m _ { 0 } ( X ) \| _ { P , 2 } \Big ) } \\ & { \quad \leqslant 4 c ^ { - 1 } \| Y \| _ { P , 2 } \| Z \| _ { P , 2 } \leqslant 4 C ^ { 2 } / c . } \end{array}
$$

Hence,

$$
\begin{array} { r l } { \big ( \mathbb { E } _ { \mathcal { P } } \big [ \| \psi ( W ; \theta _ { 0 } , \eta ) \| ^ { \mathcal { I } / 2 } \big ] \big ) ^ { 2 / 2 } } & { = \big | \psi ( W ; \theta _ { 0 } , \eta ) \big | | _ { P , q } / 2 } \\ & { = \big \| ( Y - D \theta _ { 0 } - g ( X ) ) ( \mathcal { Z } - m ( X ) ) \big \| _ { P , q / 2 } } \\ & { \leqslant \big \| [ C - m ( X ) ) \big \| _ { P , q / 2 } + \big \| ( g ( X ) - g _ { 0 } ( X ) ) ( \mathcal { Z } - m ( X ) ) \big \| _ { P , q / 2 } } \\ & { \leqslant \big \| [ \eta | _ { P , q } ] \big | _ { \mathcal { P } _ { q } } \big \| _ { \mathcal { P } _ { q } } } \\ & { \leqslant \big ( \big \| [ \mathcal { W } \big [ \| _ { P , q } + C \big ( X \big ) \| _ { P , q } + \big \| \rho ( X ) - g _ { 0 } ( X ) \big \| _ { P , q } \big \| \mathcal { Z } - m \big ( X ) \big \| _ { P , q } } \\ & { \leqslant \big ( \big \| \mathcal { W } - D \theta _ { 0 } \big \| _ { P , q } + \big \| g _ { 0 } ( X ) \big \| _ { P , q } + C \big ) } \\ & { \qquad \times \big ( \big \| \mathcal { Z } \big \| _ { P , q } + \big \| m _ { 0 } ( X ) \big \| _ { P , q } + \big \| m ( X ) - m _ { 0 } ( X ) \big \| _ { P , q } \big ) } \\ & { \leqslant \big ( 2 \big \| Y - D \theta _ { 0 } \big \| _ { P , q } + C \big ) ( 2 \big \| Z \big \| _ { P , q } + C ) } \\ & { \leqslant ( 2 \big \| Y \big [ \mathcal { P } _ { q } + 2 \big \| D \big | _ { P , q } ) \big \| _ { 0 } + C \big ) ( 2 \big \| Z \big \| _ { P , q } + C ) } \\ & { \leqslant 3 C ( 3 C + 8 C ^ { 3 } ) \epsilon , } \end{array}
$$

where we used the fact that since $g _ { 0 } ( X ) \ = \ \operatorname { E } _ { P } [ Y \ - \ D \theta _ { 0 } \ | \ X ]$ , $\| g _ { 0 } ( X ) \| _ { P , q } \leqslant \| Y -$ $D \theta _ { 0 } \| _ { P , q }$ by Jensen’s inequality. This gives the bound on $m _ { N }$ in Assumption 3.2(b). Hence, Assumption 3.2(b) holds.

Step 5. Finally, we verify Assumption 3.2(c). For any $\eta = ( g , m ) \in \mathcal { T } _ { N }$ , we have

$$
\begin{array} { r l } { \| \mathrm { E } _ { P } [ \psi ^ { a } ( W ; \eta ) ] - \mathrm { E } _ { P } [ \psi ^ { a } ( W ; \eta _ { 0 } ) ] \| } & { = | \mathrm { E } _ { P } [ \psi ^ { a } ( W ; \eta ) - \psi ^ { a } ( W ; \eta _ { 0 } ) ] | } \\ & { = | \mathrm { E } _ { P } [ D ( m ( X ) - m _ { 0 } ( X ) ) ] | } \\ & { \leqslant \| D \| _ { P , 2 } \| m ( X ) - m _ { 0 } ( X ) \| _ { P , 2 } \leqslant C \delta _ { N } \leqslant \delta _ { N } ^ { \prime } , } \end{array}
$$

which gives the bound on $r _ { N }$ in Assumption 3.2(c). Further,

$$
\begin{array} { r l } & { ( \mathrm { E } _ { P } [ \| \psi ( W ; \theta _ { 0 } , \eta ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } ] ) ^ { 1 / 2 } } \\ & { \qquad = \| \psi ( W ; \theta _ { 0 } , \eta ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \| _ { P ^ { 2 } } } \\ & { \qquad = \| ( U + g _ { 0 } ( X ) - g ( X ) ) ( Z - m ( X ) ) - U ( Z - m _ { 0 } ( X ) ) \| _ { P , 2 } } \\ & { \qquad \leqslant \| U ( m ( X ) - m _ { 0 } ( X ) ) \| _ { P , 2 } + \| ( g ( X ) - g _ { 0 } ( X ) ) ( Z - m ( X ) ) \| _ { P , 2 } } \\ & { \qquad \leqslant \sqrt { C } \| m ( X ) - m _ { 0 } ( X ) \| _ { P , 2 } + \| V ( g ( X ) - g _ { 0 } ( X ) ) \| _ { P , 2 } } \\ & { \qquad + \| ( g ( X ) - g _ { 0 } ( X ) ) ( m ( X ) - m _ { 0 } ( X ) ) \| _ { P , 2 } } \\ & { \qquad \leqslant \sqrt { C } \| m ( X ) - m _ { 0 } ( X ) \| _ { P , 2 } + \sqrt { C } \| g ( X ) - g _ { 0 } ( X ) \| _ { P , 2 } + C \| m ( X ) - m _ { 0 } ( X ) \| _ { P , 2 } } \\ & { \qquad \leqslant ( 2 \sqrt { C } + C ) \delta _ { N } \leqslant \delta _ { N } ^ { \prime } , } \end{array}
$$

which gives the bound on $r _ { N } ^ { \prime }$ in Assumption 3.2(c). Finally, let

$$
f ( r ) : = \mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) ] , \quad r \in ( 0 , 1 ) .
$$

Then for any $r \in ( 0 , 1 )$ ,

$$
f ( r ) = \mathrm { E } _ { P } [ ( U - r ( g ( X ) - g _ { 0 } ( X ) ) ) ( V - r ( m ( X ) - m _ { 0 } ( X ) ) ) ] ,
$$

and so

$$
\begin{array} { r l } & { \partial f ( r ) = - \mathrm { E } _ { P } [ ( g ( X ) - g _ { 0 } ( X ) ) ( V - r ( m ( X ) - m _ { 0 } ( X ) ) ) ] } \\ & { \phantom { \partial } - \mathrm { E } _ { P } [ ( U - r ( g ( X ) - g _ { 0 } ( X ) ) ) ( m ( X ) - m _ { 0 } ( X ) ) ] , } \\ & { \partial ^ { 2 } f ( r ) = 2 \mathrm { E } _ { P } [ ( g ( X ) - g _ { 0 } ( X ) ) ( m ( X ) - m _ { 0 } ( X ) ) ] . } \end{array}
$$

Hence,

$| \partial ^ { 2 } f ( r ) | \leqslant 2 \| g ( X ) - g _ { 0 } ( X ) \| _ { P , 2 } \times \| m ( X ) - m _ { 0 } ( X ) \| _ { P , 2 } \leqslant 2 \delta _ { N } N ^ { - 1 / 2 } \leqslant \delta _ { N } ^ { \prime } N ^ { - 1 / 2 } ,$ which gives the bound on $\lambda _ { N } ^ { \prime }$ in Assumption 3.2(c). Thus, all conditions of Assumptions 3.1 are verified. This completes the proof.

# Proof of Theorems 5.1 and 5.2

The proof of Theorem 5.2 is similar to that of Theorem 5.1 and therefore omitted. In turn, regarding Theorem 5.1, we show the proof for the case of ATE and note that the proof for the case of ATTE is similar.

Observe that the score $\psi$ in (5.3) is linear in $\theta$ :

$$
\psi ( W ; \theta , \eta ) = \psi ^ { a } ( W ; \eta ) \theta + \psi ^ { b } ( W ; \eta ) , \quad \psi ^ { a } ( W ; \eta ) = - 1 ,
$$

$$
\psi ^ { b } ( W ; \eta ) = \left( g ( 1 , X ) - g ( 0 , X ) \right) + \frac { D ( Y - g ( 1 , X ) ) } { m ( X ) } - \frac { ( 1 - D ) ( Y - g ( 0 , X ) ) } { 1 - m ( X ) } .
$$

Therefore, all asserted claims of Theorem 5.1 follow from Theorems 3.1 and 3.2 and Corollary 3.1 as long as we can verify Assumptions 3.1 and 3.2, which we do here. We do so with $\mathcal { T } _ { N }$ being the set of all $\boldsymbol { \eta } = ( g , m )$ consisting of $P$ -square-integrable functions $g$ and $m$ such that

$$
\begin{array} { r l } & { \| \eta - \eta _ { 0 } \| _ { P , q } \leqslant C , \quad \| \eta - \eta _ { 0 } \| _ { P , 2 } \leqslant \delta _ { N } , \quad \| m - 1 / 2 \| _ { P , \infty } \leqslant 1 / 2 - \varepsilon , } \\ & { \| m - m _ { 0 } \| _ { P , 2 } \times \| g - g _ { 0 } \| _ { P , 2 } \leqslant \delta _ { N } N ^ { - 1 / 2 } . } \end{array}
$$

Also, we replace the sequence $\left( \delta _ { N } \right) _ { N \geqslant 1 }$ in Assumptions 3.1 and 3.2 by $\left( \delta _ { N } ^ { \prime } \right) _ { N \geqslant 1 }$ with $\delta _ { N } ^ { \prime } = C _ { \varepsilon } ( \delta _ { N } \vee N ^ { - [ ( 1 - 4 / q ) \wedge ( 1 / 2 ) ] } )$ for all $N$ , where $C _ { \varepsilon }$ is a sufficiently large constant that depends only on $\varepsilon$ and $C$ (note that $\delta _ { N } ^ { \prime }$ satisfies $\delta _ { N } ^ { \prime } \geqslant N ^ { - [ ( 1 - 4 / q ) \wedge ( 1 / 2 ) ] }$ , which is required in Theorems 3.1 and 3.2). We proceed in five steps. All bounds in the proof hold uniformly over $P \in \mathcal { P }$ but we omit this qualifier for brevity.

Step 1. We first verify Neyman orthogonality. We have that $\operatorname { E } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) ~ = ~ 0$ by definition of $\theta _ { 0 }$ and $\eta _ { 0 }$ . Also, for any $\eta = ( g , m ) \in \mathcal { T } _ { N }$ , the Gateaux derivative in the direction $\eta - \eta _ { 0 } = \left( g - g _ { 0 } , m - m _ { 0 } \right)$ is given by

$$
\begin{array} { r l } & { \partial _ { \eta } \mathrm { E } _ { P } \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) [ \eta - \eta _ { 0 } ] = \mathrm { E } _ { P } \Big [ g ( 1 , X ) - g _ { 0 } ( 1 , X ) \Big ] - \mathrm { E } _ { P } \Big [ g ( 0 , X ) - g _ { 0 } ( 0 , X ) \Big ] } \\ & { - \mathrm { E } _ { P } \Big [ \frac { D \big ( g ( 1 , X ) - g _ { 0 } ( 1 , X ) \big ) } { m _ { 0 } ( X ) } \Big ] + \mathrm { E } _ { P } \Big [ \frac { ( 1 - D ) \big ( g ( 0 , X ) - g _ { 0 } ( 0 , X ) \big ) } { 1 - m _ { 0 } ( X ) } \Big ] } \\ & { - \mathrm { E } _ { P } \Big [ \frac { D \big ( Y - g _ { 0 } ( 1 , X ) \big ) ( m ( X ) - m _ { 0 } ( X ) \big ) } { m _ { 0 } ^ { 2 } ( X ) } \Big ] - \mathrm { E } _ { P } \Big [ \frac { ( 1 - D ) \big ( Y - g _ { 0 } ( 0 , X ) \big ) \big ( m ( X ) - m _ { 0 } ( X ) \big ) } { ( 1 - m _ { 0 } ( X ) ) ^ { 2 } } \Big ] , } \end{array}
$$

which is $0$ by the law of iterated expectations, since

$$
\operatorname { E } _ { P } [ D \mid X ] = m _ { 0 } ( X ) , \quad \operatorname { E } _ { P } [ 1 - D \mid X ] = 1 - m _ { 0 } ( X ) ,
$$

$$
\operatorname { E } _ { P } [ D ( Y - g _ { 0 } ( 1 , X ) ) ~ | ~ X ] = 0 , \quad \operatorname { E } _ { P } [ ( 1 - D ) ( Y - g _ { 0 } ( 0 , X ) ) ~ | ~ X ] = 0 .
$$

This gives Assumption $3 . 1 ( \mathrm { d } )$ with $\lambda _ { N } = 0$ .

Step 2. Note that $J _ { 0 } ~ = ~ - 1$ , and so Assumption 3.1(e) holds trivially. Hence, given that Assumptions 3.1(i,ii,iii) hold trivially as well, Steps 1 and 2 together show that all conditions of Assumption 3.1 hold.

Step 3. Note that Assumption 3.2(a) holds by construction of the set $\mathcal { T } _ { N }$ and Assumption 5.1(f). Also,

$$
\begin{array} { r l } & { \mathbb { E } _ { F } \bigg [ \mathrm { t } ^ { 2 } ( W ; \theta _ { 1 } , \eta _ { 0 } ) \Big ] = \mathbb { E } _ { F } \bigg [ \mathbb { E } _ { F } \langle \theta | \hat { \mathcal { H } } ^ { 2 } ( W ; \theta _ { 1 } , \eta _ { 0 } ) | \mathcal { X } \bigg ] \bigg ] } \\ & { \qquad = \mathbb { E } _ { F } \Big [ \mathrm { E } _ { F } \langle | \theta ( 1 , X ) - \theta _ { 1 } ( 0 , X ) - \theta _ { 1 } | \mathcal { X } \rangle ^ { 2 } \Big ] \ ( X ) } \\ & { \qquad + \mathbb { E } _ { F } \bigg [ \Big ( \frac { | \mathcal { D } ( Y - \theta _ { 1 } ( 1 , X ) ) | } { m _ { 1 } ( X ) } - \frac { \big ( 1 - | \mathcal { D } \rangle \big ) ( Y ^ { 2 } - \theta _ { 0 } ( 0 , X ) ) } { 1 - m _ { 0 } ( X ) } \Big ) ^ { 2 } \ | X \Big ] \bigg ] } \\ & { \qquad \ge \mathbb { E } _ { F } \bigg [ \Big ( \frac { \mathcal { D } ( Y - \theta _ { 0 } ( 1 , X ) ) } { m _ { 0 } ( X ) } - \frac { \big ( 1 - | \mathcal { D } \rangle \big ) ( Y ^ { 2 } - \theta _ { 0 } ( 0 , X ) ) } { 1 - m _ { 0 } ( X ) } \Big ) ^ { 2 } \bigg ] } \\ & { \qquad = \mathbb { E } _ { F } \bigg [ \Big ( \frac { \mathcal { D } ^ { 2 } ( Y - \theta _ { 0 } ( 1 , X ) ) ) ^ { 2 } } { m _ { 0 } ( X ) ^ { 2 } } + \frac { \big ( 1 - | \mathcal { D } \rangle \big ) ^ { 2 } ( Y - \theta _ { 0 } ( 0 , X ) ) } { ( 1 - m _ { 0 } ( X ) ) ^ { 2 } } \Big ) ^ { 2 } \bigg ] } \\ & { \qquad = \mathbb { E } _ { F } \bigg [ \frac { \mathcal { D } ^ { 2 } ( Y - \theta _ { 0 } ( 1 , X ) ) ^ { 2 } } { m _ { 0 } ( X ) ^ { 2 } } + \frac { \big ( 1 - | \mathcal { D } \rangle \big ) ^ { 2 } ( Y - \theta _ { 0 } ( 0 , X ) ) ^ { 2 } } { ( 1 - m _ { 0 } ( X ) ) ^ { 2 } } \bigg ] } \\ & { \qquad \ge \frac { 1 } { ( 1 - \epsilon ) ^ { 2 } } \mathbb { E } _ { F } \bigg [ \mathcal { D } ^ { 2 } ( Y - \theta _ { 0 } ( 1 , X ) ) ^ { 2 } + ( 1 - \mathcal { D } ) ^ { 2 } ( Y - \theta _ { 0 } ( 0 , X ) ) ^ { 2 } \bigg ] } \\ &  \qquad = \frac { 1 } { ( 1 - \epsilon ) ^ { 2 } } \mathbb { E } _ { F } \bigg [ \mathcal { D } \theta ^ { 2 } + ( 1 - \theta ) \mathcal  D  \end{array}
$$

This gives Assumption 3.2(d).

Step 4. Here we verify Assumption 3.2(b). We have

$$
\begin{array} { r l } & { \| g _ { 0 } ( D , X ) \| _ { P , q } = ( \mathrm { E } _ { P } [ | g _ { 0 } ( D , X ) | ^ { q } ] ) ^ { 1 / q } } \\ & { \qquad \geqslant \Big ( \mathrm { E } _ { P } \Big [ | g _ { 0 } ( 1 , X ) | ^ { q } \mathrm { P } _ { P } ( D = 1 \ | \ X ) + | g _ { 0 } ( 0 , X ) | ^ { q } \mathrm { P } _ { P } ( D = 0 \ | \ X ) \Big ] \Big ) ^ { 1 / q } } \\ & { \qquad \geqslant \varepsilon ^ { 1 / q } \Big ( \mathrm { E } _ { P } [ | g _ { 0 } ( 1 , X ) | ^ { q } ] + \mathrm { E } _ { P } [ | g _ { 0 } ( 0 , X ) | ^ { q } ] \Big ) ^ { 1 / q } } \\ & { \qquad \geqslant \varepsilon ^ { 1 / q } \Big ( \mathrm { E } _ { P } [ | g _ { 0 } ( 1 , X ) | ^ { q } ] \vee \mathrm { E } _ { P } [ | g _ { 0 } ( 0 , X ) | ^ { q } ] \Big ) ^ { 1 / q } } \\ & { \qquad \geqslant \varepsilon ^ { 1 / q } \Big ( \| g _ { 0 } ( 1 , X ) \| _ { P , q } \vee \| g _ { 0 } ( 0 , X ) \| _ { P , q } \Big ) , } \end{array}
$$

where in the third line, we used the facts that $\mathrm { P } _ { P } ( D = 1 \mid X ) = m _ { 0 } ( X ) \geqslant \varepsilon$ and $\operatorname { P } _ { P } ( D = 0 \mid X ) = 1 - m _ { 0 } ( X ) \geqslant \varepsilon$ . Hence, given that $\| g _ { 0 } ( D , X ) \| _ { P , q } \leqslant \| Y \| _ { P , q } \leqslant C$ by Jensen’s inequality and Assumption 5.1(b), it follows that

$$
\| g _ { 0 } ( 1 , X ) \| _ { P , q } \leqslant C / \varepsilon ^ { 1 / q } \quad \mathrm { ~ a n d ~ } \quad \| g _ { 0 } ( 0 , X ) \| _ { P , q } \leqslant C / \varepsilon ^ { 1 / q } .
$$

Similarly, for any $\eta \in ( g , m ) \in \mathcal { T } _ { N }$ ,

$$
\| g ( 1 , X ) - g _ { 0 } ( 1 , X ) \| _ { P , q } \leqslant C / \varepsilon ^ { 1 / q } \quad \mathrm { ~ a n d ~ } \quad \| g ( 0 , X ) - g _ { 0 } ( 0 , X ) \| _ { P , q } \leqslant C / \varepsilon ^ { 1 / q }
$$

since $\| g ( D , X ) - g _ { 0 } ( D , X ) \| _ { P , q } \leqslant C$ . In addition,

$$
| \theta _ { 0 } | = | \mathrm { E } _ { P } [ g _ { 0 } ( 1 , X ) - g _ { 0 } ( 0 , X ) ] | \leqslant \| g _ { 0 } ( 1 , X ) \| _ { P , 2 } + \| g _ { 0 } ( 0 , X ) \| _ { P , 2 } \leqslant 2 C / \varepsilon ^ { 1 / q } .
$$

Therefore, for any $\eta = ( g , m ) \in \mathcal { T } _ { N }$ , we have

$$
\begin{array} { r l } & { ( \mathrm { E } _ { P } [ \| \psi ( W ; \theta _ { 0 } , \eta ) \| ^ { q } ] ) ^ { 1 / q } = \| \psi ( W ; \theta _ { 0 } , \eta ) \| _ { P , q } } \\ & { \qquad \leqslant ( 1 + \varepsilon ^ { - 1 } ) \Big ( \| g ( 1 , X ) \| _ { P , q } + \| g ( 0 , X ) \| _ { P , q } \Big ) + 2 \| Y \| _ { P , q } / \varepsilon + | \theta _ { 0 } | } \\ & { \qquad \leqslant ( 1 + \varepsilon ^ { - 1 } ) \Big ( \| g ( 1 , X ) - g _ { 0 } ( 1 , X ) \| _ { P , q } + \| g ( 0 , X ) - g _ { 0 } ( 0 , X ) \| _ { P , q } \Big ) } \\ & { \qquad + ( 1 + \varepsilon ^ { - 1 } ) \Big ( \| g _ { 0 } ( 1 , X ) \| _ { P , q } + \| g _ { 0 } ( 0 , X ) \| _ { P , q } \Big ) + 2 C / \varepsilon + 2 C / \varepsilon ^ { 1 / q } } \\ & { \qquad \leqslant 4 C ( 1 + \varepsilon ^ { - 1 } ) / \varepsilon ^ { 1 / q } + 2 C / \varepsilon + 2 C / \varepsilon ^ { 1 / q } . } \end{array}
$$

This gives the bound on $m _ { N }$ in Assumption 3.2(b). Also, we have

$$
( \mathrm { E } _ { P } [ | \psi ^ { a } ( W ; \eta ) | ^ { q } ] ) ^ { 1 / q } = 1 .
$$

This gives the bound on $m _ { N } ^ { \prime }$ in Assumption 3.2(b). Hence, Assumption $3 . 2 ( \mathrm { b } )$ holds.

Step 5. Finally, we verify Assumption 3.2(c). For any $\eta = ( g , m ) \in \mathcal { T } _ { N }$ , we have

$$
\| \mathrm { E } _ { P } [ \psi ^ { a } ( W ; \eta ) - \psi ^ { a } ( W ; \eta _ { 0 } ) ] \| = | 1 - 1 | = 0 \leqslant \delta _ { N } ^ { \prime }
$$

which gives the bound on $r _ { N }$ in Assumption 3.2(c). Further, by the triangle inequality,

$$
\begin{array} { r l } & { ( \mathrm { E } _ { P } [ \| \psi ( W ; \theta _ { 0 } , \eta ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } ] ) ^ { 1 / 2 } = \| \psi ( W ; \theta _ { 0 } , \eta ) - \psi ( W ; \theta _ { 0 } ; \eta _ { 0 } ) \| _ { P , 2 } } \\ & { \qquad \leqslant { \cal T } _ { 1 } + { \cal T } _ { 2 } + { \cal T } _ { 3 } , } \end{array}
$$

where

$$
\begin{array} { l } { { \displaystyle \mathcal { Z } _ { 1 } : = \left\| g ( 1 , X ) - g _ { 0 } ( 1 , X ) \right\| _ { P , 2 } } + \left\| g ( 0 , X ) - g _ { 0 } ( 0 , X ) \right\| _ { P , 2 } , }  \\ { ~ { \displaystyle \mathcal { Z } _ { 2 } : = \left\| \frac { D ( Y - g ( 1 , X ) ) } { m ( X ) } - \frac { D ( Y - g _ { 0 } ( 1 , X ) ) } { m _ { 0 } ( X ) } \right\| _ { P , 2 } } , } \\ { ~ { \displaystyle \mathcal { Z } _ { 3 } : = \left\| \frac { ( 1 - D ) ( Y - g ( 0 , X ) ) } { 1 - m ( X ) } - \frac { ( 1 - D ) ( Y - g _ { 0 } ( 0 , X ) ) } { 1 - m _ { 0 } ( X ) } \right\| _ { P , 2 } } . } \end{array}
$$

To bound $\mathcal { I } _ { 1 }$ , note that by the same argument as that used in Step 4,

$\| g ( 1 , X ) - g _ { 0 } ( 1 , X ) \| _ { P , 2 } \leqslant \delta _ { N } / \varepsilon ^ { 1 / 2 } \quad \mathrm { ~ a n d ~ } \quad \| g ( 0 , X ) - g _ { 0 } ( 0 , X ) \| _ { P , 2 } \leqslant \delta _ { N } / \varepsilon ^ { 1 / 2 } ,$ and so $\mathcal { T } _ { 1 } \leqslant 2 \delta _ { N } / \varepsilon ^ { 1 / 2 }$ . To bound $\mathcal { I } _ { 2 }$ , we have

$$
\begin{array} { r l } & { \mathcal { T } _ { 2 } \leqslant \varepsilon ^ { - 2 } \Big \| D m _ { 0 } ( X ) ( Y - g ( 1 , X ) ) - D m ( X ) ( Y - g _ { 0 } ( 1 , X ) ) \Big \| _ { P , 2 } } \\ & { \quad \leqslant \varepsilon ^ { - 2 } \Big \| m _ { 0 } ( X ) ( g _ { 0 } ( 1 , X ) + U - g ( 1 , X ) ) - m ( X ) U \Big \| _ { P , 2 } } \\ & { \quad \leqslant \varepsilon ^ { - 2 } \Big ( \Big \| m _ { 0 } ( X ) ( g ( 1 , X ) - g _ { 0 } ( 1 , X ) ) \Big \| _ { P , 2 } + \Big \| ( m ( X ) - m _ { 0 } ( X ) ) U \Big \| _ { P , 2 } \Big ) } \\ & { \quad \leqslant \varepsilon ^ { - 2 } \Big ( \| g ( 1 , X ) - g _ { 0 } ( 1 , X ) \| _ { P , 2 } + \sqrt { C } \| m ( X ) - m _ { 0 } ( X ) \| _ { P , 2 } \Big ) \leqslant \varepsilon ^ { - 2 } ( \varepsilon ^ { - 1 / 2 } + \sqrt { C } ) \delta _ { N } , } \end{array}
$$

where the first inequality follows from the bounds $\varepsilon \leqslant m _ { 0 } ( X ) \leqslant 1 - \varepsilon$ and $\varepsilon \leqslant m ( X ) \leqslant$ $1 - \varepsilon$ , the second from the facts that $D \in \{ 0 , 1 \}$ and for $D = 1$ , $\begin{array} { r } { Y = g _ { 0 } ( 1 , X ) + U } \end{array}$ , the third from the triangle inequality, the fourth from the facts that $m _ { 0 } ( X ) \leqslant 1$ and $\mathrm { E } _ { P } [ U ^ { 2 } \mid$ $X ] \leqslant C$ , and the fifth from (A.39). Similarly, $\mathcal { T } _ { 3 } \leqslant \varepsilon ^ { - 2 } ( \varepsilon ^ { - 1 / 2 } + \sqrt { C } ) \delta _ { N }$ . Combining these inequalities shows that

$$
( \mathrm { E } _ { P } [ \| \psi ( W ; \theta _ { 0 } , \eta ) - \psi ( W ; \theta _ { 0 } , \eta _ { 0 } ) \| ^ { 2 } ] ) ^ { 1 / 2 } \leqslant 2 ( \varepsilon ^ { - 1 / 2 } + \varepsilon ^ { - 5 / 2 } + \sqrt { C } \varepsilon ^ { - 2 } ) \delta _ { N } \leqslant \delta _ { N } ^ { \prime } ,
$$

as long as $C _ { \varepsilon }$ in the definition of $\delta _ { N } ^ { \prime }$ satisfies $C _ { \varepsilon } \geqslant 2 ( \varepsilon ^ { - 1 / 2 } + \varepsilon ^ { - 5 / 2 } + \sqrt { C } \varepsilon ^ { - 2 } )$ . This gives the bound on $r _ { N } ^ { \prime }$ in Assumption 3.2(c).

Finally, let

$$
f ( r ) : = \mathrm { E } _ { P } [ \psi ( W ; \theta _ { 0 } , \eta _ { 0 } + r ( \eta - \eta _ { 0 } ) ) ] , \quad r \in ( 0 , 1 )
$$

Then for any $r \in ( 0 , 1 )$ ,

$$
\begin{array} { r l } { \delta ^ { 2 } f ( r ) = \mathrm { E } \rho \left[ \frac { D ( \rho ( 1 , X ) - \rho _ { 0 } ( 1 , X ) ) ( m ( X ) - m _ { 0 } ( X ) ) ) } { ( m _ { 0 } ( X ) + r ( m ( X ) - m _ { 0 } ( X ) ) ) ^ { 2 } } \right] } \\ & { + \mathrm { E } \rho \left[ \frac { ( 1 - D ) ( \rho ( 0 , X ) - \rho _ { 0 } ( 0 , X ) ) ( m ( X ) - 1 - m _ { 0 } ( X ) ) ) } { ( 1 - m _ { 0 } ( X ) - r ( m ( X ) - r ( \rho ( X ) ) ) ) ^ { 2 } } \right] } \\ & { + \mathrm { E } \rho \left[ \frac { ( \rho ( 1 , X ) - \rho _ { 0 } ( 1 , X ) ) ( m ( X ) - m _ { 0 } ( X ) ) - m _ { 0 } ( X ) ) } { ( m _ { 0 } ( X ) + r ( m ( X ) - m _ { 0 } ( X ) ) ) ^ { 2 } } \right] } \\ & { + 2 \mathrm { E } \rho \left[ \frac { ( \rho ( 1 , X ) - \rho _ { 0 } ( 1 , X ) ) ( m ( X ) - m _ { 0 } ( X ) ) } { ( m _ { 0 } ( X ) + r ( m ( X ) - m _ { 0 } ( X ) ) ) ^ { 2 } } \right] } \\ & { + 2 \mathrm { E } \rho \left[ \frac { D ( ( Y - \phi _ { 0 } ( 1 , X ) - r ( \rho ( 1 , X ) - \rho _ { 0 } ( 1 , X ) ) - g _ { 0 } ( 1 , X ) ) ) ( m ( X ) - m _ { 0 } ( X ) ) ^ { 2 } ) } { ( m _ { 0 } ( X ) + r ( m ( X ) - m _ { 0 } ( X ) ) ) ^ { 3 } } \right] } \\ & { + \mathrm { E } \rho \left[ \frac { ( \rho ( 0 , X ) - \rho _ { 0 } ( 0 , X ) ) ( m ( X ) - m _ { 0 } ( X ) ) } { ( 1 - m _ { 0 } ( X ) - r ( m ( X ) - m _ { 0 } ( X ) ) ) ^ { 2 } } \right] } \\ & { - 2 \mathrm { E } \rho \left[ \frac { ( 1 - D ) ( Y - \rho _ { 0 } ( 0 , X ) - r ( \rho ( 0 , X ) - g _ { 0 } ( 0 , X ) ) ) ( m ( X ) - m _ { 0 } ( X ) ) ^ { 2 } } { ( 1 - m _ { 0 } ( X ) - r ( m _ { 0 } ( X ) ) - r ( m ( X ) - m _ { 0 } ( X ) ) ) ( m ( X ) - m _ { 0 } ( X ) ) ^ { 2 } } \right] , } \end{array}
$$

and so, given that

$$
D ( Y - g _ { 0 } ( 1 , X ) ) = D U , \quad ( 1 - D ) ( Y - g _ { 0 } ( 0 , X ) ) = ( 1 - D ) U ,
$$

$$
\mathrm { E } _ { P } [ U \mid D , X ] = 0 , \quad | m ( X ) - m _ { 0 } ( X ) | \leqslant 2 ,
$$

it follows that for some constant $C _ { \varepsilon } ^ { \prime }$ that depends only on $\varepsilon$ and C,

as long as the constant $C _ { \varepsilon }$ in the definition of $\delta _ { N } ^ { \prime }$ satisfies $C _ { \varepsilon } \geqslant C _ { \varepsilon } ^ { \prime }$ . This gives the bound on $\lambda _ { N } ^ { \prime }$ in Assumption 3.2(c). Thus, all conditions of Assumptions 3.1 are verified. This completes the proof.
