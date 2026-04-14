
## pymupdf4llm output

50

## _CCDDHNR_

Poterba, J. M., S. F. Venti, and D. A. Wise (1994b). Do 401(k) contributions crowd out other personal saving? _Journal of Public Economics 58_ , 1–32.

Robins, J., L. Li, R. Mukherjee, E. Tchetgen, and A. van der Vaart (2017). Minimax estimation of a functional on a structured high dimensional model. _Annals of Statistics, forthcoming_ .

Robins, J., L. Li, E. Tchetgen, and A. van der Vaart (2008). Higher order influence functions and minimax estimation of nonlinear functionals. In D. Nolan and T. Speed (Eds.), _Probability and Statistics: Essays in Honor of David A. Freedman_ , pp. 335–421. Institute of Mathematical Statistics.

Robins, J. and A. Rotnitzky (1995). Semiparametric efficiency in multivariate regression models with missing data. _Journal of the American Statistical Association 90_ , 122–129.

Robins, J., P. Zhang, R. Ayyagari, R. Logan, E. Tchetgen, L. Li, A. Lumley, and A. van der Vaart (2013). New statistical approaches to semiparametric regression with application to air pollution research. Research Report 175, Health Effects Institute. Robinson, P. M. (1988). Root- _N_ -consistent semiparametric regression. _Econometrica 56_ , 931–954.

Rosenbaum, P. R. and D. B. Rubin (1983). The central role of the propensity score in observational studies for causal effects. _Biometrika 70_ , 41–55.

Scharfstein, D. O., A. Rotnitzky, and J. M. Robins (1999). Rejoinder to “adjusting for non-ignorable drop-out using semiparametric non-response models”. _Journal of the American Statistical Association 94_ , 1135–1146.

Schick, A. (1986). On asymptotically efficient estimation in semiparametric models. _Annals of Statistics 14_ , 1139–1151. Severini, T. A. and W. H. Wong (1992). Profile likelihood and conditionally parametric models. _The Annals of Statistics 20_ , 1768–1802.

Toth, B. and M. J. van der Laan (2016). TMLE for marginal structural models based on an instrument. Working Paper 350, U.C. Berkeley Division of Biostatistics Working Paper Series.

van de Geer, S., P. B¨uhlmann, Y. Ritov, and R. Dezeure (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. _Annals of Statistics 42_ , 1166–1202. arXiv, 2013.

van der Laan, M. and D. Rubin (2006). Targeted maximum likelihood learning. Working Paper 213, UC Berkeley Division of Biostatistics Working Paper Series.

van der Laan, M. J. (2015). A generally efficient targeted minimum loss based estimator. Working Paper 343, U.C. Berkeley Division of Biostatistics Working Paper Series.

van der Laan, M. J., E. C. Polley, and A. E. Hubbard (2007). Super learner. _Statistical Applications in Genetics and Molecular Biology 6_ . Retrieved 24 Feb. 2017, from doi:10.2202/1544-6115.1309.

van der Laan, M. J. and S. Rose (2011). _Targeted Learning: Causal Inference for Observational and Experimental Data_ . Springer.

van der Vaart, A. W. (1991). On differentiable functionals. _Annals of Statistics 19_ , 178–204.

van der Vaart, A. W. (1998). _Asymptotic Statistics_ . Cambridge University Press.

Wager, S. and G. Walther (2016). Adaptive concentration of regression trees, with application to random forests. _arXiv:1503.06388_ . arXiv, 2015.

Wooldridge, J. (1991). Specification testing and quasi-maximum-likelihood estimation. _Journal of Econometrics 48_ , 29–55.

_DML_

51

Zhang, C. and S. Zhang (2014). Confidence intervals for low-dimensional parameters with high-dimensional data. _Journal of the Royal Statistical Society, Series B 76_ , 217–242. arXiv, 2012.

Zheng, W., Z. Luo, and M. J. van der Laan (2016). Marginal structural models with counterfactual effect modifiers. Working Paper 348, U.C. Berkeley Division of Biostatistics Working Paper Series.

Zheng, W. and M. J. van der Laan (2011). Cross-validated targeted minimum-loss-based estimation. In _Targeted Learning_ , pp. 459–474. Springer.

## APPENDIX: PROOFS OF RESULTS

In this appendix, we use _C_ to denote a strictly positive constant that is independent of _n_ and _P ∈PN_ . The value of _C_ may change at each appearance. Also, the notation _aN_ ≲ _bN_ means that _aN_ ⩽ _CbN_ for all _n_ and some _C_ . The notation _aN_ ≳ _bN_ means that _bN_ ≲ _aN_ . Moreover, the notation _aN_ = _o_ (1) means that there exists a sequence ( _bN_ ) _n_ ⩾1 of positive numbers such that (a) _|aN |_ ⩽ _bN_ for all _n_ , (b) _bN_ is independent of _P ∈PN_ for all _n_ , and (c) _bN →_ 0 as _n →∞_ . Finally, the notation _aN_ = _OP_ ( _bN_ ) means that for all _ϵ >_ 0, there exists _C_ such that P _P_ ( _aN > CbN_ ) ⩽ 1 _− ϵ_ for all _n_ . Using this notation allows us to avoid repeating “uniformly over _P ∈PN_ ” many times in the proofs.

Define the empirical process G _n_ ( _ψ_ ( _W_ )) as a linear operator acting on measurable functions _ψ_ : _W →_ R such that _∥ψ∥P,_ 2 _< ∞_ via,

**==> picture [270 x 27] intentionally omitted <==**

Analogously, we defined the empirical expectation as:

**==> picture [186 x 26] intentionally omitted <==**

**==> picture [91 x 10] intentionally omitted <==**

The following lemma is useful particularly in the sample-splitting contexts.

Lemma 6.1. (Conditional Convergence Implies Unconditional) _Let {Xm} and {Ym} be sequences of random vectors. (a) If for ϵm →_ 0 _,_ P( _∥Xm∥ > ϵm | Ym_ ) _→_ P 0 _, then_ P( _∥Xm∥ > ϵm_ ) _→_ 0 _. In particular, this occurs if_ E[ _∥Xm∥[q] /ϵ[q] m[|][Y][m]_[]] _[→]_[P][0] _[for] some q_ ⩾ 1 _, by Markov’s inequality. (b) Let {Am} be a sequence of positive constants. If ∥Xm∥_ = _OP_ ( _Am_ ) _conditional on Ym, namely, that for any ℓm →∞,_ P( _∥Xm∥ > ℓmAm | Ym_ ) _→_ P 0 _, then ∥Xm∥_ = _OP_ ( _Am_ ) _unconditionally, namely, that for any ℓm →∞,_ P( _∥Xm∥ > ℓmAm_ ) _→_ 0 _._

**Proof** . Part (a). For any _ϵ >_ 0 P( _∥Xm∥ > ϵm_ ) ⩽ E[P( _∥Xm∥ > ϵm | Ym_ )] _→_ 0, since the sequence _{_ P( _∥Xm∥ > ϵm | Ym_ ) _}_ is uniformly integrable. To show the second part note that P( _∥Xm∥ > ϵm | Ym_ ) ⩽ E[ _∥Xm∥[q] /ϵ[q] m[|][Y][m]_[]] _[ ∨]_[1] _[→][P]_[0][by][Markov’s][inequality.][Part] (b). This follows from Part (a). ■

Let ( _Wi_ ) _[n] i_ =1[be a sequence of independent copies of a random element] _[ W]_[taking values] in a measurable space ( _W, AW_ ) according to a probability law _P_ . Let _F_ be a set of

52

## _CCDDHNR_

suitably measurable functions _f_ : _W →_ R, equipped with a measurable envelope _F_ : _W →_ R.

Lemma 6.2. (Maximal Inequality, Chernozhukov et al. (2014)) _Work with the setup above. Suppose that F_ ⩾ sup _f ∈F |f | is a measurable envelope for F with ∥F ∥P,q < ∞ for some q_ ⩾ 2 _. Let M_ = max _i_ ⩽ _n F_ ( _Wi_ ) _and σ_[2] _>_ 0 _be any positive constant such that_ sup _f ∈F ∥f ∥_[2] _P,_ 2[⩽] _[σ]_[2][⩽] _[∥][F][∥]_[2] _P,_ 2 _[.][Suppose][that][there][exist][constants][a]_[⩾] _[e][and][v]_[⩾][1] _such that_ log sup _N_ ( _ϵ∥F ∥Q,_ 2 _, F, ∥· ∥Q,_ 2) ⩽ _v_ log( _a/ϵ_ ) _,_ 0 _< ϵ_ ⩽ 1 _. Q_

_Then_

**==> picture [310 x 32] intentionally omitted <==**

_where K is an absolute constant. Moreover, for every t_ ⩾ 1 _, with probability >_ 1 _− t[−][q/]_[2] _,_

_∥_ G _n∥F_ ⩽ (1+ _α_ )E _P_ [ _∥_ G _n∥F_ ]+ _K_ ( _q_ ) ( _σ_ + _n[−]_[1] _[/]_[2] _∥M ∥P,q_ ) _√t_ + _α[−]_[1] _n[−]_[1] _[/]_[2] _∥M ∥P,_ 2 _t , ∀α >_ 0 _,_ � � _where K_ ( _q_ ) _>_ 0 _is a constant depending only on q. In particular, setting a_ ⩾ _n and t_ = log _n, with probability >_ 1 _− c_ (log _n_ ) _[−]_[1] _,_

**==> picture [357 x 32] intentionally omitted <==**

_where ∥M ∥P,q_ ⩽ _n_[1] _[/q] ∥F ∥P,q and K_ ( _q, c_ ) _>_ 0 _is a constant depending only on q and c._

## _A.6. Proof of Lemma 2.1_

**Proof.** Since _J_ exists and _Jββ_ is invertible, (2.8) has the unique solution _µ_ 0 given in (2.10), and so we have by (2.6) that E[ _ψ_ ( _W_ ; _θ_ 0 _, η_ 0)] = 0 for _η_ 0 given in (2.9). Moreover,

**==> picture [306 x 19] intentionally omitted <==**

where _Idθ×dθ_ is the _dθ × dθ_ identity matrix and _⊗_ is the Kronecker product. Hence, the asserted claim holds by the remark after Definition 2.1. ■

## _A.7. Proof of Lemma 2.2_

The proof follows similarly to that of Lemma 2.1, except that now we have to verify (2.4) intead of (2.3). To do so, take any _β ∈B_ such that _∥β − β_ 0 _∥[∗] q_[⩽] _[λ][N][/r][N]_[and][any] _[d][θ][×][ d][β]_ matrix _µ_ . Denote _η_ = ( _β[′] ,_ vec( _µ_ ) _[′]_ ) _[′]_ . Then

**==> picture [371 x 51] intentionally omitted <==**

This completes the proof of the lemma.

_DML_

53

_A.8. Proof of Lemma 2.3_

The proof is similar to that of Lemma 2.1, except that now we have

**==> picture [256 x 12] intentionally omitted <==**

where _Idθ×dθ_ is the _dθ × dθ_ identity matrix and _⊗_ is the Kronecker product.

■

## _A.9. Proof of Lemma 2.4_

The proof follows similarly to that of Lemma 2.2, except that now for any _β ∈B_ such that _∥β − β_ 0 _∥_ 1 ⩽ _λN /rN_ , any _dθ × k_ matrix _µ_ , and _η_ = ( _β[′] ,_ vec( _µ_ ) _[′]_ ) _[′]_ , we have

**==> picture [286 x 50] intentionally omitted <==**

This completes the proof of the lemma.

## _A.10. Proof of Lemma 2.5_

Take any _η ∈ T_ , and consider the function

**==> picture [284 x 11] intentionally omitted <==**

Then

**==> picture [172 x 11] intentionally omitted <==**

and so

**==> picture [227 x 11] intentionally omitted <==**

**==> picture [233 x 26] intentionally omitted <==**

Hence,

**==> picture [166 x 20] intentionally omitted <==**

since

**==> picture [272 x 20] intentionally omitted <==**

as _η_ 0( _θ_ ) = _βθ_ solves the optimization problem

**==> picture [154 x 17] intentionally omitted <==**

Here the regularity conditions are needed to make sure that we can interchange E _P_ and _∂θ_ and also _∂θ_ and _∂r_ in (A.2). This completes the proof of the lemma.

54

_CCDDHNR_

_A.11. Proof of Lemma 2.6_

First, we demonstrate that _µ_ 0 _∈L_[1] ( _R_ ; R _[d][θ][×][d][m]_ ). Indeed,

**==> picture [348 x 90] intentionally omitted <==**

which is finite by assumptions of the lemma since

**==> picture [292 x 21] intentionally omitted <==**

Next, we demonstrate that

E _P_ [ _∥ψ_ ( _W, θ_ 0 _, η_ ) _∥_ ] _< ∞_ for all _η ∈ T._

Indeed, for all _η ∈ T_ , there exist _µ ∈L_[1] ( _R_ ; R _[d][θ][×][d][m]_ ) and _h ∈H_ such that _η_ = ( _µ, h_ ), and so

**==> picture [333 x 56] intentionally omitted <==**

which is finite by assumptions of the lemma. Further, (2.1) holds because

**==> picture [321 x 41] intentionally omitted <==**

where the last equality follows from (2.22).

Finally, we demonstrate that (2.3) holds. To do so, take any _η_ = ( _µ, h_ ) _∈TN_ = _T_ . Then

**==> picture [330 x 34] intentionally omitted <==**

and so

where

**==> picture [238 x 73] intentionally omitted <==**



---

## Raw page.get_text() output

### Page 50

50
CCDDHNR
Poterba, J. M., S. F. Venti, and D. A. Wise (1994b). Do 401(k) contributions crowd out
other personal saving? Journal of Public Economics 58, 1–32.
Robins, J., L. Li, R. Mukherjee, E. Tchetgen, and A. van der Vaart (2017). Minimax
estimation of a functional on a structured high dimensional model. Annals of Statistics,
forthcoming.
Robins, J., L. Li, E. Tchetgen, and A. van der Vaart (2008). Higher order influence
functions and minimax estimation of nonlinear functionals. In D. Nolan and T. Speed
(Eds.), Probability and Statistics: Essays in Honor of David A. Freedman, pp. 335–421.
Institute of Mathematical Statistics.
Robins, J. and A. Rotnitzky (1995). Semiparametric efficiency in multivariate regression
models with missing data. Journal of the American Statistical Association 90, 122–129.
Robins, J., P. Zhang, R. Ayyagari, R. Logan, E. Tchetgen, L. Li, A. Lumley, and
A. van der Vaart (2013). New statistical approaches to semiparametric regression with
application to air pollution research. Research Report 175, Health Effects Institute.
Robinson, P. M. (1988). Root-N-consistent semiparametric regression. Econometrica 56,
931–954.
Rosenbaum, P. R. and D. B. Rubin (1983). The central role of the propensity score in
observational studies for causal effects. Biometrika 70, 41–55.
Scharfstein, D. O., A. Rotnitzky, and J. M. Robins (1999). Rejoinder to “adjusting for
non-ignorable drop-out using semiparametric non-response models”. Journal of the
American Statistical Association 94, 1135–1146.
Schick, A. (1986).
On asymptotically efficient estimation in semiparametric models.
Annals of Statistics 14, 1139–1151.
Severini, T. A. and W. H. Wong (1992). Profile likelihood and conditionally parametric
models. The Annals of Statistics 20, 1768–1802.
Toth, B. and M. J. van der Laan (2016). TMLE for marginal structural models based on
an instrument. Working Paper 350, U.C. Berkeley Division of Biostatistics Working
Paper Series.
van de Geer, S., P. B¨uhlmann, Y. Ritov, and R. Dezeure (2014). On asymptotically opti-
mal confidence regions and tests for high-dimensional models. Annals of Statistics 42,
1166–1202. arXiv, 2013.
van der Laan, M. and D. Rubin (2006). Targeted maximum likelihood learning. Working
Paper 213, UC Berkeley Division of Biostatistics Working Paper Series.
van der Laan, M. J. (2015). A generally efficient targeted minimum loss based estimator.
Working Paper 343, U.C. Berkeley Division of Biostatistics Working Paper Series.
van der Laan, M. J., E. C. Polley, and A. E. Hubbard (2007). Super learner. Statisti-
cal Applications in Genetics and Molecular Biology 6. Retrieved 24 Feb. 2017, from
doi:10.2202/1544-6115.1309.
van der Laan, M. J. and S. Rose (2011). Targeted Learning: Causal Inference for Obser-
vational and Experimental Data. Springer.
van der Vaart, A. W. (1991). On differentiable functionals. Annals of Statistics 19,
178–204.
van der Vaart, A. W. (1998). Asymptotic Statistics. Cambridge University Press.
Wager, S. and G. Walther (2016). Adaptive concentration of regression trees, with ap-
plication to random forests. arXiv:1503.06388. arXiv, 2015.
Wooldridge, J. (1991). Specification testing and quasi-maximum-likelihood estimation.
Journal of Econometrics 48, 29–55.


### Page 51

DML
51
Zhang, C. and S. Zhang (2014). Confidence intervals for low-dimensional parameters with
high-dimensional data. Journal of the Royal Statistical Society, Series B 76, 217–242.
arXiv, 2012.
Zheng, W., Z. Luo, and M. J. van der Laan (2016). Marginal structural models with coun-
terfactual effect modifiers. Working Paper 348, U.C. Berkeley Division of Biostatistics
Working Paper Series.
Zheng, W. and M. J. van der Laan (2011). Cross-validated targeted minimum-loss-based
estimation. In Targeted Learning, pp. 459–474. Springer.
APPENDIX: PROOFS OF RESULTS
In this appendix, we use C to denote a strictly positive constant that is independent
of n and P ∈PN. The value of C may change at each appearance. Also, the notation
aN ≲bN means that aN ⩽CbN for all n and some C. The notation aN ≳bN means that
bN ≲aN. Moreover, the notation aN = o(1) means that there exists a sequence (bN)n⩾1
of positive numbers such that (a) |aN| ⩽bN for all n, (b) bN is independent of P ∈PN
for all n, and (c) bN →0 as n →∞. Finally, the notation aN = OP (bN) means that for
all ϵ > 0, there exists C such that PP (aN > CbN) ⩽1 −ϵ for all n. Using this notation
allows us to avoid repeating “uniformly over P ∈PN” many times in the proofs.
Define the empirical process Gn(ψ(W)) as a linear operator acting on measurable
functions ψ : W →R such that ∥ψ∥P,2 < ∞via,
Gn(ψ(W)) := Gn,I(ψ(W)) :=
1
√n
X
i∈I
ψ(Wi) −
Z
ψ(w)dP(w).
Analogously, we defined the empirical expectation as:
En(ψ(W)) := En,I(ψ(W)) := 1
n
X
i∈I
ψ(Wi).
A.5. Useful Lemmas
The following lemma is useful particularly in the sample-splitting contexts.
Lemma 6.1. (Conditional Convergence Implies Unconditional) Let {Xm} and
{Ym} be sequences of random vectors. (a) If for ϵm →0, P(∥Xm∥> ϵm | Ym) →P 0,
then P(∥Xm∥> ϵm) →0. In particular, this occurs if E[∥Xm∥q/ϵq
m | Ym] →P 0 for
some q ⩾1, by Markov’s inequality. (b) Let {Am} be a sequence of positive constants. If
∥Xm∥= OP (Am) conditional on Ym, namely, that for any ℓm →∞, P(∥Xm∥> ℓmAm |
Ym) →P 0, then ∥Xm∥= OP (Am) unconditionally, namely, that for any ℓm →∞,
P(∥Xm∥> ℓmAm) →0.
Proof. Part (a). For any ϵ > 0 P(∥Xm∥> ϵm) ⩽E[P(∥Xm∥> ϵm | Ym)] →0, since the
sequence {P(∥Xm∥> ϵm | Ym)} is uniformly integrable. To show the second part note
that P(∥Xm∥> ϵm | Ym) ⩽E[∥Xm∥q/ϵq
m | Ym] ∨1 →P 0 by Markov’s inequality. Part
(b). This follows from Part (a).
■
Let (Wi)n
i=1 be a sequence of independent copies of a random element W taking values
in a measurable space (W, AW) according to a probability law P. Let F be a set of


### Page 52

52
CCDDHNR
suitably measurable functions f : W →R, equipped with a measurable envelope F : W →
R.
Lemma 6.2. (Maximal Inequality, Chernozhukov et al. (2014)) Work with the
setup above. Suppose that F ⩾supf∈F |f| is a measurable envelope for F with ∥F∥P,q <
∞for some q ⩾2. Let M = maxi⩽n F(Wi) and σ2 > 0 be any positive constant such
that supf∈F ∥f∥2
P,2 ⩽σ2 ⩽∥F∥2
P,2. Suppose that there exist constants a ⩾e and v ⩾1
such that
log sup
Q
N(ϵ∥F∥Q,2, F, ∥· ∥Q,2) ⩽v log(a/ϵ), 0 < ϵ ⩽1.
Then
EP [∥Gn∥F] ⩽K
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
,
where K is an absolute constant. Moreover, for every t ⩾1, with probability > 1 −t−q/2,
∥Gn∥F ⩽(1+α)EP [∥Gn∥F]+K(q)
h
(σ+n−1/2∥M∥P,q)
√
t+α−1n−1/2∥M∥P,2t
i
, ∀α > 0,
where K(q) > 0 is a constant depending only on q. In particular, setting a ⩾n and
t = log n, with probability > 1 −c(log n)−1,
∥Gn∥F ⩽K(q, c)

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
(A.1)
where ∥M∥P,q ⩽n1/q∥F∥P,q and K(q, c) > 0 is a constant depending only on q and c.
A.6. Proof of Lemma 2.1
Proof. Since J exists and Jββ is invertible, (2.8) has the unique solution µ0 given in
(2.10), and so we have by (2.6) that E[ψ(W; θ0, η0)] = 0 for η0 given in (2.9). Moreover,
∂η′EP ψ(W; θ0, η0) =

[Jθβ −µ0Jββ], E[∂β′ℓ(W; θ0, β0)] ⊗Idθ×dθ

= 0,
where Idθ×dθ is the dθ × dθ identity matrix and ⊗is the Kronecker product. Hence, the
asserted claim holds by the remark after Definition 2.1.
■
A.7. Proof of Lemma 2.2
The proof follows similarly to that of Lemma 2.1, except that now we have to verify (2.4)
intead of (2.3). To do so, take any β ∈B such that ∥β −β0∥∗
q ⩽λN/rN and any dθ × dβ
matrix µ. Denote η = (β′, vec(µ)′)′. Then



∂ηEP ψ(W, θ0, η0)[η −η0]



 =



(Jθβ −µ0Jββ)(β −β0)



⩽∥Jθβ −µ0Jββ∥q × ∥β −β0∥∗
q ⩽rn × (λN/rN) = λN.
This completes the proof of the lemma.
■


### Page 53

DML
53
A.8. Proof of Lemma 2.3
The proof is similar to that of Lemma 2.1, except that now we have
∂η′EP ψ(W, θ0, η0) = [µ0Gβ, EP m(W, θ0, β0)′ ⊗Idθ×dθ] = 0,
where Idθ×dθ is the dθ × dθ identity matrix and ⊗is the Kronecker product.
■
A.9. Proof of Lemma 2.4
The proof follows similarly to that of Lemma 2.2, except that now for any β ∈B such
that ∥β −β0∥1 ⩽λN/rN, any dθ × k matrix µ, and η = (β′, vec(µ)′)′, we have



∂ηEP ψ(W, θ0, η0)[η −η0]



 = ∥µ0Gβ(β −β0)∥
⩽∥A′Ω−1/2L −γ0L′L∥∞× ∥β −β0∥1
⩽rn × (λN/rN) = λN.
This completes the proof of the lemma.
A.10. Proof of Lemma 2.5
Take any η ∈T, and consider the function
Q(W; θ, r) := ℓ(W; θ, η0(θ) + r(η(θ) −η0(θ))),
θ ∈Θ, r ∈[0, 1].
Then
ψ(W; θ, η0 + r(η −η0)) = ∂θQ(W; θ, r),
and so
∂rEP [ψ(W; θ, η0 + r(η −η0))] = ∂rEP [∂θQ(W; θ, r)]
= ∂r∂θEP [Q(W; θ, r)] = ∂θ∂rEP [Q(W; θ, r)]
(A.2)
= ∂θ∂rEP [ℓ(W; θ, η0(θ) + r(η(θ) −η0(θ)))].
Hence,
∂rEP [ψ(W; θ, η0 + r(η −η0))]

r=0 = 0
since
∂rEP [ℓ(W; θ, η0(θ) + r(η(θ) −η0(θ)))]

r=0 = 0,
for all θ ∈Θ,
as η0(θ) = βθ solves the optimization problem
max
β∈B EP [ℓ(W; θ, β)],
for all θ ∈Θ.
Here the regularity conditions are needed to make sure that we can interchange EP and
∂θ and also ∂θ and ∂r in (A.2). This completes the proof of the lemma.


### Page 54

54
CCDDHNR
A.11. Proof of Lemma 2.6
First, we demonstrate that µ0 ∈L1(R; Rdθ×dm). Indeed,
EP [∥µ0(R)∥] ⩽EP
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
,
which is finite by assumptions of the lemma since
EP
h
∥G(Z)∥2 × ∥Γ(R)∥2i
⩽

EP [∥G(Z)∥4] × EP [Γ(R)∥4]
1/2
< ∞.
Next, we demonstrate that
EP [∥ψ(W, θ0, η)∥] < ∞
for all η ∈T.
Indeed, for all η ∈T, there exist µ ∈L1(R; Rdθ×dm) and h ∈H such that η = (µ, h),
and so
EP [∥ψ(W, θ0, η)∥] = EP [∥µ(X)m(W, θ0, h(Z))∥]
⩽EP
h
∥µ(R)∥× ∥m(W, θ0, h(Z))∥
i
= EP
h
∥µ(R)∥× EP [∥m(W, θ0, h(Z)) | R]
i
⩽ChE[∥µ(R)∥],
which is finite by assumptions of the lemma. Further, (2.1) holds because
EP [ψ(W, θ0, η0)] = EP
h
µ0(R)m(W, θ0, h0(Z))
i
= EP
h
µ0(R)EP [m(W, θ0, h0(Z)) | R]
i
= 0,
(A.3)
where the last equality follows from (2.22).
Finally, we demonstrate that (2.3) holds. To do so, take any η = (µ, h) ∈TN = T.
Then
EP [ψ(W, θ0, η0 + r(η −η0)]
= EP
h
(µ0(R) + r(µ(R) −µ0(R)))m(W, θ0, h0(Z) + r(h(Z) −h0(Z)))
i
,
and so
∂ηEP ψ(W, θ0, η0)[η −η0] = I1 + I2,
where
I1 = EP
h
(µ(R) −µ0(R))m(W, θ0, h0(Z)
i
,
I2 = EP
h
µ0(R)∂v′m(W, θ0, v)|v=h0(Z)(h(Z) −h0(Z))
i
.
