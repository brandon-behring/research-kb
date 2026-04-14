
## pymupdf4llm output

**40**

**2 Linear functions**

**==> picture [302 x 290] intentionally omitted <==**

**----- Start of picture text -----**<br>
800<br>600<br>House 5<br>House 4<br>400<br>House 1<br>200<br>House 2<br>House 3<br>0<br>0 200 400 600 800<br>Actual price y (thousand dollars)<br>dollars)<br>(thousand<br>ˆ y<br>price<br>Predicted<br>**----- End of picture text -----**<br>


**Figure 2.4** Scatter plot of actual and predicted sale prices for 774 houses sold in Sacramento during a five-day period.

**2.3 Regression model**

**41**

thousands of dollars per bedroom. It might seem strange that _β_ 2 is negative, since one imagines that adding a bedroom to a house would _increase_ its sale price, not decrease it. To understand why _β_ 2 might be negative, we note that it gives the change in predicted price when we add a bedroom, without adding any additional area to the house. If we remodel a house by adding a bedroom that _also_ adds more than around 127 square feet to the house area, the regression model (2.9) _does_ predict an increase in house sale price. The offset _v_ = 54 _._ 40 is the predicted price for a house with no area and no bedrooms, which we might interpret as the model’s prediction of the value of the lot. But this regression model is crude enough that these interpretations are dubious.

**42**

**2 Linear functions**

## **Exercises**

- **2.1** _Linear or not?_ Determine whether each of the following scalar-valued functions of _n_ - vectors is linear. If it is a linear function, give its inner product representation, _i.e._ , an _n_ -vector _a_ for which _f_ ( _x_ ) = _a[T] x_ for all _x_ . If it is not linear, give specific _x_ , _y_ , _α_ , and _β_ for which superposition fails, _i.e._ ,

**==> picture [120 x 10] intentionally omitted <==**

   - (a) The spread of values of the vector, defined as _f_ ( _x_ ) = max _k xk −_ min _k xk_ .

   - (b) The difference of the last element and the first, _f_ ( _x_ ) = _xn − x_ 1.

   - (c) The median of an _n_ -vector, where we will assume _n_ = 2 _k_ + 1 is odd. The median of the vector _x_ is defined as the ( _k_ + 1)st largest number among the entries of _x_ . For example, the median of ( _−_ 7 _._ 1 _,_ 3 _._ 2 _, −_ 1 _._ 5) is _−_ 1 _._ 5.

   - (d) The average of the entries with odd indices, minus the average of the entries with even indices. You can assume that _n_ = 2 _k_ is even.

   - (e) Vector extrapolation, defined as _xn_ + ( _xn − xn−_ 1), for _n ≥_ 2. (This is a simple prediction of what _xn_ +1 would be, based on a straight line drawn through _xn_ and _xn−_ 1.)

- **2.2** _Processor powers and temperature._ The temperature _T_ of an electronic device containing three processors is an affine function of the power dissipated by the three processors, _P_ = ( _P_ 1 _, P_ 2 _, P_ 3). When all three processors are idling, we have _P_ = (10 _,_ 10 _,_ 10), which results in a temperature _T_ = 30. When the first processor operates at full power and the other two are idling, we have _P_ = (100 _,_ 10 _,_ 10), and the temperature rises to _T_ = 60. When the second processor operates at full power and the other two are idling, we have _P_ = (10 _,_ 100 _,_ 10) and _T_ = 70. When the third processor operates at full power and the other two are idling, we have _P_ = (10 _,_ 10 _,_ 100) and _T_ = 65. Now suppose that all three processors are operated at the same power _P_[same] . How large can _P_[same] be, if we require that _T ≤_ 85? _Hint._ From the given data, find the 3-vector _a_ and number _b_ for which _T_ = _a[T] P_ + _b_ .

- **2.3** _Motion of a mass in response to applied force._ A unit mass moves on a straight line (in one dimension). The position of the mass at time _t_ (in seconds) is denoted by _s_ ( _t_ ), and its derivatives (the velocity and acceleration) by _s[′]_ ( _t_ ) and _s[′′]_ ( _t_ ). The position as a function of time can be determined from Newton’s second law

**==> picture [53 x 11] intentionally omitted <==**

where _F_ ( _t_ ) is the force applied at time _t_ , and the initial conditions _s_ (0), _s[′]_ (0). We assume _F_ ( _t_ ) is piecewise-constant, and is kept constant in intervals of one second. The sequence of forces _F_ ( _t_ ), for 0 _≤ t <_ 10, can then be represented by a 10-vector _f_ , with

**==> picture [110 x 10] intentionally omitted <==**

Derive expressions for the final velocity _s[′]_ (10) and final position _s_ (10). Show that _s_ (10) and _s[′]_ (10) are affine functions of _x_ , and give 10-vectors _a, c_ and constants _b, d_ for which

**==> picture [157 x 11] intentionally omitted <==**

This means that the mapping from the applied force sequence to the final position and velocity is affine.

_Hint._ You can use

**==> picture [232 x 25] intentionally omitted <==**

You will find that the mass velocity _s[′]_ ( _t_ ) is piecewise-linear.

**43**

**Exercises**

- **2.4** _Linear function?_ The function _φ_ : **R**[3] _→_ **R** satisfies

_φ_ (1 _,_ 1 _,_ 0) = _−_ 1 _, φ_ ( _−_ 1 _,_ 1 _,_ 1) = 1 _, φ_ (1 _, −_ 1 _, −_ 1) = 1 _._

Choose one of the following, and justify your choice: _φ_ must be linear; _φ_ could be linear; _φ_ cannot be linear.

- **2.5** _Affine function._ Suppose _ψ_ : **R**[2] _→_ **R** is an affine function, with _ψ_ (1 _,_ 0) = 1, _ψ_ (1 _, −_ 2) = 2.

   - (a) What can you say about _ψ_ (1 _, −_ 1)? Either give the value of _ψ_ (1 _, −_ 1), or state that it cannot be determined.

   - (b) What can you say about _ψ_ (2 _, −_ 2)? Either give the value of _ψ_ (2 _, −_ 2), or state that it cannot be determined.

Justify your answers.

- **2.6** _Questionnaire scoring._ A questionnaire in a magazine has 30 questions, broken into two sets of 15 questions. Someone taking the questionnaire answers each question with ‘Rarely’, ‘Sometimes’, or ‘Often’. The answers are recorded as a 30-vector _a_ , with _ai_ = 1 _,_ 2 _,_ 3 if question _i_ is answered Rarely, Sometimes, or Often, respectively. The total score on a completed questionnaire is found by adding up 1 point for every question answered Sometimes and 2 points for every question answered Often on questions 1–15, and by adding 2 points and 4 points for those responses on questions 16–30. (Nothing is added to the score for Rarely responses.) Express the total score _s_ in the form of an affine function _s_ = _w[T] a_ + _v_ , where _w_ is a 30-vector and _v_ is a scalar (number).

- **2.7** _General formula for affine functions._ Verify that formula (2.4) holds for any affine function _f_ : **R** _[n] →_ **R** . You can use the fact that _f_ ( _x_ ) = _a[T] x_ + _b_ for some _n_ -vector _a_ and scalar _b_ .

- **2.8** _Integral and derivative of polynomial._ Suppose the _n_ -vector _c_ gives the coefficients of a polynomial _p_ ( _x_ ) = _c_ 1 + _c_ 2 _x_ + _· · ·_ + _cnx[n][−]_[1] .

   - (a) Let _α_ and _β_ be numbers with _α < β_ . Find an _n_ -vector _a_ for which

**==> picture [74 x 25] intentionally omitted <==**

always holds. This means that the integral of a polynomial over an interval is a linear function of its

- (b) Let _α_ be a number. Find an _n_ -vector _b_ for which

**==> picture [50 x 11] intentionally omitted <==**

This means that the derivative of the polynomial at a given point is a linear function of its

- **2.9** _Taylor approximation._ Consider the function _f_ : **R**[2] _→_ **R** given by _f_ ( _x_ 1 _, x_ 2) = _x_ 1 _x_ 2. Find the Taylor approximation _f_[ˆ] at the point _z_ = (1 _,_ 1). Compare _f_ ( _x_ ) and _f_[ˆ] ( _x_ ) for the following values of _x_ :

**==> picture [249 x 10] intentionally omitted <==**

Make a brief comment about the accuracy of the Taylor approximation in each case.

- **2.10** _Regression model._ Consider the regression model _y_ ˆ = _x[T] β_ + _v_ , where _y_ ˆ is the predicted response, _x_ is an 8-vector of features, _β_ is an 8-vector of coefficients, and _v_ is the offset term. Determine whether each of the following statements is true or false.

   - (a) If _β_ 3 _>_ 0 and _x_ 3 _>_ 0, then _y_ ˆ _≥_ 0.

   - (b) If _β_ 2 = 0 then the prediction _y_ ˆ does not depend on the second feature _x_ 2.

   - (c) If _β_ 6 = _−_ 0 _._ 8, then increasing _x_ 6 (keeping all other _xi_ s the same) will decrease _y_ ˆ.

**2 Linear functions**

**44**

- **2.11** _Sparse regression weight vector._ Suppose that _x_ is an _n_ -vector that gives _n_ features for some object, and the scalar _y_ is some outcome associated with the object. What does it mean if a regression model _y_ ˆ = _x[T] β_ + _v_ uses a sparse weight vector _β_ ? Give your answer in English, referring to _y_ ˆ as our prediction of the outcome.

- **2.12** _Price change to maximize profit._ A business sells _n_ products, and is considering changing the price of _one_ of the products to increase its total profits. A business analyst develops a regression model that (reasonably accurately) predicts the total profit when the product prices are changed, given by _P_[ˆ] = _β[T] x_ + _P_ , where the _n_ -vector _x_ denotes the fractional change in the product prices, _xi_ = ( _p_[new] _i − pi_ ) _/pi_ . Here _P_ is the profit with the current prices, _P_[ˆ] is the predicted profit with the changed prices, _pi_ is the current (positive) price of product _i_ , and _p_[new] _i_ is the new price of product _i_ .

   - (a) What does it mean if _β_ 3 _<_ 0? (And yes, this can occur.)

   - (b) Suppose that you are given permission to change the price of _one_ product, by up to 1%, to increase total profit. Which product would you choose, and would you increase or decrease the price? By how much?

   - (c) Repeat part (b) assuming you are allowed to change the price of two products, each by up to 1%.



---

## Raw page.get_text() output

### Page 50

40
2
Linear functions
0
200
400
600
800
0
200
400
600
800
House 1
House 2
House 3
House 4
House 5
Actual price y (thousand dollars)
Predicted price ˆy (thousand dollars)
Figure 2.4 Scatter plot of actual and predicted sale prices for 774 houses
sold in Sacramento during a ﬁve-day period.


### Page 51

2.3
Regression model
41
thousands of dollars per bedroom. It might seem strange that β2 is negative, since
one imagines that adding a bedroom to a house would increase its sale price, not
decrease it. To understand why β2 might be negative, we note that it gives the
change in predicted price when we add a bedroom, without adding any additional
area to the house. If we remodel a house by adding a bedroom that also adds more
than around 127 square feet to the house area, the regression model (2.9) does
predict an increase in house sale price. The oﬀset v = 54.40 is the predicted price
for a house with no area and no bedrooms, which we might interpret as the model’s
prediction of the value of the lot. But this regression model is crude enough that
these interpretations are dubious.


### Page 52

42
2
Linear functions
Exercises
2.1 Linear or not?
Determine whether each of the following scalar-valued functions of n-
vectors is linear. If it is a linear function, give its inner product representation, i.e., an
n-vector a for which f(x) = aT x for all x. If it is not linear, give speciﬁc x, y, α, and β
for which superposition fails, i.e.,
f(αx + βy) ̸= αf(x) + βf(y).
(a) The spread of values of the vector, deﬁned as f(x) = maxk xk −mink xk.
(b) The diﬀerence of the last element and the ﬁrst, f(x) = xn −x1.
(c) The median of an n-vector, where we will assume n = 2k + 1 is odd. The median of
the vector x is deﬁned as the (k + 1)st largest number among the entries of x. For
example, the median of (−7.1, 3.2, −1.5) is −1.5.
(d) The average of the entries with odd indices, minus the average of the entries with
even indices. You can assume that n = 2k is even.
(e) Vector extrapolation, deﬁned as xn + (xn −xn−1), for n ≥2. (This is a simple
prediction of what xn+1 would be, based on a straight line drawn through xn and
xn−1.)
2.2 Processor powers and temperature. The temperature T of an electronic device containing
three processors is an aﬃne function of the power dissipated by the three processors,
P = (P1, P2, P3). When all three processors are idling, we have P = (10, 10, 10), which
results in a temperature T = 30. When the ﬁrst processor operates at full power and
the other two are idling, we have P = (100, 10, 10), and the temperature rises to T = 60.
When the second processor operates at full power and the other two are idling, we have
P = (10, 100, 10) and T = 70. When the third processor operates at full power and the
other two are idling, we have P = (10, 10, 100) and T = 65. Now suppose that all three
processors are operated at the same power P same. How large can P same be, if we require
that T ≤85? Hint. From the given data, ﬁnd the 3-vector a and number b for which
T = aT P + b.
2.3 Motion of a mass in response to applied force. A unit mass moves on a straight line (in
one dimension). The position of the mass at time t (in seconds) is denoted by s(t), and its
derivatives (the velocity and acceleration) by s′(t) and s′′(t). The position as a function
of time can be determined from Newton’s second law
s′′(t) = F(t),
where F(t) is the force applied at time t, and the initial conditions s(0), s′(0). We assume
F(t) is piecewise-constant, and is kept constant in intervals of one second. The sequence
of forces F(t), for 0 ≤t < 10, can then be represented by a 10-vector f, with
F(t) = fk,
k −1 ≤t < k.
Derive expressions for the ﬁnal velocity s′(10) and ﬁnal position s(10). Show that s(10)
and s′(10) are aﬃne functions of x, and give 10-vectors a, c and constants b, d for which
s′(10) = aT f + b,
s(10) = cT f + d.
This means that the mapping from the applied force sequence to the ﬁnal position and
velocity is aﬃne.
Hint. You can use
s′(t) = s′(0) +
Z t
0
F(τ) dτ,
s(t) = s(0) +
Z t
0
s′(τ) dτ.
You will ﬁnd that the mass velocity s′(t) is piecewise-linear.


### Page 53

Exercises
43
2.4 Linear function? The function φ : R3 →R satisﬁes
φ(1, 1, 0) = −1,
φ(−1, 1, 1) = 1,
φ(1, −1, −1) = 1.
Choose one of the following, and justify your choice: φ must be linear; φ could be linear;
φ cannot be linear.
2.5 Aﬃne function. Suppose ψ : R2 →R is an aﬃne function, with ψ(1, 0) = 1, ψ(1, −2) = 2.
(a) What can you say about ψ(1, −1)? Either give the value of ψ(1, −1), or state that
it cannot be determined.
(b) What can you say about ψ(2, −2)? Either give the value of ψ(2, −2), or state that
it cannot be determined.
Justify your answers.
2.6 Questionnaire scoring.
A questionnaire in a magazine has 30 questions, broken into
two sets of 15 questions. Someone taking the questionnaire answers each question with
‘Rarely’, ‘Sometimes’, or ‘Often’. The answers are recorded as a 30-vector a, with ai =
1, 2, 3 if question i is answered Rarely, Sometimes, or Often, respectively. The total score
on a completed questionnaire is found by adding up 1 point for every question answered
Sometimes and 2 points for every question answered Often on questions 1–15, and by
adding 2 points and 4 points for those responses on questions 16–30. (Nothing is added to
the score for Rarely responses.) Express the total score s in the form of an aﬃne function
s = wT a + v, where w is a 30-vector and v is a scalar (number).
2.7 General formula for aﬃne functions. Verify that formula (2.4) holds for any aﬃne function
f : Rn →R. You can use the fact that f(x) = aT x + b for some n-vector a and scalar b.
2.8 Integral and derivative of polynomial. Suppose the n-vector c gives the coeﬃcients of a
polynomial p(x) = c1 + c2x + · · · + cnxn−1.
(a) Let α and β be numbers with α < β. Find an n-vector a for which
aT c =
Z β
α
p(x) dx
always holds. This means that the integral of a polynomial over an interval is a
linear function of its coeﬃcients.
(b) Let α be a number. Find an n-vector b for which
bT c = p′(α).
This means that the derivative of the polynomial at a given point is a linear function
of its coeﬃcients.
2.9 Taylor approximation. Consider the function f : R2 →R given by f(x1, x2) = x1x2.
Find the Taylor approximation ˆf at the point z = (1, 1). Compare f(x) and ˆf(x) for the
following values of x:
x = (1, 1),
x = (1.05, 0.95),
x = (0.85, 1.25),
x = (−1, 2).
Make a brief comment about the accuracy of the Taylor approximation in each case.
2.10 Regression model. Consider the regression model ˆy = xT β + v, where ˆy is the predicted
response, x is an 8-vector of features, β is an 8-vector of coeﬃcients, and v is the oﬀset
term. Determine whether each of the following statements is true or false.
(a) If β3 > 0 and x3 > 0, then ˆy ≥0.
(b) If β2 = 0 then the prediction ˆy does not depend on the second feature x2.
(c) If β6 = −0.8, then increasing x6 (keeping all other xis the same) will decrease ˆy.


### Page 54

44
2
Linear functions
2.11 Sparse regression weight vector. Suppose that x is an n-vector that gives n features for
some object, and the scalar y is some outcome associated with the object. What does it
mean if a regression model ˆy = xT β + v uses a sparse weight vector β? Give your answer
in English, referring to ˆy as our prediction of the outcome.
2.12 Price change to maximize proﬁt. A business sells n products, and is considering changing
the price of one of the products to increase its total proﬁts. A business analyst develops a
regression model that (reasonably accurately) predicts the total proﬁt when the product
prices are changed, given by ˆP = βT x + P, where the n-vector x denotes the fractional
change in the product prices, xi = (pnew
i
−pi)/pi. Here P is the proﬁt with the current
prices, ˆP is the predicted proﬁt with the changed prices, pi is the current (positive) price
of product i, and pnew
i
is the new price of product i.
(a) What does it mean if β3 < 0? (And yes, this can occur.)
(b) Suppose that you are given permission to change the price of one product, by up
to 1%, to increase total proﬁt. Which product would you choose, and would you
increase or decrease the price? By how much?
(c) Repeat part (b) assuming you are allowed to change the price of two products, each
by up to 1%.
