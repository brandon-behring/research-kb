
## pymupdf4llm output

**20**

CHAPTER 1 _**Why causal AI**_

## STEP 3: DO THE STATISTICAL INFERENCE

Step 3 does the statistical estimation, and there are several ways we could estimate the quantities on the right side of that equation. For example, we could use a convolutional neural network to model _E_ ( _I_ | _W_ = _w_ , _A_ = _a_ , _D_ = _d_ , _T_ = _t_ ), and build a probability model of the joint distribution _P_ ( _A_ , _D_ , _T_ ). The choice of statistical modeling approach involves the usual statistical trade-offs, such as ease-of-use, bias and variance, scalability to large data, and parallelizability.

Other books go into great detail on preferred statistical methods for step 3. I take the strongly opinionated view that we should rely on the “commodification of inference” trend in statistical modeling and machine learning frameworks to handle step 3, and instead focus on honing our skills on steps 1 and 2: figuring out the right questions to ask, and representing the possible causes mathematically.

As you’ve seen in this section, our journey into causal AI is scaffolded by a threestep process, and the essence of causal thinking emerges prominently in the first two steps. Step 1 invites us to frame the right causal questions, while step 2 illuminates the mathematics behind these questions. Step 3 leverages patterns we’re well-accustomed to in traditional statistical prediction and inference.

Using this structured approach, we’ll transition in the coming chapters from purely predictive machine learning models—like the deep latent variable models you might be familiar with from MNIST—to causal machine learning models that offer deeper insights into and answers to our causal questions. First, we will review the underlying mathematics and machine learning foundations. Then, in part 2 of the book, we’ll delve into crafting the right questions and articulating them mathematically for steps 1 and 2. For step 3, we’ll harness the power of contemporary tools like PyTorch and other advanced libraries to bridge the causal concepts with cutting-edge statistical learning algorithms.

## _Summary_

- Causal AI seeks to augment statistical learning and probabilistic reasoning with causal logic.

- Causal inference helps data scientists extract more causal insights from observational data (the vast majority of data in the world) and experimental data.

- When data scientists can’t run experiments, causal models can simulate experiments from observational data.

- They can use these simulations to make causal inferences, such as estimating causal effects, and even to prioritize interesting experiments to run in real life.

- Causal inference also helps data scientists improve decision-making in their organizations through algorithmic counterfactual reasoning and attribution.

- Causal inference also makes machine learning more _robust_ , _decomposable_ , and _explainable_ .

- Causal analysis is useful for formally analyzing _fairness_ in predictive algorithms and for building fairer algorithms by parsing ordinary statistical bias into its causal sources.

Licensed to Brandon Behring <bb1023@nyu.edu>

**21**

_**Summary**_

- The _commodification of inference_ is a trend in machine learning that refers to how universal modeling frameworks like PyTorch continuously automate the nuts and bolts of statistical learning and probabilistic inference. The trend reduces the need for the modeler to be an expert at the formal and statistical details of causal inference and allows them to focus on turning domain expertise into better causal models of their problem domain.

- Types of causal inference tasks include _causal discovery_ , _intervention prediction_ , _causal effect estimation_ , c _ounterfactual reasoning_ , _explanation_ , and _attribution_ .

- The way we build and work with probabilistic machine learning models can be extended to causal generative models implemented in probabilistic machine learning tools such as PyTorch.

Licensed to Brandon Behring <bb1023@nyu.edu>

## _A primer on probabilistic enerative modelin g g_

## _This chapter covers_

- A primer on probability models

- Computational probability with the pgmpy and Pyro libraries

- Statistics for causality: data, populations, and models

- Distinguishing between probability models and subjective Bayesianism

Chapter 1 made the case for learning how to code causal AI. This chapter will introduce some fundamentals we need to tackle causal modeling with probabilistic machine learning, which roughly refers to machine learning techniques that use probability to model uncertainty and simulate data. There is a flexible suite of cutting-edge tools for building probabilistic machine learning models. This chapter will introduce the concepts from probability, statistics, modeling, inference, and

**22**

Licensed to Brandon Behring <bb1023@nyu.edu>

_**2.1 Primer on probability**_

**23**

even philosophy that we will need in order to implement key ideas from causal inference with the probabilistic machine learning approach.

This chapter will not provide a mathematically exhaustive introduction to these ideas. I’ll focus on what is needed for the rest of this book and omit the rest. Any data scientist seeking causal inference expertise should not neglect the practical nuances of probability, statistics, machine learning, and computer science. See the chapter notes at www.altdeep.ai/causalAIbook for recommended resources where you can get deeper introductions or review materials.

In this chapter, I’ll introduce two Python programming libraries for probabilistic machine learning:

- _pgmpy_ is a library for building probabilistic graphical models. As a traditional graphical modeling tool, it is far less flexible and cutting-edge than Pyro but also easier to use and debug. What it does, it does well.

- _Pyro_ is a general probabilistic machine learning library. It is quite flexible, and it leverages PyTorch’s cutting-edge gradient-based learning techniques.

Pyro and pgmpy are the general modeling libraries we’ll use in this book. Other libraries we’ll use are designed specifically for causal inference.

## _2.1 Primer on probability_

Let’s review the probability theory you’ll need to work with this book. We’ll start with a few basic mathematical axioms and their logical extensions without yet adding any real-world interpretation. Let’s begin with the concrete idea of a simple three-sided die (these exist).

- _2.1.1 Random variables and probability_

A _random variable_ is a variable whose possible values are the numerical outcomes of a random phenomenon. These values can be discrete or continuous. In this section, we’ll focus on the discrete case. For example, the values of a discrete random variable representing a three-sided die roll could be {1, 2, 3}. Alternatively, in a 0-indexed programming language like Python, it might be better to use {0, 1, 2}. Similarly, a discrete random variable representing a coin flip could have outcomes {0, 1} or {True, False}. Figure 2.1 illustrates three-sided dice.

The typical approach to notation is to write random variables with capitals like _X_ , _Y_ , and _Z_ . For example, suppose _X_ represents a die roll with outcomes {1, 2, 3}, and the outcome represents the number on the side of the die. _X_ =1 and _X_ =2 represent the events of rolling a 1 and 2 respectively. If we want to abstract

Figure 2.1 Three-sided dice each represent a random variable with three discrete outcomes.

Licensed to Brandon Behring <bb1023@nyu.edu>

CHAPTER 2 _**A primer on probabilistic generative modeling**_

**24**

away the specific outcome with a variable, we typically use lowercase. For example, I would use “ _X_ = _x_ ” (e.g., _X_ =1) to represent the event “I rolled an ‘ _x_ ’!” where _x_ can be any value in {1, 2, 3}. See figure 2.2.

**==> picture [186 x 65] intentionally omitted <==**

**----- Start of picture text -----**<br>
X =2<br>& < ,<br>**----- End of picture text -----**<br>


Figure 2.2 _X_ represents the outcome of a three-sided die roll. If the die roles a 2, the observed outcome is _X_ =2.

Each outcome of a random variable has a _probability value_ . The probability value is often called a _probability mass_ for discrete variables and a _probability density_ for continuous variables. For discrete variables, probability values are between zero and one, and summing up the probability values for each possible outcome yields 1. For continuous variables, probability densities are greater than zero, and integrating the probability densities over each possible outcome yields 1.

Given a random variable with outcomes {0, 1} representing a coin flip, what is the probability value assigned to 0? What about 1? At this point, we just know the two values are between zero and one, and that they sum to one. To go beyond that, we have to talk about how to _interpret_ probability. First, though, let’s hash out a few more concepts.

## _2.1.2 Probability distributions and distribution functions_

A _probability distribution function_ is a function that maps the random variable outcomes to a probability value. For example, if the outcome of a coin flip is 1 (heads) and the probability value is 0.51, the distribution function maps 1 to 0.51. I stick to the standard notation _P_ ( _X_ = _x_ ), as in _P_ ( _X_ =1) = 0.51. For longer expressions, when the random variable is obvious, I drop the capital letter and keep the outcome, so _P_ ( _X_ = _x_ ) becomes _P_ ( _x_ ), and _P_ ( _X_ =1) becomes _P_ (1).

If the random variable has a finite set of discrete outcomes, we can represent the probability distribution with a table. For example, a random variable representing outcomes {1, 2, 3} might look like figure 2.3.

**==> picture [20 x 25] intentionally omitted <==**

**----- Start of picture text -----**<br>
|||||
|---|---|---|---|
|X|
|P|(|X|)|

**----- End of picture text -----**<br>


Figure 2.3 A simple tabular representation of a discrete distribution

In this book, I adopt the common notation _P_ ( _X_ ) to represent the probability distribution over all possible outcomes of _X_ , while _P_ ( _X_ = _x_ ) represents the probability value of a specific outcome. To implement a probability distribution as an object in pgmpy, we’ll use the DiscreteFactor class.

Licensed to Brandon Behring <bb1023@nyu.edu>



---

## Raw page.get_text() output

### Page 50

20
CHAPTER 1
Why causal AI
STEP 3: DO THE STATISTICAL INFERENCE
Step 3 does the statistical estimation, and there are several ways we could estimate the
quantities on the right side of that equation. For example, we could use a convolu-
tional neural network to model E(I |W =w, A =a, D =d, T =t), and build a probability
model of the joint distribution P(A, D, T ). The choice of statistical modeling
approach involves the usual statistical trade-offs, such as ease-of-use, bias and variance,
scalability to large data, and parallelizability.
 Other books go into great detail on preferred statistical methods for step 3. I take
the strongly opinionated view that we should rely on the “commodification of infer-
ence” trend in statistical modeling and machine learning frameworks to handle step
3, and instead focus on honing our skills on steps 1 and 2: figuring out the right ques-
tions to ask, and representing the possible causes mathematically.
 As you’ve seen in this section, our journey into causal AI is scaffolded by a three-
step process, and the essence of causal thinking emerges prominently in the first two
steps. Step 1 invites us to frame the right causal questions, while step 2 illuminates the
mathematics behind these questions. Step 3 leverages patterns we’re well-accustomed
to in traditional statistical prediction and inference.
 Using this structured approach, we’ll transition in the coming chapters from
purely predictive machine learning models—like the deep latent variable models you
might be familiar with from MNIST—to causal machine learning models that offer
deeper insights into and answers to our causal questions. First, we will review the
underlying mathematics and machine learning foundations. Then, in part 2 of the
book, we’ll delve into crafting the right questions and articulating them mathemati-
cally for steps 1 and 2. For step 3, we’ll harness the power of contemporary tools like
PyTorch and other advanced libraries to bridge the causal concepts with cutting-edge
statistical learning algorithms.
Summary
Causal AI seeks to augment statistical learning and probabilistic reasoning with
causal logic.
Causal inference helps data scientists extract more causal insights from observa-
tional data (the vast majority of data in the world) and experimental data.
When data scientists can’t run experiments, causal models can simulate experi-
ments from observational data.
They can use these simulations to make causal inferences, such as estimating
causal effects, and even to prioritize interesting experiments to run in real life.
Causal inference also helps data scientists improve decision-making in their
organizations through algorithmic counterfactual reasoning and attribution.
Causal inference also makes machine learning more robust, decomposable, and
explainable.
Causal analysis is useful for formally analyzing fairness in predictive algorithms
and for building fairer algorithms by parsing ordinary statistical bias into its
causal sources.
Licensed to Brandon Behring <bb1023@nyu.edu>


### Page 51

21
Summary
The commodification of inference is a trend in machine learning that refers to how
universal modeling frameworks like PyTorch continuously automate the nuts
and bolts of statistical learning and probabilistic inference. The trend reduces
the need for the modeler to be an expert at the formal and statistical details of
causal inference and allows them to focus on turning domain expertise into
better causal models of their problem domain.
Types of causal inference tasks include causal discovery, intervention prediction,
causal effect estimation, counterfactual reasoning, explanation, and attribution.
The way we build and work with probabilistic machine learning models can be
extended to causal generative models implemented in probabilistic machine
learning tools such as PyTorch.
Licensed to Brandon Behring <bb1023@nyu.edu>


### Page 52

22
A primer on probabilistic
generative modeling
Chapter 1 made the case for learning how to code causal AI. This chapter will intro-
duce some fundamentals we need to tackle causal modeling with probabilistic
machine learning, which roughly refers to machine learning techniques that use
probability to model uncertainty and simulate data. There is a flexible suite of
cutting-edge tools for building probabilistic machine learning models. This chap-
ter will introduce the concepts from probability, statistics, modeling, inference, and
This chapter covers
A primer on probability models
Computational probability with the pgmpy and
Pyro libraries
Statistics for causality: data, populations, and
models
Distinguishing between probability models and
subjective Bayesianism
Licensed to Brandon Behring <bb1023@nyu.edu>


### Page 53

23
2.1
Primer on probability
even philosophy that we will need in order to implement key ideas from causal infer-
ence with the probabilistic machine learning approach.
 This chapter will not provide a mathematically exhaustive introduction to these
ideas. I’ll focus on what is needed for the rest of this book and omit the rest. Any data
scientist seeking causal inference expertise should not neglect the practical nuances
of probability, statistics, machine learning, and computer science. See the chapter
notes at www.altdeep.ai/causalAIbook for recommended resources where you can get
deeper introductions or review materials.
 In this chapter, I’ll introduce two Python programming libraries for probabilistic
machine learning:
pgmpy is a library for building probabilistic graphical models. As a traditional
graphical modeling tool, it is far less flexible and cutting-edge than Pyro but
also easier to use and debug. What it does, it does well.
Pyro is a general probabilistic machine learning library. It is quite flexible, and it
leverages PyTorch’s cutting-edge gradient-based learning techniques.
Pyro and pgmpy are the general modeling libraries we’ll use in this book. Other
libraries we’ll use are designed specifically for causal inference.
2.1
Primer on probability
Let’s review the probability theory you’ll need to work with this book. We’ll start with a
few basic mathematical axioms and their logical extensions without yet adding any
real-world interpretation. Let’s begin with the concrete idea of a simple three-sided
die (these exist).
2.1.1
Random variables and probability
A random variable is a variable whose possible values are the numerical outcomes of a
random phenomenon. These values can be discrete or continuous. In this section,
we’ll focus on the discrete case. For example, the values of a discrete random variable
representing a three-sided die roll could be {1, 2, 3}. Alternatively, in a 0-indexed pro-
gramming language like Python, it might be better to
use {0, 1, 2}. Similarly, a discrete random variable rep-
resenting a coin flip could have outcomes {0, 1} or
{True, False}. Figure 2.1 illustrates three-sided dice.
 The typical approach to notation is to write ran-
dom variables with capitals like X, Y, and Z. For exam-
ple, suppose X represents a die roll with outcomes {1,
2, 3}, and the outcome represents the number on the
side of the die. X=1 and X=2 represent the events of
rolling a 1 and 2 respectively. If we want to abstract
Figure 2.1
Three-sided dice each
represent a random variable with
three discrete outcomes.
Licensed to Brandon Behring <bb1023@nyu.edu>


### Page 54

24
CHAPTER 2
A primer on probabilistic generative modeling
away the specific outcome with a variable, we typically use lowercase. For example, I
would use “X=x” (e.g., X=1) to represent the event “I rolled an ‘x’!” where x can be any
value in {1, 2, 3}. See figure 2.2.
Each outcome of a random variable has a probability value. The probability value is
often called a probability mass for discrete variables and a probability density for continu-
ous variables. For discrete variables, probability values are between zero and one, and
summing up the probability values for each possible outcome yields 1. For continuous
variables, probability densities are greater than zero, and integrating the probability
densities over each possible outcome yields 1.
 Given a random variable with outcomes {0, 1} representing a coin flip, what is the
probability value assigned to 0? What about 1? At this point, we just know the two val-
ues are between zero and one, and that they sum to one. To go beyond that, we have
to talk about how to interpret probability. First, though, let’s hash out a few more
concepts.
2.1.2
Probability distributions and distribution functions
A probability distribution function is a function that maps the random variable outcomes
to a probability value. For example, if the outcome of a coin flip is 1 (heads) and the
probability value is 0.51, the distribution function maps 1 to 0.51. I stick to the stan-
dard notation P(X=x), as in P(X=1) = 0.51. For longer expressions, when the random
variable is obvious, I drop the capital letter and keep the outcome, so P(X=x)
becomes P(x), and P(X=1) becomes P(1).
 If the random variable has a finite set of discrete outcomes, we can represent the
probability distribution with a table. For example, a random variable representing out-
comes {1, 2, 3} might look like figure 2.3.
     In this book, I adopt the common notation P(X ) to
represent the probability distribution over all possible
outcomes of X, while P(X=x) represents the probability
value of a specific outcome. To implement a probability
distribution as an object in pgmpy, we’ll use the
DiscreteFactor class.
X=2
Figure 2.2
X represents the outcome of a
three-sided die roll. If the die roles a 2, the
observed outcome is X=2.
X
P(X)
Figure 2.3
A simple tabular
representation of a discrete
distribution
Licensed to Brandon Behring <bb1023@nyu.edu>
