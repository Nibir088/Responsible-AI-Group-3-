## Paper 30

## Paper 31 (Explaining and Harnessing Adversarial Examples)

__Motivation__

Despite the success of deep neural networks, they are vulnerable to adversarial examples - inputs crafted by applying small, intentional perturbations that cause the model to misclassify them with high confidence. Previously researchers attempt at explaining the phenomenon of adversarial examples focused on the extreme non-linearity and overfitting of neural networks. They also believed that adversarial examples are rare and finely tile the input space like rational numbers among the reals. In this work, author investigate this two commonly accepted notions. Particularlly, this work aims to understand the root cause of adversarial examples and explore ways to mitigate their impact. 

__Methodology__


First, authors argue that the primary cause of neural networks' vulnerability to adversarial perturbations is their linear nature, rather than their non-linearity or overfitting, as previously hypothesized. They propose a linear explanation for the existence of adversarial examples.

In many problems, the precision of individual input features is limited. For example, digital images often use only 8 bits per pixel, discarding information below 1/255 of the dynamic range. Because the precision of the features is limited, it is not rational for the classifier to respond differently to an input `x` and an adversarial input `x̃ = x + η` if every element of the perturbation `η` is smaller than the precision of the features.

__Simple Linear Model__ Consider the dot product between a weight vector `w` and an adversarial example `x̃`: w<sup>T</sup> `x̃` = w<sup>T</sup> `x` + w<sup>T</sup>η
The adversarial perturbation η causes the activation to grow by w<sup>T</sup> η. We can maximize this increase subject to the max norm constraint on η by assigning η = sign(w). If w has n dimensions and the average magnitude of an element of the weight vector is m, then the activation will grow by mn. It does not grow with the dimensionality of the problem, but the change in activation caused by perturbation by η can grow linearly with n, then for high-dimensional problems, we can make many infinitesimal changes to the input that add up to one large change to the output. This happen because ||η||<sub>∞<sub>. 

This explanation shows that a simple linear model can have adversarial examples if its input has sufficient dimensionality. The authors argue that neural networks, which are intentionally designed to behave linearly for optimization purposes, are also vulnerable to adversarial perturbations due to their linear nature.

__Complex Deep Model__ Let `x` be the input, `θ` be the model parameters, `y` be the targets, and `J(θ, x, y)` be the cost function. The "fast gradient sign method" for generating adversarial examples is: η = ϵ × sign(∇_x J(θ, x, y))
Here ϵ is a small constant, and sign(∇_x J(θ, x, y)) is the sign of the gradient of the cost with respect to the input. The perturbed input `x̃` = x + η is likely to be misclassified. The authors propose adversarial training, where the model is trained on a mixture of clean and adversarial examples:

[\
`J̃`(θ, x, y) = αJ(θ, x, y) + (1 - α)J(θ, x + sign(∇_x J(θ, x, y)))
\]

## Paper 32

## Paper 33

## Paper 34
