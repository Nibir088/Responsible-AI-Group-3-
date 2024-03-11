# Paper 30: Intriguing Properties of Neural Networks

## Motivation
The motivation for this paper is to explore two important properties of deep neural networks and understand how they can be vulnerable to adversarial examples. Deep neural networks are really good at tasks like recognizing images or speech because they're excellent at finding patterns in data. However, they sometimes learn in ways that are difficult for humans to understand. While humans can often understand the reasoning behind a decision or inference, the inner workings of DNNs can be inscrutable. This lack of transparency raises concerns, especially in critical applications where the reasoning behind a decision is as important as the decision itself.

## Findings
### Property 1
The paper says that when we look at the individual parts of neural networks, like specific neurons, they don't seem to be doing anything special on their own. Instead, it's the overall pattern of connections in the network that matters more for understanding the meaning of things.
This property is concerned with the semantic information of individual units. (Semantic information: understanding what the network has learned and how it represents concepts or features in the data it's trained on)
An example of the Property 1: Let's say we have a neural network trained to recognize different types of animals in images. The first property mentioned in the paper suggests that when we look at the individual neurons or units in the network, we might expect each one to represent a specific feature of an animal, such as a certain shape or color.
However, the research finds that if we look at what makes each neuron "light up" the most (meaning, what kind of input image makes it activate strongly), it's not always straightforward to interpret what that neuron represents. For example, a neuron might activate strongly for images of both cats and dogs, or it might activate for completely unrelated things like trees or cars.

### Property 2
This property is concerned with a DNN’s performance against adversarial examples (inputs that have been intentionally and slightly perturbed that may cause a DNN to produce incorrect outputs).
Consider a state-of-the-art deep neural network that generalises well on an object recognition task. We expect such a network to be robust to small perturbations of its input, because small perturbations cannot change the object category of an image. However, we find that applying an imperceptible non-random perturbation to a test image, it is possible to arbitrarily change the network’s prediction (see figure 5). These perturbations are found by optimising the input to maximise the prediction error.

## Mathematical Formulation and Experiments
$\phi(x)$: Activation values of some layer in a neural network when the input image 𝑥 is processed through it. 

$x \in R$: Where 𝑥 represents an input image in a dataset with m features

To reveal the above properties, they perform experiments on several datasets. One of these is the classic MNIST dataset, with the following architecture:
- A simple fully connected network with one or more hidden layers and a Softmax classifier. We refer to this network as “FC”. 
- A classifier trained on top of an autoencoder. We refer to this network as “AE”.
  
Previous research treated the activation of a hidden unit in a neural network as if it represented a meaningful feature of the input image. They would look for images that maximised the activation of a single hidden unit to try to understand what that unit was detecting in the input image.
Mathematically, they would do the following:
$x' = \arg\max \langle \mathbf{\Phi}(x), e_i \rangle$. The "arg max" operation is used to find the input image 𝑥’ that maximizes a particular property, in this case, the activation value of a specific hidden unit in a neural network.
$\arg\max \langle \mathbf{\Phi}(x), e_i \rangle$ finds the image in the dataset that causes the highest activation of the i-th hidden unit.

However, the experiments conducted in this paper state that the semantic properties observed in the images are not specific to particular directions or basis vectors in the neural network's hidden layer space, but rather generalize across many different directions or axes in the input space.

In other words, for any random direction $\(\nu \in \mathbb{R}^n\)$, we have

$$x' = \arg\max \langle \mathbf{\Phi}(x), \nu \rangle$$

## Conclusion
In summary, this paper explores two key aspects of deep neural networks and how they can be tricked by subtle changes in data. Firstly, it points out that individual parts of these networks, like specific neurons, don't always represent clear features on their own. Instead, it's the overall connections between these parts that matter most for understanding how the network works.

Secondly, the paper discusses how deep neural networks can make mistakes when faced with tiny alterations to their input data. Even small changes that humans can't notice can cause these networks to give completely wrong answers. This highlights the need for more reliable and stable neural network models, especially in tasks like image recognition.

The experiments done in the paper, particularly on datasets like MNIST, provide valuable insights into how neural networks learn and make decisions. These findings challenge some of the assumptions we've had about how neural networks function and how they should be trained.

# Paper 31 (Explaining and Harnessing Adversarial Examples)

## Motivation

Despite the success of deep neural networks, they are vulnerable to adversarial examples - inputs crafted by applying small, intentional perturbations that cause the model to misclassify them with high confidence. An example of such is shown in Figure-1. Previously researchers attempt at explaining the phenomenon of adversarial examples focused on the extreme non-linearity and overfitting of neural networks. They also believed that adversarial examples are rare and finely tile the input space like rational numbers among the reals. In this work, author investigate this two commonly accepted notions. Particularlly, this work aims to understand the root cause of adversarial examples and explore ways to mitigate their impact. 

<p align="center">
  <img src="img/gibbon.png" alt="Description of the image">
</p>
<p align="center"><em>Figure 1: A demonstration of fast adversarial example generation applied to GoogLeNet on ImageNet. By adding an imperceptibly small vector whose elements are equal to
the sign of the elements of the gradient of the cost function with respect to the input. </em></p>

## Methodology

__Explaining Adversarial Examples__

First, authors argue that the primary cause of neural networks' vulnerability to adversarial perturbations is their linear nature, rather than their non-linearity or overfitting, as previously hypothesized. They propose a linear explanation for the existence of adversarial examples.

In many problems, the precision of individual input features is limited. For example, digital images often use only 8 bits per pixel, discarding information below 1/255 of the dynamic range. Because the precision of the features is limited, it is not rational for the classifier to respond differently to an input `x` and an adversarial input `x̃ = x + η` if every element of the perturbation `η` is smaller than the precision of the features.

__Simple Linear Model__: Consider the dot product between a weight vector `w` and an adversarial example `x̃`: w<sup>T</sup> `x̃` = w<sup>T</sup> `x` + w<sup>T</sup>η
The adversarial perturbation η causes the activation to grow by w<sup>T</sup> η. We can maximize this increase subject to the max norm constraint on η by assigning η = sign(w). If w has n dimensions and the average magnitude of an element of the weight vector is m, then the activation will grow by mn. It does not grow with the dimensionality of the problem, but the change in activation caused by perturbation by η can grow linearly with n, then for high-dimensional problems, we can make many infinitesimal changes to the input that add up to one large change to the output. This happen because ||η||<sub>∞<sub>. 

This explanation shows that a simple linear model can have adversarial examples if its input has sufficient dimensionality. The authors argue that neural networks, which are intentionally designed to behave linearly for optimization purposes, are also vulnerable to adversarial perturbations due to their linear nature.

__Complex Deep Model__: Let `x` be the input, `θ` be the model parameters, `y` be the targets, and `J(θ, x, y)` be the cost function. The "fast gradient sign method" for generating adversarial examples is: 
  $$η = ϵ × sign(∇_x J(θ, x, y))$$
Here ϵ is a small constant, and sign(∇_x J(θ, x, y)) is the sign of the gradient of the cost with respect to the input. The perturbed input `x̃` = x + η is likely to be misclassified. 

__Adversarial Training__

The authors propose adversarial training as a way to regularize models and improve their robustness against adversarial examples. This training procedure involves exposing the model to both clean and adversarial examples during the training process.

__Simple Linear Model__: For linear models, such as logistic regression, the fast gradient sign method is exact. Consider a binary classification problem with labels `y ∈ {-1, 1}`, where the model computes the probability of the positive class as:

  $$P(y = 1 | x) = σ(w^{T} x + b)$$

Here, σ(z) is the logistic sigmoid function. To train the model, the objective of adversarial training is to minimize the following equation:

  $$E_{x, y \sim p_{data}} [ζ(y(||w||_1 - w^T x - b))]$$

where ζ(z) = log(1 + exp(z)) is the softplus function. This key difference between this method and L1 regularization is ***the training cost which is added (this method) instead of subracting (L1-regularization) from the model's activation during the training***. However, in underfitting scenarios, adversarial training could worsen underfitting. L1 weight decay persists even in cases of good margin, making it more "worst-case" than adversarial training. In another word, we can interpret this as learning to play an adversarial game or minimizing an upper bound on the expected cost over noisy samples with noise from U(-ε, ε) added to the inputs.

__Deep Model__: For deep model, the authors propose adversarial training, where the model is trained on a mixture of clean and adversarial examples:

  $$\overline{J}(θ, x, y) = αJ(θ, x, y) + (1 - α)J(θ, x + sign(∇_x J(θ, x, y)))$$
  
where α controls the balance between clean and adversarial examples.

The key idea is to continually update the supply of adversarial examples to make them resistant to the current version of the model. This approach means that the model is trained on a mixture of clean and adversarial examples, with the adversarial examples being generated on-the-fly using the fast gradient sign method.

Authors also show that:

- During training, the model learns to resist adversarial perturbations by adjusting its parameters to minimize the adversarial objective function. This can be seen as a form of active learning, where the model requests labels on new points (adversarial examples) to improve its robustness.
- This training can provide an additional regularization benefit beyond that provided by techniques like dropout.
- This can improve the model's robustness against adversarial examples without sacrificing accuracy on clean inputs.

## Experimenation Results


__Linear Model Training__

- The fast gradient sign method reliably causes linear models, such as softmax regression, to have a high error rate on adversarial examples.
- For example, on the MNIST dataset, a shallow softmax classifier achieved an error rate of 99.9% with an average confidence of 79.3% on adversarial examples generated with ε = 0.25.
- Adversarial training for linear models can be seen as minimizing the worst-case error when the data is perturbed by an adversary, similar to L1 regularization.
- However, adversarial training differs from L1 weight decay in that the penalty can eventually disappear if the model learns to make confident enough predictions, while L1 weight decay overestimates the damage an adversary can do.
- An example of the results is shown in Figure-2.

<p align="center">
  <img src="img/adv_linear_training.png" alt="Description of the image">
</p>

__Deep Model Training__

- On the MNIST dataset, adversarial training reduced the error rate of a maxout network on adversarial examples from 89.4% to 17.9%.
- The adversarially trained model became more resistant to adversarial examples but still made highly confident predictions on misclassified examples (average confidence of 81.4%).
- Adversarial training also resulted in the learned model weights becoming more localized and interpretable.
- Using a larger model (1600 units per layer instead of 240) and early stopping based on the adversarial validation set error, the authors achieved an error rate of 0.782% on the MNIST test set, which is the best reported result on the permutation-invariant version of MNIST.
- An visualization of adversarial training is shown is Figure-3.

<p align="center">
  <img src="img/adv_deep_training.png" alt="Description of the image">
</p>

__Alternative Hypotheses__

To support the linear explanation for the existence of adversarial examples proposed by the authors and refute other hypotheses, the authors shows that adversarial examples occur reliably for almost any sufficiently large value of ε. Note that the perturbation is in the correct direction (determined by the sign of the gradient). They also show that correct classifications occur only on a thin manifold where the input occurs in the data distribution. In addition, they show that most of the high-dimensional input space consists of adversarial examples and "rubbish" class examples that are confidently misclassified by the model. The result is shown in the following figure.

<p align="center">
  <img src="img/effect_of_eta.png" alt="Description of the image">
</p>

## Findings and Discussion

- Adversarial examples are a result of models being too linear, rather than too non-linear or overfitting, as previously thought.
- The generalization of adversarial examples across different models can be explained by the models learning similar linear functions when trained on the same task.
- Linear models lack the capacity to resist adversarial perturbations, while models with hidden layers (where the universal approximator theorem applies) can potentially learn to resist adversarial examples through adversarial training.
- The existence of adversarial examples suggests that deep learning models may not truly understand the underlying concepts they are trained on, but rather rely on patterns that are brittle to small perturbations.
- Developing optimization procedures that can train more non-linear models may be necessary to escape the trade-off between ease of optimization and robustness to adversarial examples.


# Paper 32: Towards Evaluating the Robustness of Neural Networks

## Motivation
The paper focuses on the vulnerability of neural networks to adversarial examples. These are inputs to a neural network that have been intentionally designed to cause the network to make a mistake. They are visually indistinguishable from natural inputs but cause the network to misclassify them. Defensive distillation, a technique previously thought to enhance the robustness of neural networks by significantly reducing the success rate of attacks aimed at generating adversarial examples, is scrutinized in this work. The authors propose three new attack algorithms that demonstrate the insufficiency of defensive distillation in significantly enhancing the robustness of neural networks. These attacks are designed to be effective across three common distance metrics and demonstrate higher success rates and effectiveness compared to existing adversarial example generation techniques. Through this work, the authors aim to highlight the need for better methods to evaluate and improve the robustness of neural networks against adversarial examples, suggesting that defenses should be tested against more powerful attacks such as those introduced in their study.

## Methodology
The methodology involves creating three novel attack algorithms tailored to different distance metrics (L0, L2, and L∞), which are commonly used in the literature to measure perturbations in adversarial examples. Each attack algorithm is meticulously designed to efficiently find adversarial examples that are minimally distant from original inputs according to their respective metrics, ensuring high success rates across both distilled and undistilled neural networks. 
- L2 Attack: 
- 
- 

## Conclusion/Critical Analysis
- The comprehensive evaluation demonstrates that defensive distillation, despite previously reported successes, does not significantly enhance the robustness of neural networks against sophisticated adversarial examples crafted using their novel attack algorithms. 

- The authors' three tailored attacks (for L0, L2, and L∞ metrics) consistently achieve 100% success rates on both distilled and undistilled networks, showcasing that these attacks can bypass distillation defenses more effectively and with less perturbation than existing methods. 

- Furthermore, their work emphasizes the critical need for more reliable metrics and methods for evaluating neural network robustness. They advocate for the development of neural networks that can resist not just current but future, potentially more powerful adversarial attacks. For this, the authors encourage the use of following evaluation approaches :
  - Employ a potent attack (such the ones this study suggests) to assess the secured model's robustness directly. Defenders should make cautious to create robustness against the L2 distance metric because stopping our L2 attack will stop our other attacks. 
  - By building high-confidence adversarial cases on an unsecured model and demonstrating that they are unable to transfer to the secured model, one can demonstrate the failure of transferability.

### Critical Analysis
-	The methodology is thorough, detailing optimized attack strategies against distilled networks and commendably using various distance metrics for a broad analysis of adversarial examples' impact on neural networks.
-	However, the paper's focus on defensive distillation exclusively as a defense mechanism could be seen as a limitation. A more holistic approach that evaluates the proposed attacks against a wider range of defense mechanisms could provide a more comprehensive view of their effectiveness and the overall robustness of neural networks.
-	Further research is needed to explore the practical implementation of adversarial attacks and develop defenses that meet real-world operational requirements of machine learning applications.



# Paper 33 (Adversarial examples in the physical world)

## Introduction/ Motivation
The rapid integration of machine learning (ML) into various aspects of technology, especially those reliant on sensory data (such as images from cameras for classification tasks), has spotlighted significant vulnerabilities. These systems are susceptible to adversarial examples—subtly modified input data designed to cause incorrect model classification. Such vulnerabilities are not merely academic curiosities; they represent tangible security risks in real-world applications. Previous research predominantly explored adversarial attacks within digital domains, assuming direct access to the ML model's input. This paper shifts focus towards a more challenging and realistic scenario where adversarial examples interact with physical-world systems, such as those interpreting sensory data, without direct digital input manipulation. Demonstrating the persistence of these vulnerabilities in physical contexts highlights an urgent need for robust defenses in ML deployments across various sectors.

## Methods
Below are the basic notations followed: 

- **X**: An image, which is typically a 3-D tensor (width × height × depth). We assume that the values of the pixels are integer numbers in the range [0, 255].

- **y_true**: The true class for the image \(X\).

- **J(X, y)**: The cross-entropy cost function of the neural network, given image \(X\) and class \(y\). We intentionally omit network weights (and other parameters) in the cost function because we assume they are fixed (to the value resulting from training the machine learning model) in the context of the paper. For neural networks with a softmax output layer, the cross-entropy cost function applied to integer class labels equals the negative log-probability of the true class given the image: \(J(X, y) = -\log p(y | X)\), this relationship will be used below.
- **$Clip_{X,\epsilon}{\{X'\}}$**: A function which performs per-pixel clipping of the image \(X\), so the result will be in\(L_{\infty}\) \epsilon neighbourhood of the source image \(X\). The exact clipping equation is as follows:
  



# Paper 34 (Adversarial Examples Are Not Bugs, They Are Features)

## Introduction / Motivation
Adversarial examples have attracted significant attention in machine learning, but the reasons for their existence and pervasiveness remain unclear.

## Methods
### Problem setup
Consider binary classification, where input-label pairs $(x, y) \in \mathcal{X} \times \{\pm 1\}$ are sampled from a (data) distribution $\mathcal{D}$; the goal is to learn a classifier $C : \mathcal{X} \rightarrow \{\pm 1\}$ which predicts a label $y$ corresponding
to a given input $x$. Features are function mappings from the input space to the real numbers.

### Useful, robust, and non-robust features
- **$\rho$-useful features**: A feature $f$ is $\rho$-useful ($\rho > 0$) if it is correlated with the true label in expectation, that is if $$\mathbb{E}_{(x,y)\sim\mathcal{D}}[y \cdot f(x)] \geq \rho.$$
- **$\gamma$-robustly useful features**: A $\rho$-useful feature $f$ is a robust feature ($\gamma$-robustly useful feature for $\gamma > 0$) if under adversarial perturbation (for some specified set of valid perturbations $\Delta$) $f$ remains $\gamma$-useful. Formally, if we have that $$\mathbb{E}_{(x,y)\sim\mathcal{D}}[\inf_{\delta \in \Delta(x)} y \cdot f(x + \delta)] \geq \gamma.$$
- **Useful non-robust features**: These features are $\rho$-useful for some $\rho$ bounded away from zero but are not $\gamma$-robust for any $\gamma \geq 0$.

### Robust and Non-Robust Feature Disentanglement
The authors isolated robust features through the generation of a new dataset derived from a robustly trained model. This dataset emphasizes only those features deemed relevant by the model. The process hinges on an optimization strategy aimed at minimizing the discrepancy between the activations produced by the original input, $x$, and its transformed version, $x_R$, in the penultimate layer of the model. The optimization goal is formulated as follows:
$$x_{r} = \underset{x_r}{\text{argmin}} \; \|g(x_r) - g(x)\|_2,$$
where $g$ denotes the mapping from an input to its representation in the robust model's penultimate layer. The robust training set is termed as $\hat{\mathcal{D}}_R$.

A non-roust training set $\hat{\mathcal{D}}_{NR}$ is alse constructed by utilizing features from a standard model (instead of a robust one).

<p align="center">
  <img src="img/Mar_13/paper34_robust_and non_robust.png" alt="Description of the image" width="50%">
</p>

### Non-Robust Feature Consruction for Standard Classification
To demonstrate the capability of non-robust features to enable significant classification accuracy on standard test sets, the authors curated a dataset such that the connection between inputs and labels relies exclusively on non-robust features. To achieve this, adversarial modifications were applied to each input $x$ to produce $x_{\text{adv}}$, ensuring classification as a specific target class $t$. The modification process is defined by the optimization problem:
$$x_{\text{adv}} = \underset{x' : \|x' - x\|_2 \leq \epsilon}{\text{argmin}} \; L_{C}(x', t),$$
with $L_{C}$ representing the loss under a standard classifier $C$ and $\epsilon$ denoting the allowable perturbation magnitude.

<p align="center">
  <img src="img/Mar_13/paper34_non_robust.png" alt="Description of the image" width="40%">
</p>

## Results / Key Findings

- **Disentangling Robust and Non-Robust Features**: Robust features contribute to the model's generalization ability in an adversarial setting, whereas non-robust features, although highly predictive, lead to vulnerability. Through experiments, the authors show it is possible to construct datasets that emphasize either robust or non-robust features, influencing the trained model's adversarial robustness or lack thereof.

<p align="center">
  <img src="img/Mar_13/paper34_result_1.png" alt="Description of the image" width="45%">
</p>

- **Non-Robust Features Suffice for Standard Classification**: Surprisingly, models trained on data that appears mislabeled to humans (but is aligned with non-robust features) still achieve high accuracy on standard test sets. This underscores the non-intuitive power of non-robust features in driving model predictions, challenging traditional notions of what constitutes useful data for learning.

<p align="center">
  <img src="img/Mar_13/paper34_result_2.png" alt="Description of the image" width="30%">
</p>

- **Transferability Arising from Non-Robust Features**: The paper delves into the phenomenon of adversarial transferability, where adversarial examples crafted for one model effectively fool another. This transferability is attributed to models learning similar non-robust features from the data, further emphasizing the intrinsic connection between adversarial examples and non-robust features.


<p align="center">
  <img src="img/Mar_13/paper34_result_3.png" alt="Description of the image" width="35%">
</p>

These findings underscore the nuanced relationship between model accuracy, robustness, and the types of features learned during training. The work suggests that achieving both high accuracy and robustness may require rethinking how models are trained and the features they prioritize.

## Critical Analysis

The research brings to light a groundbreaking perspective on adversarial examples in machine learning, showcasing that these examples aren't mere glitches but are indicative of "non-robust features" intrinsic to datasets. These features, although predictive, are fragile and easily exploitable by adversarial attacks, thus highlighting a fundamental misalignment between machine learning models' optimization for accuracy and the nuanced concept of robustness from a human perspective.
