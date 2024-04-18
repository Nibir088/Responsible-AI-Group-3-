
# Machine Unlearning

Machine unlearning is a concept that mirrors the human ability to forget, granting Artificial Intelligence (AI) systems the power to discard specific information. It is the converse of machine learning that allows the models to unlearn or forget certain aspects of their training data. It holds promise not only for complying with regulations but also for rectifying factually incorrect information within a model.
Authors in paper [2] addresses the difficulty of removing a user’s data from machine learning models once it has been included. It introduces a framework called SISA training to expedite the unlearning process by strategically limiting the influence of a data point during training. The paper evaluates SISA training across various datasets and demonstrates its effectiveness in reducing computational overhead associated with unlearning, while maintaining practical data governance standards.

## Motivation

These days deep learning models are trained on a large dataset. It’s difficult for users to revoke access and request deletion of their data once shared online. ML models potentially memorizes the data that raises privacy concerns. Consider another example where an employee leaves the company and the employee wants the company removed all his information from the trained model. 
However, the challenge is we can not train the model from scrath as it is time consuming and cost inefficient. Moreover, some of the used data may be changed later. For instance, consider the example "Man City is the current winner of uefa champion league." However, this information will change every year. So, model should update this information by forgetting the previous information. Therefore, unlearning data from ML models is notoriously challenging, yet crucial for privacy protection. In this context, the paper introduces SISA (Sharded, Isolated, Sliced, and Aggregated) training, a framework designed to expedite the unlearning process and reduce computational overhead.

## Formalizing Machine Unlearning
Unlearning is difficult due to the stochastic and complex nature of ML training methods. There’s no clear method to measure the impact of a single data point on the model’s parameters. Moreover, randomness in training, such as batch sampling and non-deterministic parallelization, complicates unlearning. Model updates reflect all prior updates, making it hard to isolate the influence of a single data point. Authors of paper [2] formalize machine unlearning:


## Conclusion & Discussion

In conclusion, the paper [2] introduces a framework to expedite the unlearning process by strategically limiting the influence of a data point in the training procedure. Moreover, it is applicable to any learning algorithm but particularly beneficial for stateful algorithms like stochastic gradient descent for deep neural networks.

**Computational Overhead Reduction**: Demonstrates how SISA training significantly reduces the computational overhead associated with unlearning. Even it shows improvement in scenarios where unlearning requests are uniformly distributed across the training set.

**Practical Data Governance**: Contributes to practical data governance by enabling machine learning models to unlearn data efficiently, thereby supporting the right to be forgotten as mandated by privacy regulations like GDPR.

**Evaluation Across Datasets**: Provides an extensive evaluation of the SISA training approach across several datasets from different domains. This shows its effectiveness in handling streams of unlearning requests with minimal impact on model accuracy.

