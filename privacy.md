# paper 59
# paper 60
# paper 61: SEMI-SUPERVISED KNOWLEDGE TRANSFER FOR DEEP LEARNING FROM PRIVATE TRAINING DATA

The paper presents a semi-supervised learning approach called Private Aggregation of Teacher Ensembles (PATE). It protects the privacy of sensitive training data by training an ensemble of teacher models on disjoint datasets.
These teacher models are then used to train a student model without direct access to the sensitive data, ensuring privacy even against adversaries with access to the student model’s parameters. The approach is validated with strong privacy guarantees and high utility on benchmark datasets like MNIST and SVHN.
Further authors demonstrate its applicability to deep learning methods without assumptions on the learning algorithm.

## Motivation
The paper addresses the challenge of training machine learning models on sensitive data, such as medical records or personal photographs. ML model often inadvertently memorize and expose private information. The authors aim to develop a method that provides strong privacy guarantees for training data to prevent such unintended disclosures. Key contribution of this work as follows:

- Introduction of Private Aggregation of Teacher Ensembles (PATE) approach that ensures privacy by aggregating the knowledge of multiple models trained on disjoint datasets.
- Demostrate the applicability of PATE methods in machine learning algorithms and achieve state-of-the-art privacy and utility trade-offs.
- The use of semi-supervised learning to enhance the student model’s performance without compromising privacy.
- Provision of formal privacy guarantees in terms of differential privacy, ensuring that the student model’s training is not influenced by any single sensitive data point.
- Empirical validation of the approach on benchmark datasets like MNIST and SVHN, demonstrating competitive accuracy with meaningful privacy bounds.

## Methodology

***Private Aggregation of Teacher Ensembles (PATE)***

PATE is an innovative approach that ensures privacy during machine learning model training by aggregating knowledge from multiple teacher models trained on disjoint datasets. It ultimately enabling the creation of a student model without direct access to sensitive data. The model has five major components: $(i)$ sensitive data, $(ii)$ teacher models, $(iii)$ student model, $(iv)$ aggregate teacher, and $(v)$ privacy protection. A short description of these components are given below:

+ __Sensitive Data__: This represents the private training data that is divided into multiple disjoint datasets.
+ __Teacher Models__: Each dataset is used to train a separate “teacher” model. These models are knowledgeable about their respective datasets but do not share information with each other to maintain data privacy.
+ __Student Model__: A “student” model is trained using public, non-sensitive data that is labeled based on the aggregated output of the teacher models. The student model learns to mimic the ensemble of teachers.
+ __Aggregate Teacher__: The predictions from all teacher models are aggregated to form a consensus on the output for new data points.
+ __Privacy Protection__: The approach ensures privacy by limiting the student’s exposure to the teachers’ knowledge, which is quantified and bounded.

First, the sensitive data are partitioned into $n$ disjoint dataset. Each dataset is used to train a teacher model and $n$ teacher model is aggregated to a new teacher model. Aggregate teacher model provides the predicted labels. This labels is open to used but the aggregate model and data is not accessible by the adversary. On the other hand, a student model is trained on public data using the labels provided by the teacher model. This process ensures that the student model learns to accurately mimic the ensemble without having direct access to the sensitive training data. The student’s training is influenced by noisy voting among the teachers, which is a mechanism to protect the privacy of the training data. The student model can then be deployed for predictions without compromising the privacy of the sensitive data it was trained on. An overview of this approach is shown in the figure below.

