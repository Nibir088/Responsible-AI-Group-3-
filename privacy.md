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

<p align="center">
  <img src="img/overview_pate.png" alt="Description of the image">
</p>

***Private Learning with ensembles of teachers***


We can deviside this into following 
+ __Training the ensemble of teachers__: Assume (X,Y) is the dataset, where X denotes inputs and Y is the outputs. Let m be the number of the classes in the task. The dataset is partionined into n disjoint datasets $(X_i, Y_i)$. Further, n classifiers $f_i$ called teacher is trained using $(X_i,Y_i)$. For an input $\tilde{x}$, each teacher provides a prediction $f_i(\tilde{x})$. The label count $n_j(\tilde{x})$ for a given class $j \in m$ is the number of teachers that assigned class $j$ to input $\tilde{x}$. To ensure privacy, random noise is added to the vote counts using the Laplace mechanism. This process ensures that the student model’s training does not depend on the details of any single sensitive training data point.

  <p align="center">
  $n_j​(\tilde{x})=∣{i:i\in [n],f_i​(\tilde{x})=j}∣$
</p>
<p align="center">
    $f(\tilde{x})=\text{arg max}_j​(n_j​(\tilde{x})+\text{Lap}(\frac{1}{\gamma}​))$ ,
</p>
<p align="center">
  where γ is a privacy parameter and Lap(b) is the Laplacian distribution with location 0 and scale b.

</p>
     
+ __Semi-supervised transfer of the knowledge from an ensemble to a student__: The student model is trained using non-sensitive and unlabeled data. Some of this data is labeled using the aggregation mechanism from the teacher ensemble. The student model replaces the teacher ensemble for deployment. Here privacy loss is fixed and does not increase with user queries to the student model. Even if the student model’s architecture and parameters are public or reverse-engineered, the privacy of original training dataset contributors is maintained. Furthermore, the privacy cost is determined by queries made during teacher ensemble training. In addition, various techniques were considered (distillation, active learning, etc.), but the most successful one is semi-supervised learning with GANs (PATE-G).

***Privacey analysis of the approach***


## Evaluation and Result analysis 

For the evaluation of the PATE and its generative variant PATE-G, the authors first train a teacher ensemble for each dataset. The trade-off between label accuracy and privacy depends on the number of teachers in the ensemble. Further, they minimize the privacy budget spent on training the student by using as few queries to the ensemble as possible. The experiments focus on MNIST and SVHN datasets, comparing the private student’s accuracy with that of a non-private model trained on the entire dataset under different privacy guarantees.

***TRAINING AN ENSEMBLE OF TEACHERS PRODUCING PRIVATE LABELS***

The study evaluates how well the MNIST and SVHN datasets can be partitioned without significantly impacting individual teacher performance. Despite injecting substantial random noise for privacy, ensembles of 250 teachers achieve accurate aggregated predictions: 93.18% accuracy for MNIST and 87.79% for SVHN, with a low privacy budget of ε = 0.05 per query. Key findings of this experiments are as follows:
+ __Prediction Accuracy__: For MNIST and SVHN datasets, even with n = 250 teachers, individual teacher test accuracy remains reasonably high. For MNIST, average test accuracy is 83.86% and for SVHN, average test accuracy is 83.18%. The larger size of SVHN compensates for its increased task complexity. These findings highlight the delicate balance between privacy, ensemble size, and teacher accuracy in the context of privacy-preserving machine learning.
+ __Prediction confidence__: Privacy of predictions by the teacher ensemble requires agreement among a quorum of teachers. Data-dependent privacy analysis provides stricter bounds when the quorum is strong. Authors find the gap between the most popular and second most popular label. Even as the ensemble size (n) increases, the gap remains larger than 60% of the teachers, allowing accurate label output despite noise during aggregation (Follow figure 3).
<p align="center">
  <img src="img/results_ensemble_teacher.png" alt="Description of the image">
</p>

+ __Noisy Aggregation__: Authors consider three ensembles with varying numbers of teachers: $n \in$ {10, 100, 250}. Larger ensemble sizes are necessary to mitigate the impact of noise injection on accuracy. The accuracy of test set labels inferred by the noisy aggregation mechanism is reported for different values of ε.
Notably, the number of teachers must be substantial to maintain accuracy despite noise injection. (Follow Figure 2)

***Semi-Supervised Training of the Student with Pprivacy***

To reduce the privacy budget spent on student training, authors are interested in making as few label queries to the teachers as possible. We therefore use the semi-supervised training approach. For MNIST dataset, the student uses 9,000 samples, with subsets of 100, 500, or 1,000 labeled using noisy aggregation and for SVHN, the student has 10,000 training inputs, labeling 500 or 1,000 samples. Student performance is evaluated on remaining test samples.  Leveraging semi-supervised training with GANs, the MNIST and SVHN students achieve impressive accuracies.
Specifically, the MNIST student achieves 98.00% accuracy with strict differential privacy bound of ε = 2.04 (at $10^-5$ failure probability), and the SVHN student achieves 90.66% accuracy with privacy bound of ε = 8.19 (likely due to more queries to the aggregation mechanism). These results surpass the differential privacy state-of-the-art for these datasets. (Follow Figure 4)
<p align="center">
  <img src="img/privacy_61_table.png" alt="Description of the image">
</p>
