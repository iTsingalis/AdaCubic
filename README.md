# AdaCubic  🚀
Implementation of paper: "*AdaCubic: An Adaptive Cubic Regularization Optimizer for Deep Learning*", submitted in TMLR 2025.

## Paper Abstract

A novel regularization technique called AdaCubic that adapts the weight of the cubic term is proposed. 
At the heart of AdaCubic lies an auxiliary optimization problem with cubic constraints, employed to 
dynamically adjust the weighting of the cubic term in Newton's cubic regularized method. To reduce the computation 
costs, we utilize Hutchinson’s method to approximate the Hessian matrix. We demonstrate that AdaCubic 
inherits the global and local convergence guarantees of the cubically regularized Newton method. Our experiments 
in Computer Vision and Natural Language Processing tasks show that AdaCubic outperforms or performs 
competitively with several widely used optimizers. Unlike other adaptive algorithms that require fine-tuning of 
hyper-parameters, AdaCubic is evaluated with a pre-fixed set of hyper-parameters, making it a highly 
attractive optimizer in situations where fine-tuning is not feasible. This makes AdaCubic convenient to 
researchers and practitioners alike. To the best of our knowledge, AdaCubic is the first optimizer leveraging 
the power of cubic regularization for large-scale applications

## Usage
AdaCubic optimizer implementation is designed to seamlessly integrate with PyTorch as a drop-in replacement for any 
existing optimizer. Simply by setting `create_graph=True` in the `backward()` call, you can enjoy its benefits without 
the need for additional adjustments.

```python
from AdaCubic import AdaCubic
...
optimizer = AdaCubic(model.parameters())
...
for i, (samples, labels) in enumerate(train_loader):
    ...
    def closure(backward=True):
    if backward:
        optimizer.zero_grad()
    model_outputs = model(samples)
    cri_loss = criterion(model_outputs, labels)
    
    create_graph = type(optimizer).__name__ == "AdaCubic"
    if backward:
        cri_loss.backward(create_graph=create_graph)
    return cri_loss
    ...
    optimizer.step(closure=closure)
    ...
```



## Documentation

#### `AdaCubic.__init__`

| **Argument**                           | **Description**                                                                        |
|:---------------------------------------|:---------------------------------------------------------------------------------------|
| `params` (iterable)                    | A collection of parameters to optimize, or dictionaries defining parameter groups.     |
| `eta1`(float, optional)                | Threshold related to the acceptance or rejection of the trial point *(default: 0.05)*  |
| `eta1` (float, optional)               | Threshold related to the acceptance or rejection of the trial point *(default: 0.75)*  |
| `alpha1` (float, optional)             | Constant that defined the portion of trust radius increase *(default: 2.5)*            |
| `alpha2` (float, optional)             | Constant that defined the portion of trust radius decrease *(default: 0.25)*           |
| `kappa_easy` (int, optional)           | The accuracy for the estimation of the root in Algorithm 2 *(default: 0.01)*           |
| `grad_tol` (float, optional)           | The accuracy of gradient to stop Algorithm 1 *(default: 1e-4)*                         |
| `xi0` (float, optional)                | The size of the initial trust radius *(default: 0.05)*                                 |
| `gamma1` (float, optional)             | The gamma1 parameter in the algorithm Chapter 17 in Conn's book *(default: 0.9)*       |
| `hutchinson_iters` (int, optional)     | Number of times iterations for approximating the Hessian trace. *(default: 1)*         |
| `average_conv_kernel` (bool, optional) | Compute the average of the Hessian traces of convolutional kernels. *(default: false)* |
| `solver` (str, optional)               | The solver to use *(default: exact)*                                                   |


#### `AdaCubic.step`

Performs a single optimization step.

| **Argument**                        | **Description** |
|:------------------------------------| :-------------- |
| `closure` (callable, optional)      | A closure that reevaluates the model and provides the loss as its output. *(default: None)* |


### Train and Test

```angular2
runResNet.py --task cifar10 --optimizer AdaCubic --depth 20 --seed 45 --n_epochs 200
```

```angular2
run_mlm_no_trainer.py --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --model_name_or_path bert-base-uncased --optimizer AdaCubic --num_train_epochs 10 --seed 45 --output_dir /your_root/AdaCubic/Code/MLAlgorithms/LM/mlm/
```

## Confidentiality Notice
This implementation should be treated as confidential. The repository will be made public upon paper acceptance.

## Disclaimer

This repository is provided for **peer-review purposes only** in connection with the submission of the manuscript "*AdaCubic: An Adaptive Cubic Regularization Optimizer for Deep Learning*". 
The code is shared to facilitate the review process and **must not be redistributed, modified, or made public** prior to the acceptance of the associated paper. Any use of this code outside the scope of peer review is not permitted at this stage. 
Upon acceptance of the paper, the repository will be made publicly available under an appropriate open-source license.
