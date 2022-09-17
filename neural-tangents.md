# Neural tangents: infinite width networks 

Kernel functions are a tough concept to understand in mathematics. The basic idea behind it is to transform input data from a lower-dimensional space, which is originally non-linearly seperable, into a higher-dimensional space where it then becomes linearly seperable. This allows use to calculate expressions pertaining to machine learning in closed form (i.e., the kernel function allows the use of a kernel trick which is essentially called a "generalized dot product"). As such, we can just train a linear model (i.e., SVM) in this higher dimensional space to e.g., classify data. A visual depiction is shown below ([image source](https://towardsdatascience.com/the-kernel-trick-c98cdbcaeb3f)):

https://miro.medium.com/max/1400/1*zWzeMGyCc7KvGD9X8lwlnQ.png


Why is this relevant for neural networks? Because... 

## Infinite width networks 

Recent research has shown that neural networks follow a similar principle to the above. More precisely, as the width of the neural network increases towards infinity, the network simplifies to a linear model with a multivariate Gaussien distribution for its output. Let's visualize this (credit: slides from [tutorial](https://iclr.cc/virtual_2020/poster_SklD9yrFPS.html)): 






## The neural tangents library 

A team of researchers at Google released a library in Jax, called `neural-tangents` ([link](https://github.com/google/neural-tangents)) that mimmicks a PyTorch-like API for constructing *infinite*-width neural networks. It is a drop-in replacement for the Jax `stax` neural network API (which is similar to PyTorch's API). Thus, you can define these infinite-width NN's in a relatively simple manner thanks to this library (in the past, you'd need to mathematically derive the kernel expression for each individual network). Here is an example from the README: 

**Model**:

```python
from jax import random
from neural_tangents import stax

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512), stax.Relu(),
    stax.Dense(512), stax.Relu(),
    stax.Dense(1)
)

key1, key2 = random.split(random.PRNGKey(1))
x1 = random.normal(key1, (10, 100))
x2 = random.normal(key2, (20, 100))

nngp, ntk = kernel_fn(x1, x2, ('nngp', 'ntk'))
```

where `kernel_fn` gives back both a Neural Network Gaussian Process (NNGP) and Neural Tangent Kernel (NTK). The former is used for Bayesian inference and the latter can be used via the familiar gradient descent paradigm. 


**Inference**:

The `kernel_fn` (which, remember, was derived automatically from your stax-based network) provides you with a prediction function for inference. The best part? No need for stochastic gradient descent training... everything is in closed-form via a *Gaussian process* (just a line of code!). 

```python
# get the predict_fn 
predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,
                                                      y_train)

# use it for inference 
y_test_nngp, y_test_ntk = predict_fn(x_test=test_feature_vector, get=('nngp', 'ntk'))
```



## References 

- Main reference (with links to other references) is the [neural-tangents library repo](https://github.com/google/neural-tangents). 
- Tutorial (video + slides): https://iclr.cc/virtual_2020/poster_SklD9yrFPS.html
- Paper: [Neural Tangents: Fast and Easy Infinite Neural Networks in Python](https://arxiv.org/abs/1912.02803)
- Paper: [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165)
- Paper: 
- Blog post 1: https://lilianweng.github.io/posts/2022-09-08-ntk/
- Blog post 2: https://rajatvd.github.io/NTK/

