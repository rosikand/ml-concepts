# Multi-task learning 

In a traditional supervised ML paradigm, you have a labeled dataset, $\mathscr{D}=\left\{(\mathbf{x}, \mathbf{y})_k\right\}$, where the goal is to minimize some loss function ($\mathscr{L}$) given $\mathscr{D}$, by finding relevant parameters, $\theta$ (i.e., the **task**): 

$$
\min _\theta \mathscr{L}(\theta, \mathscr{D}).
$$

## Multi-task setup

The setup is the same as above, but instead of optimizing one objective, we want to optimize several (under the same model!) at once. 

$$
\text { Solve multiple tasks } \mathscr{T}_1, \cdots, \mathscr{T}_T \text { at once. }
$$ 

**Example** ([credit](https://pyimagesearch.com/2022/08/17/multi-task-learning-and-hydranets-with-pytorch/)): 

instead of doing this (as usual): 

image 


we do this:  

image 


Three components: 

- Model 
- Loss (objective)
- Optimization 

### Model 

We ask, should we share parameters across tasks? Or should the learnable parameters in the architecture be entirely separate for each task? The latter may look something like this (credit: CS 330 slides): 

image.png

image.png 




### Loss 

To obtain the loss, we perform a simple summation over each of the individual task's loss: 

$$
\min _\theta \sum_{i=1}^T \mathscr{L}_i\left(\theta, \mathscr{D}_i\right)
$$

Of course, some tasks might be important than others so we can performed a weighted summation: 

$$
\min _\theta \sum_{i=1}^T w_i \mathscr{L}_i\left(\theta, \mathscr{D}_i\right)
$$

Note that we can update the weights during training (as a learnable parameter) or manually determime the weights of each loss beforehand (a hyperparameter). 

### Optimization 

Optimization is straightforward and is nearly the same as in the supervised setting. The CS 330 slides defines the following sequence: 


1. Sample mini-batch of tasks $\mathscr{B} \sim\left\{\mathscr{T}_i\right\}$
2. Sample mini-batch datapoints for each task $\mathscr{D}_i^b \sim \mathscr{D}_i$
3. Compute loss on the mini-batch: $\hat{\mathscr{L}}(\theta, \mathscr{B})=\sum_{\mathscr{T}_k \in \mathscr{B}} \mathscr{L}_k\left(\theta, \mathscr{D}_k^b\right)$
4. Backpropagate loss to compute gradient $\nabla_\theta \hat{\mathscr{L}}$
5. Apply gradient with your favorite neural net optimizer 



## PyTorch Example 

Coming soon... see [this blog post](https://pyimagesearch.com/2022/08/17/multi-task-learning-and-hydranets-with-pytorch/) for an example. 

## Applications

- Tesla uses a multi-task "hydranet" model for its inference in self-driving cars. The vision-only system takes in a tuple of 8 images from the car's cameras, passes the concatenated feature vector through a shared backbone encoder, and separates each task via a decoder head. For example, Tesla uses one network for things like depth estimation, lane segmentation, etc. See [here](https://www.thinkautonomous.ai/blog/how-tesla-autopilot-works/). 

## Confusions/quesions 

- Does the input feature vector need to be the same for each task? (I believe the answer is no) 


## References 

- [CS 330](https://cs330.stanford.edu/slides/cs330_multitask_transfer_2021.pdf) 
- [Tutorial blog](https://pyimagesearch.com/2022/08/17/multi-task-learning-and-hydranets-with-pytorch/)