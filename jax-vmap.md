# `vmap`

`vmap` is one of the four core functionalities in Jax (next to `pmap`, `grad`, and `jit`). 

It allows you to compile a function that was originally intended for scalar inputs and make it accessible for vector inputs. This is useful for things like batch processing in machine learning. 

Here is a simple example. 

```python
import jax 
import jax.numpy as jnp     
def add_two(input_value):
	return input_value + 2
sample_vector = [1,2,3,4,5,6,7]
add_two_vmap = jax.vmap(add_two)
result = add_two_vmap(jnp.array(sample_vector))
print(result)
```
which prints
```
[3 4 5 6 7 8 9]
```

This is a simple example so it would have been easy to create a loop inside the function to handle this case. However, this is inefficient. Jax automatically pre-compiles and vectorizes the function before running it which means we don't need to perform the operation in a loop. 
