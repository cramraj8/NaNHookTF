
## Small description of what should we do for our custom made NaN Tensor Hook function.

Remeber code for Flow of model training would look like below,

```python
x, y = load_data(file_path)

# data perprocessing
# define graph
# model architecture
loss = ...
optimizer = ...

hooks = [NanTensorHook_Ram_Created(loss, fail_on_nan_loss=True)]
with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
    for epoch in range(epochs):
        pass
```

1. First we override the class NaNTensorHook(). 
2. Use the statement to import the class into the current script.
```python 
from NanTensorHookCustom import NanTensorHookCustom as nanhook
```
3. Provide our need of actions under the after_run(self, run_context, run_values): function.
4. Create the instance of the custom class and provide arguments(loss, fail_on_nan_loss).
5. Use MonitoredTrainingSession instead of regular tf.Session.
6. Pass our NaN hook as an argument.

** That's all what we need to do. **

