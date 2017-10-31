

This is read me file.

Flow of model training code would look like below,

```python
x, y = load_data(file_path)

# define graph
loss = ...
optimizer = ...

hooks = [NanTensorHook_Ram_Created(loss, fail_on_nan_loss=True)]
with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
  for epoch in range(epochs):
  pass
```

We override the defualt NaNTensorHook() class.
Provide our need of actions on the after_run(self, run_context, run_values): function.


