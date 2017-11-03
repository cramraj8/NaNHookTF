
## Small description of what should we do for our custom made NaN Tensor Hook function.

Remeber code for Flow of model training would look like below,

```python
x, y = load_data(file_path)

# data perprocessing
# define graph
# model architecture
loss = ...
optimizer = ...

hooks = [nanhook(loss, fail_on_nan_loss=True)]
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

**That's all what we need to do.**
</br>
</br>
</br>
</br>
### NaN Hook handling in tensorflow-slim version
Also the tensorflow-slim version has Abstract APIs, which contradict with the conventional APIs.

There are 2 ways to overcome NaN effect.

1. **tf.check_numerics()** in **slim.learning.create_train_op()** will check for
    **NaNs** or **Infs** and throws exception.
    So we can use **try & catch** to handle the exception to do some desired work.

2. If we set **check_numerics=False** in **slim.learning.create_train_op()**
    sometimes Loss might have NaNs.
    So we need to seek for **NaNTensorHook**.
    That is the method provided here.
    
    
The slim code-flow is below,

```python
x, y = load_data(file_path)

# data perprocessing
# define graph
# model architecture
loss = ...

train_op = slim.learning.create_train_op(
   loss,
   optimizer,
   check_numerics=False,
   ...)

hooks = [NanLossHook(total_loss, fail_on_nan_loss=True)]
with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:

   while not sess.should_stop():
      sess.run(train_op)

   # create the training loop
   final = slim.learning.train(
      train_op,
      init_op=tf.global_variables_initializer(),  # Or set to 'None'
      ...)
```

**I checked the combinations** of **MonitoredTrainingSession()'s** scope with the **create_train_op()** and **train()** functions' locations.
However, this strutural-coding format brings the use of **NanLossHook Class**, where we handle our desired functions and operations.

