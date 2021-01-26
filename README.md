# Scavenger #

Scavenger is a framework built on top of Tensorflow and Tensorflow-Estimator for running distributed machine learning models on low cost resources with fluctuating availability. This system extends the **OmniLearn** framework published in *Autonomic Computing and Self Organizing Systems (ACSOS), 2020*. The OmniLearn paper is available [here](https://www.researchgate.net/publication/343054677_Taming_Resource_Heterogeneity_In_Distributed_ML_Training_With_Dynamic_Batching).



## Developing

If you want to build TensorFlow Estimator locally, you will need to [install Bazel](https://docs.bazel.build/versions/master/install.html) and [install TensorFlow]((https://www.tensorflow.org/get_started/os_setup.html)).

```sh
# To build TensorFlow Estimator whl file.
bazel build //tensorflow_estimator/tools/pip_package:build_pip_package
bazel-bin/tensorflow_estimator/tools/pip_package/build_pip_package /tmp/estimator_pip

# To run all Estimator tests
bazel test //tensorflow_estimator/...
```

To start training with Scavenger, make the following additions:

```
run_config = tf.estimator.RunConfig(..., data_dir='*path to data dir*', switched_input_fn=*your input_fn*)

optimizer = *any TF optimizer*
optimizer = tf.train.ScavengerOptimizer(optimizer, replicas_to_aggregate=*num of workers*)
replicas_hook = optimizer.make_session_run_hook(is_chief, num_tokens=0)
train_hooks.append(replicas_hook)
loss = ...
grad_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grad_vars)
```

In the ```RunConfig``` object, ```data_dir``` refers to the location of the train/test data and ```switched_input_fn``` refers to the training input fn. With ```ScavengerOptimizer```, user needs to define a hook on the master/chief worker as well and append to ```the train_hooks``` list, which is given to the ```training_hooks``` argument of the ```EstimatorSpec``` object.


*TF_CONFIG* environment variable json has an additional ```batch_size_list``` argument which is a comma separated list of per-worker batch-sizes. This parameter is updated in subsequent batchsize adjustments. The format of the parameter starts with batchsize of master, then worker-0, worker-1 and so on. The final *TF_CONFIG* for master in a 3 worker cluster looks like:
```
{"environment": "cloud", "batch_size_list": "[128,128,128]", "task": {"index": 0, "type": "master"}, "model_dir": "/data", "cluster": {"worker": ["172.17.0.4:8000", "172.17.0.5:8000", "172.17.0.6:8000"], "ps": ["172.17.0.2:8000"], "master": ["172.17.0.3:8000"]}
```

In Scavenger, while defining the input fn, the user must specify *ONLY* 3 arguments in the input fn:
* ```subset```: either train or test input fn
* ```data_dir```: the location of the input data for training/testing/cross val.
* ```node_batch_size```: the per-worker batch-size specified initially. If this value is different from ```batch_size_list``` batch-sizes in *TF_CONFIG*, the value will be overwritten to the workers batch-size in the *TF_CONFIG* environment variable.
```
def input_fn(subset='train', data_dir='/data', node_batch_size=128):
	...
	...
	return feature, labels
```

The model fn returns an EstimatorSpec object with the following arguments:
```
return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        reactive_adjustment_threshold=None,
        namescope='gradients',
        window_size=None,
        sync_mode=None,
        staleness=50,
        mini_batchsize_threshold=16,
        global_batch_size_value=512,
        asp_adjust_strategy=None,
        adjustment_mode=None,
        gradnorm_window=50,
        training_hooks=train_hooks,
        eval_metric_ops=metrics)
```

## License

[Apache License 2.0](LICENSE)
