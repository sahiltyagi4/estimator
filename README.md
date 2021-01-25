-----------------
| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/python/tf/estimator) |

Scavenger is a framework for running distributed machine learning models on low cost resources with fluctuating availability. This system is built on top of the **OmniLearn** framework published in *Autonomic Computing and Self Organizing Systems (ACSOS), 2020*. The OmniLearn paper is available [here](https://www.researchgate.net/publication/343054677_Taming_Resource_Heterogeneity_In_Distributed_ML_Training_With_Dynamic_Batching).

This branch 'scaven' aims to do variable and dynamic batching without the kill-restart technique used so far. Here, whenever the necessary condition
for readjustment is encountered, the training loop is terminated (and the model parameters checkpointed). But the TF server and and outer-loop isn't. They
are re-run again with the new input fn. Here, the new input fn means the same initial input fn but with a different batch-size. Will add more details as
progress is made. 

TensorFlow Estimator is a high-level TensorFlow API that greatly simplifies machine learning programming.
Estimators encapsulate training, evaluation, prediction, and exporting for your model.

## Getting Started

See our Estimator [getting started guide](https://www.tensorflow.org/guide/estimators) for an introduction to the Estimator APIs.

## Installation

`tf.estimator` is installed when you install the TensorFlow pip package. See [Installing TensorFlow](https://www.tensorflow.org/get_started/os_setup.html) for instructions.

## Developing

If you want to build TensorFlow Estimator locally, you will need to [install Bazel](https://docs.bazel.build/versions/master/install.html) and [install TensorFlow]((https://www.tensorflow.org/get_started/os_setup.html)).

```sh
# To build TensorFlow Estimator whl file.
bazel build //tensorflow_estimator/tools/pip_package:build_pip_package
bazel-bin/tensorflow_estimator/tools/pip_package/build_pip_package /tmp/estimator_pip

# To run all Estimator tests
bazel test //tensorflow_estimator/...
```

To start training with Scavenger, please make the following additions:

```
run_config = tf.estimator.RunConfig(..., data_dir='*path to data dir*')

optimizer = *any TF optimizer*
optimizer = tf.train.ScavengerOptimizer(optimizer, replicas_to_aggregate=num_workers)
replicas_hook = optimizer.make_session_run_hook(params.is_chief, num_tokens=0)
train_hooks.append(replicas_hook)
```

TF_CONFIG json has an additional ```batch_size_list``` argument which is a comma separated list of per-worker batch-sizes. This parameter is updated in subsequent batchsize adjustments. The final TF_CONFIG for master in a 3 worker cluster looks like:
```
{"environment": "cloud", "batch_size_list": "[128,128,128]", "task": {"index": 0, "type": "master"}, "model_dir": "/data", "cluster": {"worker": ["172.17.0.4:8000", "172.17.0.5:8000", "172.17.0.6:8000"], "ps": ["172.17.0.2:8000"], "master": ["172.17.0.3:8000"]}
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
