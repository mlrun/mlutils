# mlutils
reusable code components for working with MLRun functions

## modules
----------

### data

* **`get_sample`**<br>
generate data sample to be split

* **`get_splits`**<br>
generate train and test sets

* **`save_heldout`**<br>
save a dict of datasets

### models

* **`get_class_fit`**<br>
generate a model config

* **`create_class`**<br>
Create a class from a package.module.class string

* **`create_function`**<br>
Create a function from a package.module.function string

* **`gen_sklearn_model`**<br>
generate an sklearn model configuration

* **`eval_class_model`**<br>
generate predictions and validation stats

### plots

* **`gcf_clear`**<br>
call this to clear matplotlib's figure and axes before generating a new plot
* **`plot_importance`**<br>
custom plot for models that have a `feature_importances` attribute or where
one can be generated.

