### Structure
```
data_dict
┣ brain_mask: Q domain training
┣ brain_mask_test: Q domain testing
┣ brain_train: original paper Q domain training data
┣ brain_val: original paper Q domain testing data
┣ knee_mask: P domain training
┣ knee_train: original paper P domain training data
┣ knee_val: original paper P domain testing data
┣ TTT_brain_test.yaml: Q domain testing
┣ TTT_brain_train_300.yaml: Q domain training
┣ TTT_knee_test.yaml: P domain testing
┗ TTT_knee_train_300.yaml: P domain training
```

Since the training set data provided by the [original paper repository](https://github.com/MLI-lab/ttt_for_deep_learning_cs) are not of the same size, we deleted some data so that the training set for both knee and brain have 300 data. 