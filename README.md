# Description

[In-Progress]. The project combined the impatient reader model in [DeepMind Teaching Machines to Read and Comprehend](https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend) and [LSTM-based Deep Learning Models for Non-factoid Answer Selection](http://arxiv.org/abs/1511.04108) in [tensorflow](https://github.com/tensorflow/tensorflow).

The training set was chosen from TOEFL, consists of: context (up to 6000 words), question, and four options. Context is splitted into sentences, embedded into a vector with the question using an impatient reader. The embedded vector and options are later calculated using cosine similarity. The target is to minimize cosine cost.

```
python3 train.py
```

# Design

![Algorithm](tf_algorithm.png)

# References

- [DeepMind: Teaching Machines to Read and Comprehend](https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend)
- [Teaching Machines to Read and Comprehend (blog)](http://rsarxiv.github.io/2016/06/13/Teaching-Machines-to-Read-and-Comprehend-PaperWeekly/)
- [Implementing A Cnn for Text Classfication in Tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Deep Learning for Answer Sentence Selection](http://ttic.uchicago.edu/~haotang/speech/1412.1632v1.pdf) (*)
- [LSTM-Based Deep Learning Models For Nonfactoid Answer Selection](https://arxiv.org/pdf/1511.04108v4.pdf) (*)
- [Official Tensorflow Tutorials](https://www.tensorflow.org/versions/r0.12/tutorials/index.html)
- [Tensorflow Doc(中文)](http://wiki.jikexueyuan.com/project/tensorflow-zh/get_started/basic_usage.html)
