# Description

The project consists of two models. The impatient reader model in ref[1] comprehends very long articles and a related question. A cosine similarity cost model from ref[5] to choose best answer from four options. I implemented the model in [TensorFlow](https://github.com/tensorflow/tensorflow).

The training set consists of: a context, a question, and four multiple choices, with only one of them is true. Context was splitted into sentences, embedded into a vector with the question. The embedded vector and choices are later calculated using cosine similarity. The non-supervisual target is to minimize cosine cost.

Instead of simply solving [cloze-like questions](https://en.wikipedia.org/wiki/Cloze_test), the model can comprehend both article and the related question, and choose the best answer from a list of candidates.

```shell
python3 train.py
```
# Train Set

```
{
    "answer": 3, 
    "question": "The Basilica of the Sacred heart at Notre Dame is beside to which structure?", 
    "answer_list": [
        "a copper statue of Christ", 
        "a golden statue of the Virgin Mary", 
        "Saint Bernadette Soubirous", 
        "the Main Building"
    ], 
    "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."
}
```

# Similar Works

- [AllienAI](https://github.com/tambetm/allenAI), the problem they solved did not have a context.
- [Attentive Reader(tensorflow)](https://github.com/carpedm20/attentive-reader-tensorflow), the problem they solved was cloze-like, the project is deprecated.

# References

1. [DeepMind: Teaching Machines to Read and Comprehend](https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend) (*)
2. [Teaching Machines to Read and Comprehend (blog)](http://rsarxiv.github.io/2016/06/13/Teaching-Machines-to-Read-and-Comprehend-PaperWeekly/)
3. [Implementing A Cnn for Text Classfication in Tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
4. [Deep Learning for Answer Sentence Selection](http://ttic.uchicago.edu/~haotang/speech/1412.1632v1.pdf) (*)
5. [LSTM-Based Deep Learning Models For Nonfactoid Answer Selection](https://arxiv.org/pdf/1511.04108v4.pdf) (*)
6. [Official Tensorflow Tutorials](https://www.tensorflow.org/versions/r0.12/tutorials/index.html)
7. [Tensorflow Doc(中文)](http://wiki.jikexueyuan.com/project/tensorflow-zh/get_started/basic_usage.html)

# Design

![Algorithm](tf_algorithm.png)
