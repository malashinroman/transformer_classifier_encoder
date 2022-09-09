# Research goal

Verify ability of transfomer to handle CNN responses instead of word embeddings.\

## Initial setup

a) muliple cnn-classifiers trained differently,\
b) dataset with images X

## Transformer to be trained

for x in X: s = [cnn1(x), cnn2(x), ...] \
|----- masked_sequence = [y1, ...,MASK, yk, ...]\
|----- restored_sequence = Transformer(masked_sequence)

## Objective

loss = disssimilarity(restored_sequence, sequence)


## Results

| base model   | input type       | zeroout prob | val acc. |
| :-:          | :-:              |          :-: |      :-: |
| bert-uncased | logits           |          0.8 |    65.85 |
| bert-uncased | classifier index |          0.8 |    46.59 |


| shown classifiers | reward |
|               :-: |    :-: |
|                 1 |   5.35 |
|                 2 |   5.57 |
|                 3 |   5.83 |
|                 4 |   6.10 |
|                 5 |   6.38 |
|                 6 |   6.67 |
|                 7 |   6.99 |
|                 8 |   7.31 |
|                 9 |   7.62 |
