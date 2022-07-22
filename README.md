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
