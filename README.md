#Research goal

To verify ability of transfomer to handle CNN responses as input instead of word embeddings.

##Initial setup:  
a) muliple cnn-classifiers traned differently,
b) dataset with images

##Transformer to be trained
sequence: image__i -> [cnn_resp1, cnn_resp2, cnn_resp3, ...]

masked_sequence = [cnn_resp1, ...,<MASK>, cnn_respk, ...]
restored_sequence = Transformer(masked_sequence)

##Objective
loss = disssimilarity(restored_sequence, sequence)


