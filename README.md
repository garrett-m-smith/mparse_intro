# Mparse: A new framework for self-organized, incremental sentence comprehension

The framework is described in the paper under intro_paper/

## Setting up a model

To run mparse, you need to set up an instantiation of the model for each
sentence you want to test. The mparse model requires a number of inputs,
including the dependency grammar rules, the transition rate function, free
parameter values, and any link combinations to exclude. Details are given
below. To test multiple sentences, you set up a model for each one using the
same set of inputs, and the outputs will be comparable between sentences.

1. Grammar
The grammar is specified as a nested Python dictionary. The highest level of
the dictionary is indexed by head-dependency relations like `det-cat` for the
determiner attachment site on the word *cat*. The dependency type should
precede the head, and the two should be separated by a hyphen. Then, each
possible dependent is specified within a head-dependency relation
pair, e.g., *the* can be a determiner for `det-cat`. The dependent is also a
key of a dictionary, where the value is a list. The list contains the harmony
of that link (greater than zero) and the head direction preference. The head
direction preference is specified as -1 if the dependent precedes the head and
+1 if the dependent follows the head. Here's an example grammar that would be
appropriate for the sentence *the cat sleeps quietly*:

`gramm = {'det-cat': {'the': [1.0, -1]}, 'nsubj-sleeps': {'cat': [1.0, -1]}, 'adv-sleeps': {'quietly': [1.0, 1]}}`

