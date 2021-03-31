# Mparse: A new framework for self-organized, incremental sentence comprehension

The framework is described in the paper under intro_paper/

## Setting up a model

To run mparse, you need to set up an instantiation of the model for each
sentence you want to test. The mparse model requires a number of inputs,
including the dependency grammar rules, the transition rate function, free
parameter values, and any link combinations to exclude. Details are given
below. To test multiple sentences, you set up a model for each one using the
same set of inputs, and the outputs will be comparable between sentences.

1. **Grammar:**
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

2. **Free parameters:**
Mparse has two free parameters, $\tau$ and $T$. $\tau$ is a scaling parameter
that determines the rate at which the model explores its state space. It is
given in units of the number of times the model could explore its whole state
space per unit time. To get mean processing times that are roughly on the scale
of what is typical in reading studies with native English speakers, use a
$\tau$ of about 1.25. This means that the model on average can explore its state
space about 1.25 times per second; the average processing time will come out to
around 0.4s per word (roughly, with large variation). The parameter $T$
controls the noise. With very low noise ($T < 2$), the model very strongly
prefers to jump from low harmony states to higher harmony states. In this
range, it can be difficult for the model to reanalyze the sentence by deleting
links, which requires jumping from a higher-harmony state to a lower harmony
state. $T$ also affects the effect sizes, i.e., differences in processing times
between experimental condtitions. However, for $T > 10$ or so, the effect size
differences reach their asymptotic levels and do not vary very much any more.

3. **Transition rate function:**
This function determines how the transition rates between states are
calculated. I recommend either the exponential/Arrhenius function (transition
rate decreases exponentially as the difference in harmonies between the states
increases) or the Glauber/sigmoidal function. The other implemented option is
the Metropolis transition rate function, but this one does not allow for
graded differences in harmony to affect the transition rates; if the new state
has a higher harmony, make the transition regardless of how much better it is.
In the submitted paper, I used the exponential function because it seems to be
the most assumption-free choice. The transition function choice is set using
the argument `method='arrhenius'`, for example.

4. *Link combinations to exclude:*
Mparse generates parse states via a brute-force enumeration procedure. If a
link is in the grammar and the words involved in that link are in the input so
far, mparse assumes it can use that link in its states. This can sometimes
over-generate, though, and create states that are too far from what is commonly
assumed about what humans consider while parsing. This can (but doesn't have to
be) specified using the `incompat=` argument. For example, in the locally
coherent string *...smiled at the player tossed the frisbee...*, we do not want
to include the links `nsubj(tossed, player)` and `nsubjpass(tossed, player)` in
the same state. This amounts to enforcing the rule that *tossed* cannot
simultaneously act as a main verb (which needs a nominal subject `nsubj`) and a
passivized participle needing a subject (`nsubjpass`). The argument `incompat`
is specified as a list of lists, where each sub-list is a pair of
head-dependency type couples that are not allowed to be included in the same
state, e.g., `incompat = [['nsubj-tossed', 'nsubjpass-tossed']]`.

## Running a sentence
Once you have specified the inputs to mparse, you can have it process a
sentence using the function `run_sent`. It takes a number of arguments:
- `words`: a list of words, e.g., `'the cat sleeps quietly'.split()`
- `grammar`: a grammar specified as above
- `incompat`: a list of pairs of incompatible links, specified as above
- `tau` and `T`: the free parameters
- `method`: the transition rate function, either `arrhenius`, `glauber`, or
  `metropolis`
- `plot_trans`: True or False (default), whether to plot the transition matrix
  for each word in the sentence. Can be useful for making sure the model is set
      up correctly.
- `plot_fptd`: True or False (default), whether to plot the first passage time
  distribution, the distribution of predicted reading times for each word.
- `check_fptd`: Do a sanity check to make sure the first passage time
  distribution is properly normalized
- `exit_only`: if True and if `plot_fptd` is True and there is more than one
  absorbing state, plot the distribution of predicted processing times to reach
  *any* absorbing state instead of separate distributions for each absorbing
  state.
- `log_axis`: True or False (default), whether plot first-passage times or exit
  time distributions on a logarithmic scale
- `verbose`: True (default) or False, whether to print mean and variance of
  predicted reading times for each word as it's being processed

There is also a function for exploring different parameter settings, called
`explore_params`. It takes the arguments `words, grammar, incompat` as above,
as well as a list of $\tau$ values and a list of $T$ values to test, and
finally the transition rate function, again specified with `method`. This
function returns a data frame (using the Pandas package) with information about
the processing time predictions for each combination of the parameters.

Simple examples of these functions are given at the bottom of the mparse.py
Python script in the directory mparse. 
