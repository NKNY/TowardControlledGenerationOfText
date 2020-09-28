### 28.09.20
Model was pretrained for 10K steps and trained for 10K steps. Upon querying the model with `print_sampled_sentence`
it is clear that the model has learned that EOS is followed by all UNKs. SOS is actually missing from the sampled
sentences but it's does not affect the remaining tokens. The remaining tokens, irrespective of the content, generally
form a only few combinations of words, seemingly distinct for different types of one hot style vector. The temperature
after 10K steps is <1e-5. This obviously could affect sampling because it effectively makes the generation deterministic.
Setting temperature to 1. leads to a wider variety of sentences, which whilst still follow the {SOS}{TOKEN}\*{EOS}{UNK}\*
structure, are otherwise gibberish. From tensorboard graphs it appears that all losses are going down with the exception
of entropy loss. **It is crucial I understand why it would be the case**. Furthermore, it would be useful to **train
and evaluate the model using only a tiny amount of data** e.g. corpus built from ~10 sentences. Finally, the true scale
of how good/bad the implementation is would be aided by **implementing the evaluation metrics/step.**

Subsequent reading identified the entropy term just as a form of regularisation loss pushing the model to generate
one-hot style class predictions. If implementing multiple labels being possible then this regularisation should 
potentially be avoided.