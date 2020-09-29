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

Following some bugfixes, running training only on 10 first sentences appears to be able to generally capture some
patterns but the same sentence can be recreated fully and then 100 steps of size 32 later only 2 words of the sentence
are output. Moreover, following 5K pretraining steps sentences still end with multiple copies of EOS. These problems
are seen before we removed temperature from discriminator training. However this should not matter for pretraining
because discriminator is not used at that stage. Additional 3K training steps for some reason resulted the same
sentence more or less being printed every time the model was queried for a sentence (every 100 steps). Seems like there
is something wrong with checkpoints as restarting from a model with 3+5k steps led to very different sampled sentences
than at 3+4.9K. Moreover the sentence is effectively always the same.

Temperature annealing appears valid, although there may be a need to confirm that not capping it at a certain value
does not make softmax explode.

### 29.09.20
Following the observation that the model is acting weird following a restart, the first thing to do is a local test
to check the model weights before and after the restart? Appears that embeddings are not being trained following
pretraining. Other weights are trained fine. Restarting from a checkpoint after 40+20 steps takes the models to have
the same weights as they were at the end of 40+20 steps. Unclear what the issue then is. In fact, restarting the model
leads to substantially higher disc/gen/enc losses for the first few iterations. So something must being going wrong...

As previously desribed another, issue appears to be the fact that at least the pretraining portion does not manage to 
learn that EOS is only followed by padding and that padding is only followed by padding. Therefore will check the 
samples that are being fed into the model during pretraining to see if it has anything to do with inability to learn 
this pattern.

It seems that the issue of sentences not ending with EOS is fixed by a combination of removing masking from padded step 
loss and including them in the total loss. Furthermore, encoder was previously ignoring any masking and outputting the
outputs from the end of the sequence, which was likely associated with padding. Changed this to return the output from
the last unpadded step (EOS). Also fixed incorrect use of mask for the generated samples. 
