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

The above changes appear to have allowed the model to memorise all the samples correctly, including everything that
happens after EOS. However it is also observed that different class labels still lead to the generation of the same
sentence - the model may not be learning to map the sentence to the class - might need to increase the weight.
Furthermore, still, restarting the training leads to samples in the first few steps of the second round of training
that are unlike those seen at the end of the first round.

### 01.09.20
Fixed a bug whereby the vocabulary included words that were only present in sentences that were later filtered out
due to their length. Removed max_unpadded_timesteps. Default sentence length is now 17 (15+SOS+EOS).
Added a script to generate the same data used by Hu et al (as opposed to the glue version of SST-2) and a flag to choose
which data to consume during training.

Made it so that separate parts of the model get the True `training` flag only when they are being trained e.g. 
the discriminator doesn't apply dropout when we're training the generator. This may potentially stabilise the training
procedure following pretraining. May wish to separately see whether just setting dropout to 0 benefits the model
training.

Examined the classifier from Hu et al. 2016a that Hu et al. 2017 uses to measure validation accuracy. Turns out it's
a separate classifier leveraging knowledge distillation. Hu provides a big uncommented wall of code implementation in 
theano. The options are to either try to minimally adapt the code to be able to run it on generated samples in theano,
recreate it with tensorflow or find an alternative classifier to measure the performance. 

For hidden state init decided that theoretically there is not much difference between training and not training the
initial hidden state.

### 02.09.20
Ran 10+20K w/o regularisation. Most losses were still going down at 20K steps but generator style and content losses
are fluctuating/going up over time. It appears the most logical to fix the issues with restarting from a saved
checkpoing and then train the model with a very low number of datapoints (possibly reduce the dimensionality) to see
if the model is able to reconstruct correct samples given a class.

### 05.09.20
Restarting from 1+0K checkpoint showed that the restarted model is generally able to maintain the pattern of having SOS
and EOS at the start and the end of the sentence. Additional inspection of the dataset initialisation identified that
instantiating the dataset in a different runtime leads to different token2idx mappings as well as even different
vocabularies. This is due to the use of sets. Fixed this. 

Furthermore, made all inputs to the model lowercase.


TODO: Address the issues from the last paragraph of 29.09