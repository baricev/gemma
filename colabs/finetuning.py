#!/usr/bin/env python
# coding: utf-8

# # Finetuning
# 
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/finetuning.ipynb)
# 
# This is an example on fine-tuning Gemma. For an example on how to run a pre-trained Gemma model, see the [sampling](https://gemma-llm.readthedocs.io/en/latest/sampling.html) tutorial.
# 
# To fine-tune Gemma, we use the [kauldron](https://kauldron.readthedocs.io/en/latest/) library which abstract most of the boilerplate (checkpoint management, training loop, evaluation, metric reporting, sharding,...).
# 

# In[ ]:


!pip install -q gemma

# In[21]:


# Common imports
import os
import optax
import treescope

# Gemma imports
from kauldron import kd
from gemma import gm

# By default, Jax do not utilize the full GPU memory, but this can be overwritten. See [GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html):

# In[ ]:


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

# ## Data pipeline
# 
# First create the tokenizer, as it's required in the data pipeline.

# In[22]:


tokenizer = gm.text.Gemma3Tokenizer()

tokenizer.encode('This is an example sentence', add_bos=True)

# First we need a data pipeline. Multiple pipelines are supported including:
# 
# * [HuggingFace](https://kauldron.readthedocs.io/en/latest/api/kd/data/py/HuggingFace.html)
# * [TFDS](https://kauldron.readthedocs.io/en/latest/api/kd/data/py/Tfds.html)
# * [Json](https://kauldron.readthedocs.io/en/latest/api/kd/data/py/Json.html)
# * ...
# 
# It's quite simple to add your own data, or to create mixtures from multiple sources. See the [pipeline documentation](https://kauldron.readthedocs.io/en/latest/data_py.html).
# 
# We use `transforms` to customize the data pipeline, this includes:
# 
# * Tokenizing the inputs (with `gm.data.Tokenize`)
# * Creating the model inputs (with `gm.data.Tokenize`))
# * Adding padding (with `gm.data.Pad`) (required to batch inputs with different lengths)
# 
# Note that in practice, you can combine multiple transforms into a higher level transform. See the `gm.data.ContrastiveTask()` transform in the [DPO example](https://github.com/google-deepmind/gemma/tree/main/examples/dpo.py) for an example.
# 
# Here, we try [mtnt](https://www.tensorflow.org/datasets/catalog/mtnt), a small translation dataset. The dataset structure is `{'src': ..., 'dst': ...}`.

# In[27]:


ds = kd.data.py.Tfds(
    name='mtnt/en-fr',
    split='train',
    shuffle=True,
    batch_size=8,
    transforms=[
        # Create the model inputs/targets/loss_mask.
        gm.data.Seq2SeqTask(
            # Select which field from the dataset to use.
            # https://www.tensorflow.org/datasets/catalog/mtnt
            in_prompt='src',
            in_response='dst',
            # Output batch is {'input': ..., 'target': ..., 'loss_mask': ...}
            out_input='input',
            out_target='target',
            out_target_mask='loss_mask',
            tokenizer=tokenizer,
            # Padding parameters
            max_length=200,
            truncate=True,
        ),
    ],
)

ex = ds[0]

treescope.show(ex)

# We can decode an example from the batch to inspect the model input. We see that the `<start_of_turn>` / `<end_of_turn>` where correctly added to follow Gemma dialog format.

# In[28]:


text = tokenizer.decode(ex['input'][0])

print(text)

# ## Trainer
# 
# The [kauldron](https://kauldron.readthedocs.io/en/latest/) trainer allow to train Gemma simply by providing a dataset, a model, a loss and an optimizer.
# 
# Dataset, model and losses are connected together through a `key` strings system. For more information, see the [key documentation](https://kauldron.readthedocs.io/en/latest/intro.html#keys-and-context).
# 
# Each key starts by a registered prefix. Common prefixes includes:
# 
# * `batch`: The output of the dataset (after all transformations). Here our batch is `{'input': ..., 'loss_mask': ..., 'target': ...}`
# * `preds`: The output of the model. For Gemma models, this is `gm.nn.Output(logits=..., cache=...)`
# * `params`: Model parameters (can be used to add a weight decay loss, or monitor the params norm in metrics)
# 
# 
# 
# 
# 

# In[ ]:


model = gm.nn.Gemma3_4B(
    tokens="batch.input",
)

# In[ ]:


loss = kd.losses.SoftmaxCrossEntropyWithIntLabels(
    logits="preds.logits",
    labels="batch.target",
    mask="batch.loss_mask",
)

# We then create the trainer:

# In[ ]:


trainer = kd.train.Trainer(
    seed=42,  # The seed of enlightenment
    workdir='/tmp/ckpts',  # TODO(epot): Make the workdir optional by default
    # Dataset
    train_ds=ds,
    # Model
    model=model,
    init_transform=gm.ckpts.LoadCheckpoint(  # Load the weights from the pretrained checkpoint
        path=gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
    ),
    # Training parameters
    num_train_steps=300,
    train_losses={"loss": loss},
    optimizer=optax.adafactor(learning_rate=1e-3),
)

# Trainning can be launched with the `.train()` method.
# 
# Note that the trainer like the model are immutables, so it does not store the state nor params. Instead the state containing the trained parameters is returned.

# In[ ]:


state, aux = trainer.train()

# ## Checkpointing

# To save the model params, you can either:
# 
# * Activate checkpointing in the trainer by adding:
# 
#   ```python
#   trainer = kd.train.Trainer(
#       workdir='/tmp/my_experiment/',
#       checkpointer=kd.ckpts.Checkpointer(
#           save_interval_steps=500,
#       ),
#       ...
#   )
#   ```
# 
#   This will also save the optimizer, step, dataset state,...
# 
# 
# * Manually save the trained params:
# 
#   ```python
#   gm.ckpts.save_params(state.params, '/tmp/my_ckpt/')
#   ```

# ## Evaluation
# 
# Here, we only perform a qualitative evaluation by sampling a prompt.
# 
# For more info on evals:
# 
# * See the [sampling](https://gemma-llm.readthedocs.io/en/latest/sampling.html) tutorial for more info on running inference.
# * To add evals during training, see the Kauldron [evaluator](https://kauldron.readthedocs.io/en/latest/eval.html) documentation.
# 

# In[ ]:


sampler = gm.text.ChatSampler(
    model=model,
    params=state.params,
    tokenizer=tokenizer,
)

# We test a sentence, using the same formatting used during fine-tuning:

# In[ ]:


sampler.chat('Hello! My next holidays are in Paris.')

# The model correctly translated our prompt to French!
# 
# ## Next steps
# 
# To fine-tune outside of Colab, you can look at our [examples/](https://github.com/google-deepmind/gemma/tree/main/examples/) folder for more complexes trainer configs, including LoRA, DPO and sharding.
