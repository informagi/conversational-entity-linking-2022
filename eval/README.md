Evaluation
==========

## Evaluate your method

First, annotate conversations with your method
- The dataset for Val and Test are available [here](https://github.com/informagi/conversational-entity-linking-2022/tree/main/dataset/Conversational_Entity_Linking_Annotations)
- The ConEL21-PE dataset are available [here](https://github.com/informagi/conversational-entity-linking)

Second, create run files. The data format should be like:

```py
[
    [
        "10060", # Wizard-of-Wikipedia document ID
        0, # Turn number
        "my favorite color", # Mention
        "Blue" # Entity
    ], ...
]
```

Finally, evaluate use `eval.ipynb`. The detailed instructions are in the notebook.