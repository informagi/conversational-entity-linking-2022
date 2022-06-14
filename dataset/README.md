Conversational Entity Linking Dataset
=====================================

This directory contains conversational entity linking dataset.
The dataset consist of two parts: (1) Conversational Entity Linking Annotations and (2) Personal Entity Mention Annotations.

# Conversational Entity Linking Annotations

Contains personal entity linking annotations, where named entities, concepts, and personal entities are annotated. This can be used for training and evaluating entity linking for conversations.

Directory: `./Conversational_Entity_Linking_Annotations`

## Statistics

**Table 1: Statistics of conversational entity linking dataset**

|                                        |   Train |   Val |   Test |
|:---------------------------------------|--------:|------:|-------:|
| Conversations                          |     174 |    58 |     58 |
| User utterance                         |     800 |   267 |    260 |
| NE and concept annotations             |    1428 |   523 |    452 |
| Personal entity annotations            |     268 |    89 |     73 |

## Data Format
This section explains ground truth files data format (e.g., `./Conversational_Entity_Linking_Annotations/ConEL22_EL_Test.json`, etc.)\
Each element in a list has a dict structure as follows:

```py
{
    "dialogue_id": "9161",
    "turns": [
        {
            "speaker": "USER", # or "SYSTEM"
            "utterance": "Alpacas are definitely my favorite animal.  I have 10 on my Alpaca farm in Friday harbor island in Washington state.",
            "turn_number": 0,
            "el_annotations": [ # Ground truth annotations
                {
                    "mention": "Alpacas",
                    "entity": "Alpaca",
                    "span": [0, 7],
                }, ...]
            "personal_entity_annotations": [ # Personal entity annotations
                {
                    "personal_entity_mention": "my favorite animal",
                    "explicit_entity_mention": "Alpacas",
                    "entity": "Alpaca"
                }
            ],
            "personal_entity_annotations_without_eems": [ # Personal entity annotations where EEM annotated as not found
                {
                    "personal_entity_mention": "my Alpaca farm"
                }
            ]
        },
```


- `dialogue_id`: dialogue id provided by each original dataset (i.e., Wizard of Wikipedia). 
- `turns`: each element contains an user or system turns
  - `speaker`: USER or SYSTEM
  - `utterance`: utterance acquired from the dataset
  - `el_annotations`: annotations with MTurk workers
  - `personal_entity_annotations`: Personal entity annotations.
  - `personal_entity_annotations_without_eems`: Personal entity annotations where EEM annotated as not found.


# Personal Entity Mention Detection Annotations

Contains personal entity mention annotations. This can be used for training and evaluating personal entity mention detection 
  (e.g., detecting "my city", etc.).

Directory: `./Personal_Entity_Mention_Detection_Annotations`

## Statistics

**Table 2: Statistics of personal entity mention detection dataset**

|                                               |   Train |   Val |   Test |
|:-------------------------------------------   |--------:|------:|-------:|
| dialogues                                     |     591 |   197 |    197 |
| User utterances                               |    2689 |   905 |    907 |
| User utterances with personal entity mentions |     740 |   256 |    260 |
| Personal entity mentions                      |     803 |   286 |    280 |

## Data Format

The same as the previous section, except this only contain `personal_entity_mention` annotations.

