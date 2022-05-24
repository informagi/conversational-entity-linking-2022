Conversational Entity Linking Dataset
=====================================

This directory contains conversational entity linking dataset.
The dataset consist of two parts: (1) Conversational Entity Linking Annotations and (2) Personal Entity Mention Annotations.

# Conversational Entity Linking Annotations

Contains personal entity linking annotations, where named entities, concepts, and personal entities are annotated. This can be used for training and evaluating entity linking for conversations.

Directory: `./Conversational_Entity_Linking_Annotations`

## Statistics

**General statistics**
|                                        |   Train |   Val |   Test |
|:---------------------------------------|--------:|------:|-------:|
| conversation                           |     174 |    58 |     58 |
| user utterance                         |     800 |   267 |    260 |
| user utterance with entity annotations |     685 |   229 |    210 |
| entity annotation                      |    1428 |   523 |    452 |
| user utterance with PEM                |     190 |    64 |     59 |

**PEM related statistics**
|                                        |   Train |   Val |   Test |
|:---------------------------------------|--------:|------:|-------:|
| PEM w/ single-EEM                      |     186 |    63 |     57 |
| PEM w/ multiple-EEM                    |       5 |     1 |      3 |
| PEM w/o EEM(*1)                        |      77 |    25 |     13 |
| PEM total                              |     268 |    89 |     73 |

(PEM: Personal Entity Mention, EEM: Explicit Entity Mention)  
(*1) This annotations also contain PEM without corresponding EEM case (i.e., annotated as `not in dialogue`, cannot find from options, etc.)

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

|                                            |   Train |   Val |   Test |
|:-------------------------------------------|--------:|------:|-------:|
| dialogues                                  |     591 |   197 |    197 |
| dialogues with personal entity annotations*|     587 |   195 |    195 |
| user utterances                            |    2689 |   905 |    907 |
| user utterances with PEM                   |     740 |   256 |    260 |
| PEM                                        |     803 |   286 |    280 |

*For some HITs, turkers agreed on "none" option, representing the given spans are not PEM (e.g., "Oh my god"), thus, this number is smaller than dialogues.

## Data Format

The same as the previous section, except this only contain `personal_entity_mention` annotations.

