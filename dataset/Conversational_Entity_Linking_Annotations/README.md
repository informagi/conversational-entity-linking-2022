Conversational Entity Linking Annotations
============================


## Statistics

**General statistics**
|                                        |   Train |   Val |   Test |
|:---------------------------------------|--------:|------:|-------:|
| dialogue                               |     174 |    58 |     58 |
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

(*1) This annotations also contain PEM without corresponding EEM case (i.e., annotated as `not in dialogue`, cannot find from options, etc.)

## Data Format
This section explains ground truth files data format (`ConEL_CNE.json` and `ConEL_PE.json`)\
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