![https://raw.githubusercontent.com/RameshAditya/scoper/master/github-resources/logo.jpg?token=AECQW6LFNMRDVEGLBMMVKFK5OH7AI](https://raw.githubusercontent.com/RameshAditya/scoper/master/github-resources/logo.jpg?token=AECQW6LFNMRDVEGLBMMVKFK5OH7AI)
--------------------------------------

## Fuzzy and Semantic Caption-Based Searching for YouTube Videos 

![https://github.com/RameshAditya/scoper/blob/master/github-resources/demo_fuzzy.gif](https://github.com/RameshAditya/scoper/blob/master/github-resources/demo_fuzzy.gif)

---------------------------------------------------------------------------------------------------------
## Contents
- [What Scoper is](#what-scoper-is)
- [How Scoper works](#how-scoper-works)
- [How to use Scoper](#how-to-use-scoper)
- [Future plans](#future-plans)
- [Support me](#support-me)

---------------------------------------------------------------------------------------------------------
## What Scoper is
Scoper is a python script that takes a youtube URL and a user query string as inputs, and returns the timestamps in the video where the content of the caption closely matches the user's query string.

For example, in the video - [https://www.youtube.com/watch?v=bfHEnw6Rm-4](https://www.youtube.com/watch?v=bfHEnw6Rm-4) - which is Apple's October 2018 event, if you were to query `Photoshop for ipad`, you'd see the following output -

```
photoshop on ipad.                   1h 6m 29s
for.                                 54m 16s
ipad.                                50m 37s
photoshop.                           1h 14m 8s
this is a historic center for        3m 48s
would love to play it for you        4m 50s
pro users but designed for all       7m 52s
exactly what you're looking for,     8m 0s
go and use for everything they       8m 52s
product line for years to come,      9m 29s

```

---------------------------------------------------------------------------------------------------------
## How Scoper works
Scoper works in two ways. 

- Extract captions and timestamps from the YouTube URL
- Preprocess the user query and train a Word2Vec model
- Query over the captions and find the best match. This is done in two ways, as decided by the user -
  - Fuzzy searching
    - Scoper enables you to query over the video's captions by using fuzzy matching algorithms.
    - This means it searches for the most relevant captions in terms of spelling and finds the nearest match.
    - Done by using variants of Levenshtein's distance algorithms.
    - Supports multiple languages.
  
  - Semantic searching
    - Scoper also enables you to query over the video's captions using semantic sentence similarity algorithms.
    - The performance of semantic searching is highly dependent on the dataset on which the Word2Vec model used is trained on.
    - By default, the Brown's corpus is used to train the Word2Vec model, and additionally a modified word-mover's distance algorithm is used to evaluate sentence-sentence similarity.
    - For non-english language querying, the user will have to provide their own dataset.
- Map back the chosen captions to the original timestamps and return them

---------------------------------------------------------------------------------------------------------
## How to use Scoper

Shell usage
```python
>>> obj = scoper()
>>> obj.main("Apple Watch", 'https://www.youtube.com/watch?v=wFTmQ27S7OQ', mode = 'FUZZY', limit = 10) 
[('Apple Watch.', 1796.994), ('the iPad to the Apple watch, and', 318.617), ('Apple Watch has grown ... ]
```

CLI usage
```
> python -W ignore scoper.py --video https://www.youtube.com/watch?v=bfHEnw6Rm-4 --mode FUZZY --limit 10 --language en
Enter query string: prjct airo

air.                                 9m 0s
project aero, our new augmented      1h 6m 7s
well, with project aero, now you     1h 9m 54s
we also showed you project aero,     1h 11m 28s
pro.                                 49m 43s
ipad pro and it protects both        57m 15s
tap.                                 59m 52s
so now with photoshop, project       1h 10m 41s
products, every ipad pro is made     1h 15m 41s
previous air.                        18m 13s
```

---------------------------------------------------------------------------------------------------------
## Future Plans
- Improve the sentence similarity algorithm
- Include out-of-the-box support for use of pretrained word embeddings
- Include support for general audio searching using SpeechRecognition APIs to generate a corpus from non-captioned audios

---------------------------------------------------------------------------------------------------------
## Support Me
If you liked this, leave a star! :star:

If you liked this and also liked my other work, be sure to follow me for more! :slightly_smiling_face:
