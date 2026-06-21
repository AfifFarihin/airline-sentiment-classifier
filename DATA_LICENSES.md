# Data License and Privacy

The training data is the Kaggle **Twitter US Airline Sentiment** dataset, a
reformatted version of CrowdFlower's Data for Everyone release.

- Source: <https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment>
- Upstream license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0)
- License text: <https://creativecommons.org/licenses/by-nc-sa/4.0/>

The raw dataset is not redistributed in this repository. Run
scripts/download_data.py to retrieve the upstream archive and verify the
expected SHA-256 checksum. The file remains under its original license and is
excluded from Git.

The source contains public tweet text and fields such as usernames and
locations. The training loader retains only the fields needed for validation,
deduplication, chronology, labels, and modelling. Usernames are pseudonymized
in memory only to group uncertainty resamples and compare seen with unseen
authors; they are never model features or published. Published metrics and
figures contain no tweet text, usernames, locations, tweet IDs, or example
errors.

The MIT license applies only to original code and documentation in this
repository. It does not relicense the upstream dataset.
