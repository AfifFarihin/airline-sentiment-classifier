# Model Card

## Intended Use

The model demonstrates reproducible multiclass text classification and
time-aware evaluation. It is suitable for learning, portfolio review, and
offline analysis of similarly structured English airline feedback.

It is not intended for automated customer decisions, employee evaluation,
safety decisions, or production deployment without current data and additional
validation.

## Data

The source contains 2015 U.S. airline tweets labelled as negative, neutral, or
positive. Exact duplicate IDs and duplicate text are removed. Usernames and
URLs are replaced during text normalization, and direct identifier columns are
not used as model features.

## Evaluation

The latest 20% of deduplicated rows by timestamp are reserved as a final
holdout. Model selection uses five-fold cross-validation only on the
earlier training period. The folds use expanding windows rather than random
shuffling. Macro F1 is the primary selection metric because the negative class
is substantially larger than the neutral and positive classes. Only the
selected model is evaluated on the final holdout.

The selected class-balanced logistic regression achieved 0.726 macro F1
(author-grouped 95% interval 0.707-0.744) and 0.775 accuracy (0.759-0.791) on
2,886 later tweets. Negative, neutral, and positive class F1 scores were 0.854,
0.645, and 0.679 respectively.

The holdout contains 504 messages from authors also observed in training and
2,382 from previously unseen authors. Macro F1 was 0.720 and 0.727 respectively,
suggesting that the headline result is not dependent on repeat authors.

## Limitations

- The dataset covers approximately one week in February 2015.
- Labels and language may not represent current airline customers.
- A chronological split tests later messages in the same collection, not a
  different platform, year, or domain.
- Public social-media text can contain annotation errors, sarcasm, dialect, and
  demographic bias.
- Feature coefficients describe model associations and are not causal.

## Version

Model evidence was regenerated locally on 21 June 2026 using Python 3.11 and
the dependency versions recorded in uv.lock.
