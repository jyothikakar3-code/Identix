# Animal guard evaluation set

Place licensed, non-private images into these folders before production calibration:

- `human/`: selfies, groups, profiles, babies, masks, low light, and blur
- `animal/`: dogs, cats, horses, cows, monkeys, wildlife, birds, rabbits,
  foxes, wolves, bears, elephants, goats, sheep, deer, cartoons, and toys
- `invalid/`: cars, fruit, houses, blank images, and corrupt/unsupported images
- `mixed/`: at least one human and at least one animal

Run:

```bash
python3 scripts/evaluate_animal_guard.py tests/fixtures/animal_guard
```

The command fails if an animal is accepted as human or if overall labelled-set
accuracy is below 95%. Images are intentionally not committed to protect privacy
and avoid redistributing copyrighted test data.
