import unicodedata

GROUPS = {
    'three_models': ('R20', 'A', 'base'),
    'five_models': ('R20', 'A', 'base', 'hard_answer', 'hard_reasoning'),
}

def choose_answer(predictions, group):
    names = GROUPS[group]
    assert set(predictions) == set(names)
    assert all(value is None or isinstance(value, str) for value in predictions.values())
    votes = {}
    keys = {}
    for name in names:
        value = predictions[name]
        key = ' '.join(unicodedata.normalize('NFKC', value).casefold().split()) if value is not None else ''
        keys[name] = key or None
        if key:
            votes[key] = votes.get(key, 0) + 1
    selected = 'R20'
    if votes:
        highest = max(votes.values())
        winners = [key for key, count in votes.items() if count == highest]
        if highest >= 2 and len(winners) == 1 and keys['R20'] != winners[0]:
            selected = next(name for name in names if keys[name] == winners[0])
    return predictions[selected], selected
