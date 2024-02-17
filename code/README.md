# Requirements (update as needed)

See in the `requirements.txt` file.

If you are using a virtual environment, simply do:

```
pip install -r code/requirements.txt
```

Deprecated?

1. numpy
2. tqdm
3. black (for formatting)

# Running
```
python -m code.main --config_path=./code/configs/sarsa_tilecoding.json
```

# Testing
```
python -m unittest code/tests/*.py
```