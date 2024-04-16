# 📦 TEMPLATE 📦


### Setup

If you are using a `macos` and you have `pyenv` installed:
```shell
source initialize.sh
```
Following step will be performed:

1. Checks if correct python version is installed.
    1. Install correct version if `pyenv` is installed.
    2. Set it as global python versiion
2. Create `venv`
    1. Activate it
    2. Install all `requirements.txt`
3. Create `.env`

--- 

### Code structure

```shell
.
├── .env
├── .gitignore
├── README.md
├── initialize.sh
├── main.py
├── requirements.txt
├── src
│   └── __init__.py
└── tests
    └── __init__.py
```

- **Functionalities** -> `src`
- **Tests**: -> `tests`
- **Necessary requirements**: `requirements.txt`

--- 

### Guidelines

1. Add new requirements to the `requirements.txt`. 📦
2. Put your code into `src`. `main.py` should be as clean as possible and only serve as the entry point of the application/tool. ✍️
3. Tests are run with `pytest`.