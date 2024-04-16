#!/bin/bash
REQUIRED_PYTHON_VERSION=3.11


# Function to convert string to green color
to_green() {
    echo -e "\e[32m$1\e[0m"
}

to_red() {
    echo -e "\e[31m$1\e[0m"
}  

is_pyenv_installed() {
    if command -v pyenv >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

is_correct_python_version_installed() {
    if pyenv versions --bare | grep -q "^$REQUIRED_PYTHON_VERSION"; then
        return 0
    else
        return 1
    fi
}

install_python() {
    to_green "Installing Python $REQUIRED_PYTHON_VERSION"
    pyenv install "$REQUIRED_PYTHON_VERSION" || { echo "pyenv install failed"; exit 1; }
    pyenv global "$REQUIRED_PYTHON_VERSION"
    to_green "Python $REQUIRED_PYTHON_VERSION is installed"
}

venv_exists() {
    if [ -d "venv" ]; then
        return 0
    else
        return 1
    fi
}

create_venv() {
    to_green "Creating venv"
    python3 -m venv venv
}

activate_venv() {
    to_green "Activating venv"
    source venv/bin/activate
}

install_requirements() {
    to_green "Installing requirements"
    pip install -r requirements.txt 
}

create_env_file() {
    if [ ! -f ".env" ]; then
        to_green "Creating .env file"
        touch .env
    else
        to_green ".env file already exists"
    fi
}

main() {

    if ! is_pyenv_installed; then
        to_red "pyenv is not installed. Please install pyenv"
    fi


    if ! is_correct_python_version_installed; then
        to_red "Python $REQUIRED_PYTHON_VERSION is not installed"
        install_python
    else
        to_green "Python $REQUIRED_PYTHON_VERSION is already installed"

        if ! venv_exists; then
        to_red "venv is not installed"
            create_venv
        fi

        activate_venv
        install_requirements
        create_env_file
    fi
}

main
