# Skeleton DL project
Toy example to explain how to build a DL project, from model development to deployment.

Note: this is **NOT** production ready, some extra changes would be needed

## Setup
1. Install `virtualenvwrapper` to easily manage python virtual environments. You can use the vanilla `virtualenv` or 
`condaenv` if you are more familiar with it.
To install `virtualenvwrapper` follow the [installation guide](https://virtualenvwrapper.readthedocs.io/en/latest/install.html).

2. Create a new virtual environment (give it a name like `skeleton`):

        mkvirtualenv skeleton
    
3. Install `pip-tools`, one of the best packages to deal with project dependencies:

        pip install pip-tools
    
4. You can now syncronize the dependencies in `requirements.txt` with you venv with:

        pip-sync requirements.txt
    
5. To develop this project you'll also need it installed. Best way to do so is by installing it in editable mode:

        pip install -e .
    
## Executables
In the folder `bin/` you can find some runnable scripts to perform different actions. To be able to run them you will
need to have the previous virtual environment active. To do so (if it is not already):

    workon skeleton
    
After that you can easily run the scripts:

- `run_pipeline.py`: Run the input pipeline only to improve its performance
- `train.py`: Train the model