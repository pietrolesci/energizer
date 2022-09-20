# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at [https://github.com/pietrolesci/energizer/issues](https://github.com/pietrolesci/energizer/issues).

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

Pytorch-Energizer could always use more documentation, whether as part of the
official Pytorch-Energizer docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/pietrolesci/energizer/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Set up local development environment!

Ready to contribute? Here's how to set up `energizer` for local development.

1. Fork the `energizer` repo on GitHub.

1. Clone your fork locally

    ```bash
    git clone git@github.com:your_name_here/energizer.git
    ```

1. Ensure [conda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) is installed, otherwise install it

    ```bash
    LINK_TO_CONDA_INSTALLER=  #(1)
    wget $LINK_TO_CONDA_INSTALLER
    #(2)
    ```

    1. Check [here](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) for a suitable version of the conda installer, copy the link, and paste it here to download the file.

    2. Run the installer.

1. Create a new conda environment
    ```bash
    CONDA_ENV_NAME=  #(1)
    conda create -n $CONDA_ENV_NAME python=3.9 -y
    conda activate $CONDA_ENV_NAME
    ```

    1. Put here the name of the conda environment.

1. Ensure [poetry](https://python-poetry.org/docs/) is installed, otherwise install it

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

1. Install dependencies and start your virtualenv:

    ```bash
    poetry install --all-extras --sync
    ```

1. Create a branch for local development:

    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

1. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox:

    ```bash
    poetry run tox
    ```

1. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

1. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.7, 3.8, 3.9, 3.10. Check
   https://github.com/pietrolesci/energizer/actions
   and make sure that the tests pass for all supported Python versions.

## Tips

```bash
poetry run pytest tests/some_test_file.py
```

To run a subset of tests.


## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in CHANGELOG.md).
Then run:

```bash
poetry run bump2version patch  #(1)
git push
git push --tags
```

1. Possible values: `major` / `minor` / `patch`.

GitHub Actions will then deploy to PyPI if tests pass.
