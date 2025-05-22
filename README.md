# Code to reproduce the experiments from our paper ``The Origins of Representation Manifolds in Large Language Models''

This repository contains code necessary to reproduce the figures in our paper. It is made up of three parts:

**Part 1:** Processing token activations with SAE (following Engels et al. (2025)).

Here, we follow Engel's et al (2025) (and the code in [the accompanying Github repository](https://github.com/JoshEngels/MultiDimensionalFeatures)) to extract token activations corresponding to the features "years of the 20th century", "months of the year" and "days of the week". 

Running this part is optional. The output of this part is saved in the directory `representations`.

To run this part, download gpt-2 layer 7, and Mistral-7B layer 8 token activations from [https://www.dropbox.com/scl/fo/frn4tihzkvyesqoumtl9u/AFPEAa6KFb8mY3NTXIEStnA?rlkey=z60j3g45jzhxwc5s5qxmbjvxs&st=da2tzqk5&dl=0](https://www.dropbox.com/scl/fo/frn4tihzkvyesqoumtl9u/AFPEAa6KFb8mY3NTXIEStnA?rlkey=z60j3g45jzhxwc5s5qxmbjvxs&st=da2tzqk5&dl=0), and put them in the `data` directory.

Set up the following virtual environment (this will work for the following parts as well)
```
python3.11 -m venv RepresentationManifolds
source RepresentationManifolds/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib plotly ipykernel nbformat einops sae_lens tqdm openai
```
Set up a Huggingface login token and save it in an environment variable `HUGGINGFACE_TOKEN`. Get access to `Mistral-7B-v0.1`.

Then run the code in the notebook `1-process_sae_activations.ipynb`.

**Part 2:** Getting text embeddings from OpenAI's `text-embedding-large-3`.

Running this code is currently required, but we will shortly provide a link to download the embeddings obtained in this part.

Set up the following virtual environment if you didn't do the first part (this will work for the following parts as well)
```
python3.11 -m venv RepresentationManifolds
source RepresentationManifolds/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib plotly ipykernel nbformat openai
```

Setup an account with OpenAI and save your access token in an environment variables `OPENAI_API_KEY`.

Then run the code in the notebook `2-get_text_embeddings.ipynb`.

**Part 3:** Reproducing the figures from the paper.

If you didn't run either of the previous two parts, set up the following virtual environment:
```
python3.11 -m venv RepresentationManifolds
source RepresentationManifolds/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib plotly ipykernel nbformat
```

Then run the code in the notebook `3-reproduce_figures.ipynb`.


##### References
Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. Not All Language Model Features Are One-Dimensionally Linear. In *The Thirteenth International Conference on 368 Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*.
