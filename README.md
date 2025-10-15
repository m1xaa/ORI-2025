# Video Transcriber

AI tool used for extracting the most important theses from video. The output of the application is k best rated theses with timestamps so if you are interested in any part of the video you can quickly go and watch that part.

## Description

How it works:
1. *whisper_asr.py* is used for converting video into segments containing text and timespan
2. *load_annotations.py* is used for loading the annotations of the videos from the training dataset which give each part of the video some importance score
3. *text_processing.py* is used for text normalization
4. *featurization.py* is used to make a feature matrix from the previously loaded annotations, using tfidf score, textrank, segment length, position and segment embeddings
5. *ranker.py* containts the code for training the model
6. *evaluation.py* this file contains the logic for computing bertscore, the evalution metric for this model

## Prerequisites

In order to run this application you need to have the following installed on your computer:
- [Python](https://www.python.org/downloads/) - 3.10+ preferably
- [Pip](https://pypi.org/project/pip/) - suggest you to go for the newest stable version

Also in order to run the application you will need a code editor, any will do, but I recommend pycharm:
- [Pycharm for Windows](https://www.jetbrains.com/pycharm/download/?section=windows)
- [Pycharm for Linux](https://www.jetbrains.com/pycharm/download/?section=linux)

## Installation

1. Clone the repo first
    ```bash
    git clone https://github.com/m1xaa/ORI-2025.git
    ```


2. Open the root folder(pythonProject) in your code editor

   
3. Create the virtual environment
    ```bash
    python -m venv venv
    ```


4. Then activate it
    - on Windows:
    ```bash
    venv\Scripts\activate
    ```
    - on Linux:
    ```bash
    . venv/bin/activate
    ```


5. Finally to install all the packages and libraries run
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can either run this project to train your own model or to extract theses from a video.  


### Train model

Run the following command 

```bash
python -m train
```
And then you need to specify command line arguments in this format, some are required some are optionally
```bash
--[arg name] [arg value]
```
Required arguments:
- **root**, the root directory of your training videos. You need to have the following structure
```text
root/
├── videos/
│   ├── video_name
│   ├── video_name
└── video_annotations.mat
```
- **whisper_model**, spearch recognition model, default is small, also supports base and large
- **language**, default is en, you must type the two character code of the language used in the video
- **model_out**, the output directory of your trained model

Optional arguments:
- **embedding_model**, the model for sentence embeddings
- **top_k**, the top k most rated theses that model will be traind on, default is 5

### Extract theses

Run the following command
```bash
python -m main
```

Then you will need to specify the following command line arguments:
- **video_path**, the path of the video you want to extract theses from
- **output**, the output directory where you want to store extracted theses
- **whisper_model**, spearch recognition model, default is small, also supports base and large
- **language**, default is en, you must type the two character code of the language used in the video
- **ranker_model**, the path to your trained model for ranking theses
- **top_k**, the top k most rated theses that model will be traind on, default is 5








