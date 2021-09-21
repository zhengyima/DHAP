# DHAP
Source code of SIGIR2021 Long Paper: 

[One Chatbot Per Person: Creating Personalized Chatbots based on Implicit User Profiles ](https://arxiv.org/abs/2108.09355).

# Preinstallation
First, install the python packages in your **Python3** environment:
```
  git clone https://github.com/zhengyima/DHAP.git DHAP
  cd DHAP
  pip install -r requirements.txt
```

Then, you should download the pre-trained word embeddings to initialize the model training. We provide two word embeddings in the Google Drive:
- sgns.weibo.bigram-char, folloing [Li et al.](https://github.com/Embedding/Chinese-Word-Vectors), Chinese word embeddings pre-trained on Weibo. [Google Drive](https://drive.google.com/drive/folders/1UqUNtO5SVjyYTERfi4IvVTHopjFtqNNO?usp=sharing)
- Fasttext embeddings, English word embedding pre-trained on Reddit set. [Google Drive](https://drive.google.com/drive/folders/1UqUNtO5SVjyYTERfi4IvVTHopjFtqNNO?usp=sharing)

You can pre-train your own embeddings(with the same format, i.e., the standard txt format), and use it in the model.

After downloading, you should put the embedding file to the path ```EMB_FILE```.

# Data

You should provide the dialogue history of users for training the model. For convenience, we provide a very small subset of [PChatbot](https://github.com/qhjqhj00/SIGIR2021-Pchatbot) in the ```data/``` as the demo data. In the direcotry, each user's dialogue history is saved in one text file. Each line in the file should contain ```post text, user id of post, post timestamp, response text, user id of response, response timestamp, _, _ ```, with tab as the seperator. 

You can refer to ```seq2seq/dataset/perdialogDatasets.py``` for more details about the data processing.

If you are interested in the dataset [PChatbot](https://github.com/qhjqhj00/SIGIR2021-Pchatbot), please go to its official repository for more details. 

# Model Training

We provide a shell script ```scripts/train_chat.sh``` to start model pre-training. You should modify the ```DATA_DIR``` and ```EMB_FILE``` to your own paths. Then, you can start training by the following command: 
```
bash scripts/train_chat.sh
```

The hyper-parameters are defined and set in the ```configParser.py```.

After training, the trained checkpoints are saved in ```outputs```. The inferenced result is saved in ```RESULT_FILE```, which you define in ```bash scripts/train_chat.sh```


# Evaluating

For calculating varities of evaluation metrics(e.g. BLEU, P-Cover...), we provide a shell script ```scripts/eval.sh```. You should modify the ```EMB_FILE``` to your own path, then evaluate the results by the following command: 
```
bash scripts/eval.sh
```

# Citations

If our code helps you, please cite our work by:
```
@inproceedings{DBLP:conf/sigir/madousigir21,
     author = {Zhengyi Ma and Zhicheng Dou and Yutao Zhu Hanxun Zhong and Ji-Rong Wen}, 
     title = {One Chatbot Per Person: Creating Personalized Chatbots based onImplicit User Profiles}, 
     booktitle = {Proceedings of the {SIGIR} 2021}, 
     publisher = {{ACM}}, 
     year = {2021}, 
     url = {https://doi.org/10.1145/3404835.3462828}, 
     doi = {10.1145/3404835.3462828}}
```

# Links
- [Pytorch](https://pytorch.org)
- [PChatbot Dataset](https://github.com/qhjqhj00/SIGIR2021-Pchatbot)




