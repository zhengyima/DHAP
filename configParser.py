import argparse

parser = argparse.ArgumentParser()

# File paths
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='File path of the checkpoint to load. //需要恢复的检查点路径。')

parser.add_argument('--train_post_path', action='store', dest='train_post_path', help='Train data file path. //训练集路径。')
parser.add_argument('--dev_post_path', action='store', dest='dev_post_path', help='Dev data file path. //验证集路径。')
parser.add_argument('--test_post_path', action='store', dest='test_post_path', help='Train data file path. //测试集路径。')

parser.add_argument('--train_resp_path', action='store', dest='train_resp_path', help='Train data file path. //训练集路径。')
parser.add_argument('--dev_resp_path', action='store', dest='dev_resp_path', help='Dev data file path. //验证集路径。')
parser.add_argument('--test_resp_path', action='store', dest='test_resp_path', help='Train data file path. //测试集路径。')

parser.add_argument('--src_vocab_file', action='store', dest='src_vocab_file', 
                    help='Source vocab file path. //输入文件词表路径。')
parser.add_argument('--tgt_vocab_file', action='store', dest='tgt_vocab_file', 
                    help='Target vocab file path. //输出文件词表路径。')

parser.add_argument('--model_dir', action='store', dest='model_dir', default='./experiment_2', 
                    help='Path to model directory. //模型保存目录。')
parser.add_argument('--best_model_dir', action='store', dest='best_model_dir', default='./experiment/best', 
                    help='Path to best model directory. //最好模型保存目录。')
parser.add_argument('--max_checkpoints_num', action='store', dest='max_checkpoints_num', default=150, 
                    help='Max num of checkpoints. //最多保存模型数量。', type=int)

parser.add_argument('--log_level', action='store', dest='log_level', default='info', help='Logging level. //日志的输出等级。')
parser.add_argument('--log_file', action='store', dest='log_file', default='train.log', help='Logging file path. //日志的输出路径。')


# Model learning
parser.add_argument('--batch_size', action='store', dest='batch_size', 
                    help='Size of batch. //batch大小。', default=2, type=int)
parser.add_argument('--resume', action='store_true', dest='resume', default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint. \
                          If load_checkpoint is set, then train from loaded. //是否从最新的检查点恢复训练，若指定load_checkpoint，则从指定检查点恢复。')
parser.add_argument('--max_steps', action='store', dest='max_steps', 
                    help='Maximum num of steps for training. //最大训练步数。', default=10, type=int)
parser.add_argument('--max_epochs', action='store', dest='max_epochs', 
                    help='Maximum num of epochs for training. //最大训练轮数。', default=1, type=int)
parser.add_argument('--skip_steps', action='store', dest='skip_steps', 
                    help='Num of steps skipped at the beginning of training. //在训练开始时跳过步数量。', default=0, type=int)
parser.add_argument('--checkpoint_every', action='store', dest='checkpoint_every', 
                    help='Num of batches to checkpoint. //每多少步保存检查点。', default=5, type=int)
parser.add_argument('--print_every', action='store', dest='print_every', 
                    help='Num of batches to print loss. //每多少步输出loss。', default=2, type=int)
parser.add_argument('--init_weight', action='store', dest='init_weight', 
                    help='Initial weights from [-this, this]. //参数初始化范围。', default=0.08, type=float)
parser.add_argument('--clip_grad', action='store', dest='clip_grad', 
                    help='Clip gradients to this norm. //最大梯度截断。', default=5.0, type=float)
parser.add_argument('--learning_rate', action='store', dest='learning_rate', 
                    help='Learning rate. //学习率', default=0.001, type=float)
parser.add_argument('--decay_factor', action='store', dest='decay_factor',
                    help='How much we decay learning rate. //学习率衰减因子。', default=0.995, type=float)
parser.add_argument('--best_ppl', dest='best_ppl', 
                    help='Initial ppl threshold for saving best model. //用做保存最好模型的初始PPL阈值。', default=100000.0, type=float)


# Model structure
parser.add_argument('--src_vocab_size', action='store', dest='src_vocab_size', 
                    help='Size of source vocab. //输入词表大小。', default=40000, type=int)
parser.add_argument('--tgt_vocab_size', action='store', dest='tgt_vocab_size', 
                    help='Size of target vocab. //输出词表大小。', default=40000, type=int)
parser.add_argument('--embedding_size', action='store', dest='embedding_size', 
                    help='Size of embedding. //词向量维度。', default=100, type=int)

parser.add_argument('--rnn_cell', action='store', dest='rnn_cell', 
                    help='Type of RNN cell. gru or lstm. //RNN的类型，gru或lstm。', default='gru')
parser.add_argument('--n_hidden_layer', action='store', dest='n_hidden_layer', 
                    help='Num of hidden layer in each RNN. //RNN的隐藏层数。', default=1, type=int)
parser.add_argument('--hidden_size', action='store', dest='hidden_size', 
                    help='Size of each RNN hidden layer. //RNN的隐藏层维度。', default=64, type=int)
parser.add_argument('--bidirectional', action='store_true', dest='bidirectional', 
                    help='If use bidirectional RNN. //是否使用双向RNN。', default=False)
parser.add_argument('--max_src_length', action='store', dest='max_src_length', 
                    help='Max length of source. //输入的最大长度。', default=50, type=int)
parser.add_argument('--max_tgt_length', action='store', dest='max_tgt_length', 
                    help='Max length of target. //输出的最大长度。', default=50, type=int)

parser.add_argument('--use_attn', action='store_true', dest='use_attn', 
                    help='If use attention. //是否用注意力机制。', default=True)

parser.add_argument('--use_knowledge', action='store_true', dest='use_know', 
                    help='If use history for background knowledge. //是否用历史。', default=True)

parser.add_argument('--teacher_forcing_ratio', action='store', dest='teacher_forcing_ratio', 
                    help='teacher forcing ratio. //teacher forcing率。', default=0.5, type=float)


# Others
parser.add_argument('--random_seed', action='store', dest='random_seed', default=None, help='Dandom seed. //随机种子。', type=int)
parser.add_argument('--device', action='store', dest='device', default=None, help='GPU device. //使用的GPU编号。')
parser.add_argument('--phase', action='store', dest='phase', default='train', help='train or infer. //训练或预测。')
parser.add_argument('--beam_width', action='store', dest='beam_width', 
                    help='Beam width when using beam search decoder in inference. //Beam Search宽度。', default=1, type=int)

parser.add_argument('--data_path', action='store', dest='data_path', 
                    help='Type of RNN cell. gru or lstm. //RNN的类型，gru或lstm。', default='/home/zhengyi_ma/pcb/Data/PChatbot_byuser_filter')
parser.add_argument('--user_limit', action='store', dest='user_limit', 
                    help='Type of RNN cell. gru or lstm. //RNN的类型，gru或lstm。', type=int, default=100)
parser.add_argument('--word2vec_path', action='store', dest='word2vec_path', 
                    help='Type of RNN cell. gru or lstm. //RNN的类型，gru或lstm。',default='/home/zhengyi_ma/pcb/Data/PChatbot.word2vec.200d.txt')    
parser.add_argument('--result_path', action='store', dest='result_path', 
                    help='path of the result file. //RNN的类型，gru或lstm。',default='./res.txt')                 
opt = parser.parse_args()