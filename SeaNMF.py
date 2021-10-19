import os
import time
import numpy as np
from numpy.linalg import norm
from gensim import corpora, models


class SeaNMFL1(object):
    '''
    source: https://github.com/tshi04/SeaNMF
    '''
    def __init__(
        self,
        A, S, 
        IW1=[], IW2=[], IH=[],
        alpha=1.0, beta=0.1, n_topic=10, max_iter=100, max_err=1e-3, 
        rand_init=True, fix_seed=False):
        '''
        0.5*||A-WH^T||_F^2+0.5*alpha*||S-WW_c^T||_F^2+0.5*beta*||W||_1^2
        W = W1
        Wc = W2
        '''
        if fix_seed: 
            np.random.seed(0)

        self.A = A
        self.S = S

        self.n_row = A.shape[0]
        self.n_col = A.shape[1]

        self.n_topic = n_topic
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.B = np.ones([self.n_topic,1])
        self.max_err = max_err

        if rand_init:
            self.nmf_init_rand()
        else:
            self.nmf_init(IW1, IW2, IH)
        self.nmf_iter()

    def nmf_init_rand(self):
        self.W1 = np.random.random((self.n_row, self.n_topic))
        self.W2 = np.random.random((self.n_row, self.n_topic))
        self.H = np.random.random((self.n_col, self.n_topic))

        for k in range(self.n_topic):
            self.W1[:, k] /= norm(self.W1[:, k])
            self.W2[:, k] /= norm(self.W2[:, k])

    def nmf_init(self, IW1, IW2, IH):
        self.W1 = IW1
        self.W2 = IW2
        self.H = IH

        for k in range(self.n_topic):
            self.W1[:, k] /= norm(self.W1[:, k])
            self.W2[:, k] /= norm(self.W2[:, k])

    def nmf_iter(self):
        loss_old = 1e20
        print('loop begin')
        start_time = time.time()
        for i in range(self.max_iter):
            self.nmf_solver()
            loss = self.nmf_loss()
            if loss_old-loss < self.max_err:
                break
            loss_old = loss
            end_time = time.time()
            print('Step={}, Loss={}, Time={}s'.format(i, loss, end_time-start_time))

    def nmf_solver(self):
        '''
        using BCD framework
        '''
        epss = 1e-20
        # Update W1
        AH = np.dot(self.A, self.H)
        SW2 = np.dot(self.S, self.W2)
        HtH = np.dot(self.H.T, self.H)
        W2tW2 = np.dot(self.W2.T, self.W2)
        W11 = self.W1.dot(self.B)

        for k in range(self.n_topic):
            num0 = HtH[k,k]*self.W1[:,k] + self.alpha*W2tW2[k,k]*self.W1[:,k]
            num1 = AH[:,k] + self.alpha*SW2[:,k]
            num2 = np.dot(self.W1, HtH[:,k]) + self.alpha*np.dot(self.W1, W2tW2[:,k]) + self.beta*W11[0]
            self.W1[:,k] = num0 + num1 - num2
            self.W1[:,k] = np.maximum(self.W1[:,k], epss) # project > 0
            self.W1[:,k] /= norm(self.W1[:,k]) + epss # normalize
        # Update W2
        W1tW1 = self.W1.T.dot(self.W1)
        StW1 = np.dot(self.S, self.W1)
        for k in range(self.n_topic):
            self.W2[:,k] = self.W2[:,k] + StW1[:,k] - np.dot(self.W2, W1tW1[:,k])
            self.W2[:,k] = np.maximum(self.W2[:,k], epss)
        #Update H
        AtW1 = np.dot(self.A.T, self.W1)
        for k in range(self.n_topic):
            self.H[:,k] = self.H[:,k] + AtW1[:,k] - np.dot(self.H, W1tW1[:,k])
            self.H[:,k] = np.maximum(self.H[:,k], epss) 

    def nmf_loss(self):
        '''
        Calculate loss
        '''
        loss = norm(self.A - np.dot(self.W1, np.transpose(self.H)), 'fro')**2/2.0
        if self.alpha > 0:
            loss += self.alpha*norm(np.dot(self.W1, np.transpose(self.W2))-self.S, 'fro')**2/2.0
        if self.beta > 0:
            loss += self.beta*norm(self.W1, 1)**2/2.0
        
        return loss
    
    def get_lowrank_matrix(self):
        return self.W1, self.W2, self.H
    
    def save_format(self, W1file='W.txt', W2file='Wc.txt', Hfile='H.txt'):
        np.savetxt(W1file, self.W1)
        np.savetxt(W2file, self.W2)
        np.savetxt(Hfile, self.H)
        


def prep_data(source_docs_filepath, docs_filepath, vocab_filepath):
    '''
    source_docs_filepath: 文档文件，一行为一个文档
    '''
    # build vocab and represent docs in vocab index
    docs = list(map(lambda line: line.split(), open(source_docs_filepath).read().splitlines()))
    dictionary = corpora.Dictionary(docs)
    docs_mat = [dictionary.doc2idx(doc) for doc in docs]
    with open(vocab_filepath, 'w') as f:
        f.write('\n'.join(list(dictionary.values())))
    with open(docs_filepath, 'w') as f:
        for doc in docs_mat:
            f.write(' '.join(map(str, doc)) + '\n')




def train(docs_filepath, vocab_filepath):
    '''
    vocab_filepath(.txt): 所有词汇，一行为一个词，位置与其在词汇表中的索引对应
    docs_filepath(.txt): 所有文档，一行为一个文档，文档中的词用词汇表索引表示
    '''
    MAX_ITER = 500
    N_TOPICS = 100
    ALPHA = 0.1
    BETA = 0.0
    MAX_ERR = 0.1
    FIX_SEED = True

    docs = list(map(lambda line: map(int, line.split()) , open(docs_filepath).read().splitlines()))
    vocab = open(vocab_filepath).read().splitlines()
    n_docs = len(docs)
    n_terms = len(vocab)
    print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

    tmp_folder = 'results/SeaNMF'
    if not os.access(tmp_folder, os.F_OK):
        os.mkdir(tmp_folder)
    print('calculate co-occurance matrix')
    dt_mat = np.zeros([n_terms, n_terms])
    for itm in docs:
        for kk in itm:
            for jj in itm:
                dt_mat[int(kk),int(jj)] += 1.0
    print('co-occur done')
    print('-'*50)
    print('calculate PPMI')
    D1 = np.sum(dt_mat)
    SS = D1*dt_mat
    for k in range(n_terms):
        SS[k] /= np.sum(dt_mat[k])
    for k in range(n_terms):
        SS[:,k] /= np.sum(dt_mat[:,k])
    dt_mat = [] # release memory
    SS[SS==0] = 1.0
    SS = np.log(SS)
    SS[SS<0.0] = 0.0
    print('PPMI done')
    print('-'*50)
    
    print('read term doc matrix')
    dt_mat = np.zeros([n_terms, n_docs])
    for k in range(n_docs):
        for j in docs[k]:
            dt_mat[j, k] += 1.0
    print('term doc matrix done')
    print('-'*50)
    
    model = SeaNMFL1(
        dt_mat, SS,  
        alpha=ALPHA, 
        beta=BETA, 
        n_topic=N_TOPICS, 
        max_iter=MAX_ITER, 
        max_err=MAX_ERR,
        fix_seed=FIX_SEED)

    model.save_format(
        W1file=tmp_folder+'/W.txt',
        W2file=tmp_folder+'/Wc.txt',
        Hfile=tmp_folder+'/H.txt')



def print_topics(docs_filepath, vocab_filepath, result_filepath):
    '''
    result_filepath: 保存SeaNMF结果的W.txt文件
    '''
    docs = list(map(lambda line: map(int, line.split()) , open(docs_filepath).read().splitlines()))
    vocab = open(vocab_filepath).read().splitlines()
    n_docs = len(docs)
    n_terms = len(vocab)
    print('n_docs={}, n_terms={}'.format(n_docs, n_terms))

    dt_mat = np.zeros([n_terms, n_terms])
    for itm in docs:
        for kk in itm:
            for jj in itm:
                if kk != jj:
                    dt_mat[int(kk), int(jj)] += 1.0
    print('co-occur done')
            
    W = np.loadtxt(result_filepath, dtype=float)
    n_topic = W.shape[1]
    print('n_topic={}'.format(n_topic))

    PMI_arr = []
    n_topKeyword = 10
    for k in range(n_topic):
        topKeywordsIndex = W[:,k].argsort()[::-1][:n_topKeyword]
        PMI_arr.append(calculate_PMI(dt_mat, topKeywordsIndex))
    print('Average PMI={}'.format(np.average(np.array(PMI_arr))))

    index = np.argsort(PMI_arr)
    
    for k in index:
        print('Topic ' + str(k+1) + ': ', end=' ')
        print(PMI_arr[k], end=' ')
        for w in np.argsort(W[:,k])[::-1][:n_topKeyword]:
            print(vocab[w], end=' ')
        print()    



def calculate_PMI(AA, topKeywordsIndex):
    '''
    Reference:
    Short and Sparse Text Topic Modeling via Self-Aggregation
    '''
    D1 = np.sum(AA)
    n_tp = len(topKeywordsIndex)
    PMI = []
    for index1 in topKeywordsIndex:
        for index2 in topKeywordsIndex:
            if index2 < index1:
                if AA[index1, index2] == 0:
                    PMI.append(0.0)
                else:
                    C1 = np.sum(AA[index1])
                    C2 = np.sum(AA[index2])
                    PMI.append(np.log(AA[index1,index2]*D1/C1/C2))
    avg_PMI = 2.0*np.sum(PMI)/float(n_tp)/(float(n_tp)-1.0)

    return avg_PMI


if __name__ == "__main__":
    source_docs_filepath = "data/SeaNMF/docs.txt"
    docs_filepath = "data/SeaNMF/docs_mat.txt"
    vocab_filepath = "data/SeaNMF/vocab.txt"
    result_filepath = "results/SeaNMF/W.txt"

    prep_data(source_docs_filepath, docs_filepath, vocab_filepath)
    train(docs_filepath=docs_filepath, vocab_filepath=vocab_filepath)
    print_topics(docs_filepath, vocab_filepath, result_filepath)