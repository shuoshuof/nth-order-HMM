import math
import pickle
import numpy as np

class HMM:
    def __init__(self,dataset_path,model_path=None):
        self.path= dataset_path
        self.dataset=self.read_dataset()
        self.model=self.load(model_path) if model_path is not None else None
    def read_dataset(self):
        with open(self.path,'rb') as f:
            x_train = pickle.load(f)
            y_train = pickle.load(f)
            x_test = pickle.load(f)
            y_test = pickle.load(f)
            label2num = pickle.load(f)
            return (x_train,y_train,x_test,y_test,label2num)
    def train(self):
        x_train,y_train,x_test,y_test,label2num = self.dataset
        A = np.zeros((4, 4))#状态转移概率矩阵
        B = np.zeros((4, 65536))#发射概率矩阵
        pi = np.zeros(4)
        for sentence,label_seq in zip(x_train,y_train):
            for idx,(word,label) in enumerate(zip(sentence,label_seq)):
                if idx == 0:
                    pi[label]+=1
                else:
                    A[label_seq[idx-1]][label_seq[idx]]+=1
                B[label][ord(word)]+=1
        A = A/np.sum(A,axis=1).reshape(-1,1)
        A = np.where(A==0,np.full_like(A,-3.14e+100),np.log(A))
        B = B/np.sum(B,axis=1).reshape(-1,1)
        B = np.where(B==0,np.full_like(B,-3.14e+100),np.log(B))
        pi = pi/np.sum(pi)
        pi = np.where(pi == 0, np.full_like(pi, -3.14e+100), np.log(pi))
        self.model = (pi,A,B)
        self.evaluate('train')
        self.evaluate('test')
    def predict(self, sentence):

        pi, A, B = self.model
        T = len(sentence)
        psi = np.zeros((T, 4))
        delta = np.zeros((T, 4))

        for t, word in enumerate(sentence):
            o_i = ord(word)
            for i in range(4):
                if t == 0:
                    delta[t][i] = pi[i] + B[i][o_i]
                    psi[t][i] = 0
                else:
                    p = [delta[t - 1][j] + A[j][i] for j in range(4)]
                    delta[t][i] = np.max(p) + B[i][o_i]
                    psi[t][i] = np.argmax(p)

        i_star = np.argmax(delta[T - 1])
        label_seq = []
        states = ['B', 'M', 'E', 'S']
        last_state = i_star
        label_seq.insert(0, states[last_state])
        for t in range(T - 1, 0, -1):
            last_state = int(psi[t][last_state])
            label_seq.insert(0, states[last_state])
        result =''
        for word,label in zip(sentence,label_seq):
            result+=word
            if label =='S' or label =='E':
                result+=' '
        return label_seq
    def _split_label(self,seq):
        splited_set =[]
        word_label = ''
        for idx,label in enumerate(seq):
            word_label+=str(idx)+' '
            if label == 'S' or label == 'E':
                splited_set.append(word_label)
                word_label=''
        splited_set = set(splited_set)
        return splited_set
    def evaluate(self,dataset:str):
        if dataset=='train':
            x,y = self.dataset[0:2]
        elif dataset == 'test':
            x,y = self.dataset[2:4]
        states = ['B', 'M', 'E', 'S']
        sum_base_A = 0
        sum_base_B = 0
        sum_base_TP = 0
        for sentence, label_seq in zip(x, y):
            if len(sentence):
                y_pred = self.predict(sentence)
                label_seq = list(map(lambda idx:states[idx],label_seq))
                A = self._split_label(label_seq)
                B = self._split_label(y_pred)
                TP = A.intersection(B)
                base_TP = len(TP)
                base_A = len(A)
                base_B = len(B)
                sum_base_B+=base_B
                sum_base_A+=base_A
                sum_base_TP+=base_TP
        P = sum_base_TP/sum_base_B
        R = sum_base_TP/sum_base_A
        print(dataset)
        print('正确率',P)
        print('召回率',R)
        print('F1',(2*P*R)/(P+R))

    def save(self):
        with open('model.pickle', 'wb') as f:
            pickle.dump(self.model, f)
    def load(self,path):
        with open(path, 'rb') as f:
            return pickle.load(f)




if __name__ =='__main__':
    # dataset = HMM_model('./dataset/HMMTrainSet.txt')
    # model = dataset.train()
    # sentence = '在这一年中，中国的改革开放和现代化建设继续向前迈进。'
    # predict(model,sentence)
    # model = HMM('./dataset/data.pickle')

    model = HMM(dataset_path='./dataset/data.pickle',model_path='./model.pickle')
    model.train()
    model.save()
    # sentence = '在这一年中，中国的改革开放和现代化建设继续向前迈进。'
    # model.predict(sentence)
    # print(list(sentence))
    # model.predict(sentence)





