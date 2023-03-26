import pickle
import numpy as np
import pandas as pd
class HMM:
    def __init__(self,dataset_path,model_path=None,n=2):
        self.path= dataset_path
        self.dataset=self.read_dataset()
        self.model=self.load(model_path) if model_path is not None else None
        self.n = n
        self.metrics =None
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
        A = np.zeros((4,)*(self.n+1))#状态转移概率矩阵
        B = np.zeros((4, 65536))#发射概率矩阵
        pi = np.zeros(4)
        A_ = np.zeros((4,4))
        for sentence,label_seq in zip(x_train,y_train):
            for idx in range(len(sentence)):
                word = sentence[idx]
                if idx == 0:
                    label=label_seq[idx]
                    pi[label]+=1
                elif idx <self.n:
                    A_[label_seq[idx-1]][label_seq[idx]]+=1
                else:
                    indices = tuple([ label_seq[i] for i in range(idx-self.n,idx+1)])
                    A[indices]+=1
                B[label_seq[idx]][ord(word)]+=1
        A_sum = np.sum(A,axis=-1).reshape((4,)*self.n+(1,))
        A_sum = np.where(A_sum==0,1,A_sum)
        A = A/A_sum
        A = np.where(A==0,np.full_like(A,-3.14e+100),np.log(A))
        B = B/np.sum(B,axis=1).reshape(-1,1)
        B = np.where(B==0,np.full_like(B,-3.14e+100),np.log(B))
        if self.n>2:
            A_ = A_/np.sum(A_,axis=-1).reshape(-1,1)
            A_ = np.where(A_==0,np.full_like(A_,-3.14e+100),np.log(A_))
        pi = pi/np.sum(pi)
        pi = np.where(pi == 0, np.full_like(pi, -3.14e+100), np.log(pi))
        self.model = (pi,A,B,A_)
        train_metrics = self.evaluate('train')
        test_metrics = self.evaluate('test')
        self.metrics = np.array(train_metrics+test_metrics)
    def predict(self, sentence):

        pi, A, B, A_ = self.model
        T = len(sentence)
        psi = np.zeros([T]+[4]*self.n)
        delta = np.zeros([T]+[4]*self.n)
        def loop_fun1(t,n,state_seq:list):
            # t=0时最大概率
            o_i = ord(sentence[0])
            p = pi[state_seq[0]] + B[state_seq[0]][o_i]
            for t1 in range(1, n):
                o_i = ord(sentence[t1])
                p += A_[state_seq[t1 - 1]][state_seq[t1]] + B[state_seq[t1]][o_i]
            delta[t][tuple(state_seq)] = p
            psi[t][tuple(state_seq)] = 0
        def loop_fun2(t,n,state_seq:list):
            # state_seq t时刻的结尾下标序列
            p = [delta[t-1][i][tuple(state_seq[:-1])] + A[i][tuple(state_seq)] for i in range(4)]
            o_i = ord(sentence[t])
            delta[t][tuple(state_seq)] = np.max(p)+B[state_seq[n-1]][o_i]
            psi[t][tuple(state_seq)]= np.argmax(p)
        def n_loop(t,n,state_seq:list,fun):
            if len(state_seq)==n:
                fun(t,n,state_seq)
            else:
                for state in range(4):
                    n_loop(t,n,state_seq+[state],fun)
        n_loop(
            t=self.n - 1,
            n=self.n,
            state_seq = [],
            fun=loop_fun1
        )
        for t in range(self.n,T):
            n_loop(
                t =t,
                n=self.n,
                state_seq=[],
                fun = loop_fun2
            )
        # 标签序列为l_1~l_T
        # 选出l_(T-n)~l_T的最佳状态，也就是说l_(T-n+1)~l_T如何取才最佳
        indices_star = np.unravel_index(np.argmax(delta[T - 1]),delta[T - 1].shape)


        states = ['B', 'M', 'E', 'S']
        last_state_indices = tuple(indices_star)
        label_seq = tuple(indices_star)


        for t in range(T - 1, self.n-1, -1):
            last_state_indices = (psi[t][last_state_indices],)+last_state_indices[:-1]
            last_state_indices = tuple(map(lambda x:int(x),last_state_indices))
            label_seq=(last_state_indices[0],)+label_seq
        label_seq  = list(map(lambda x:states[x],label_seq))
        # print(label_seq)
        result =''

        for word,label in zip(sentence,label_seq):
            result+=word
            if label =='S' or label =='E':
                result+=' '
        # print(result)
        # print(len(label_seq),T)
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
        F1 = (2*P*R)/(P+R)
        print(dataset)
        print('正确率',P)
        print('召回率',R)
        print('F1',F1)

        return [P,R,F1]

    def save(self):
        with open('{}_model.pickle'.format(self.n), 'wb') as f:
            pickle.dump(self.model, f)
    def load(self,path):
        with open(path, 'rb') as f:
            return pickle.load(f)




if __name__ =='__main__':

    metrics = []
    for n in range(1,4):
        model = HMM(dataset_path='./dataset/data.pickle',model_path=None,n=n)
        model.train()
        model.save()
        metrics.append(model.metrics)
    with open('./metrics.pickle', 'wb') as f:
        pickle.dump(metrics,f)
    metrics = pd.DataFrame(metrics,
                           columns=['训练集正确率', '训练集召回率','训练集F1','验证集正确率', '验证集召回率','验证集F1'],
                           index=['一阶','二阶','三阶']
                        )
    metrics.to_csv('./metrics.csv')





