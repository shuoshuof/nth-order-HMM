import pickle

class dataset:
    def __init__(self,train_path,test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.label2num = {"B-CWS":0,"I-CWS":1,"E-CWS":2,"S-CWS":3}
    def read_data(self,path):
        with open(path,encoding='utf-8',mode='r') as data:
            sentence_num=0
            sentences = []
            label_seqs = []
            sentence = []
            label_seq = []
            end_flag = 0
            for idx,line in enumerate(data.readlines()):
                if not line.isspace():
                    word, label = line.strip().split()
                    label = self.label2num[label]
                    if word=='“' and len(word):
                        sentences.append(sentence)
                        label_seqs.append(label_seq)
                        # print(sentence)
                        # print(label_seq)
                        sentence=[]
                        label_seq=[]
                    else:
                        sentence.append(word)
                        label_seq.append(label)
                        if end_flag==1:
                            end_flag=0
                else:
                    end_flag=1
            return sentences,label_seqs
    def save_data(self,x_train,y_train,x_test,y_test):
        with open('data.pickle','wb') as f:
            pickle.dump(x_train,f)
            pickle.dump(y_train,f)
            pickle.dump(x_test,f)
            pickle.dump(y_test,f)
            pickle.dump(self.label2num,f)
    def __call__(self):
        x_train,y_train = self.read_data(self.train_path)
        x_test,y_test = self.read_data(self.test_path)
        self.save_data(x_train,y_train,x_test,y_test)

if __name__ =='__main__':
    dataset = dataset('./msra_cws/train.txt','./msra_cws/test.txt')
    dataset()
    with open('./data.pickle','rb') as f:
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)
        label2num = pickle.load(f)
    print(x_test[0])
    print(y_test[0])