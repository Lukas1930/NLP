import numpy as np 
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from models.BiLSTM_CNN_2.prepro import readfile,createBatches,createMatrices,iterate_minibatches,addCharInformatioin,padding
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def tag_dataset(dataset, model):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing,char], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    b.update(i+1)
    return predLabels, correctLabels

def train_model(train_data_path, validation_data_path, statwars_data_path, emeddings_path, epochs=10, finetune=False):
    trainSentences = readfile(train_data_path)
    devSentences = readfile(validation_data_path)
    starwarsSentences = readfile(statwars_data_path)

    trainSentences = addCharInformatioin(trainSentences)
    devSentences = addCharInformatioin(devSentences)
    starwarsSentences = addCharInformatioin(starwarsSentences)

    labelSet = set()
    words = {}

    for dataset in [trainSentences, devSentences, starwarsSentences]:
        for sentence in dataset:
            for token,char,label in sentence:
                labelSet.add(label)
                words[token.lower()] = True

    # :: Create a mapping for the labels ::
    label2Idx = {'B-ORG': 1,
                 'O': 2,
                 'B-MISC': 3,
                 'B-PER': 4,
                 'B-LOC': 5,
                 'I-PER': 6,
                 'I-ORG': 7,
                 'I-LOC': 8,
                 'I-MISC': 9,
                 'PADDING': 0}

    # :: Hard coded case lookup ::
    case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
    caseEmbeddings = np.identity(len(case2Idx), dtype='float32')


    # :: Read in word embeddings ::
    word2Idx = {}
    wordEmbeddings = []

    fEmbeddings = open(emeddings_path, encoding="utf-8")

    for line in fEmbeddings:
        split = line.strip().split(" ")
        word = split[0]
        
        if len(word2Idx) == 0: #Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
            wordEmbeddings.append(vector)
            
            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            wordEmbeddings.append(vector)

        if split[0].lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[split[0]] = len(word2Idx)
            
    wordEmbeddings = np.array(wordEmbeddings)

    char2Idx = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char2Idx[c] = len(char2Idx)

    train_set = padding(createMatrices(trainSentences,word2Idx,  label2Idx, case2Idx,char2Idx))
    dev_set = padding(createMatrices(devSentences,word2Idx, label2Idx, case2Idx,char2Idx))
    starwars_set = padding(createMatrices(starwarsSentences, word2Idx, label2Idx, case2Idx,char2Idx))

    idx2Label = {v: k for k, v in label2Idx.items()}
    np.save("models/idx2Label.npy",idx2Label)
    np.save("models/word2Idx.npy",word2Idx)

    train_batch,train_batch_len = createBatches(train_set)
    dev_batch,dev_batch_len = createBatches(dev_set)

    finetune_sw_set, test_sw_set = train_test_split(
        starwars_set, test_size=0.2, random_state=42
    )
    
    words_input = Input(shape=(None,),dtype='int32',name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False)(casing_input)
    character_input=Input(shape=(None,52,),name='char_input')
    embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
    dropout= Dropout(0.5)(embed_char_out)
    conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
    maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    char = Dropout(0.5)(char)
    output = concatenate([words, casing,char])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
    model = Model(inputs=[words_input, casing_input,character_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
    model.summary()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):    
        print("Epoch %d/%d"%(epoch,epochs))
        a = Progbar(len(train_batch_len))
        for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
            labels, tokens, casing,char = batch       
            model.train_on_batch([tokens, casing,char], labels)
            a.update(i)
        a.update(i+1)
        print(' ')

        # Calculate training loss for the epoch
        train_epoch_loss = 0
        train_batches = 0
        for batch in iterate_minibatches(train_batch, train_batch_len):
            labels, tokens, casing, char = batch
            loss = model.test_on_batch([tokens, casing, char], labels)
            train_epoch_loss += loss
            train_batches += 1
        train_epoch_loss /= train_batches
        train_losses.append(train_epoch_loss)
        
        # Calculate validation loss for the epoch
        val_epoch_loss = 0
        val_batches = 0
        for batch in iterate_minibatches(dev_batch, dev_batch_len):
            labels, tokens, casing, char = batch
            loss = model.test_on_batch([tokens, casing, char], labels)
            val_epoch_loss += loss
            val_batches += 1
        val_epoch_loss /= val_batches
        val_losses.append(val_epoch_loss)

    # Create the learning curve graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    if finetune:
        # Fine-tuning on the finetune_train_batch
        finetune_epochs = 5  # Number of epochs for fine-tuning
        finetune_train_losses = []
        finetune_val_losses = []

        finetune_sw_train_set, finetune_sw_dev_set = train_test_split(
            finetune_sw_set, test_size=0.2, random_state=42
        )

        finetune_train_batch,finetune_train_batch_len = createBatches(finetune_sw_train_set)
        finetune_dev_batch,finetune_dev_batch_len = createBatches(finetune_sw_dev_set)

        for epoch in range(finetune_epochs):
            print("Fine-tuning Epoch %d/%d" % (epoch, finetune_epochs))
            a = Progbar(len(finetune_train_batch_len))
            
            for i, batch in enumerate(iterate_minibatches(finetune_train_batch, finetune_train_batch_len)):
                labels, tokens, casing, char = batch
                model.train_on_batch([tokens, casing, char], labels)
                a.update(i)
            
            a.update(i + 1)
            print(' ')
            
            # Calculate fine-tuning train loss for the epoch
            finetune_train_epoch_loss = 0
            finetune_train_batches = 0
            
            for batch in iterate_minibatches(finetune_train_batch, finetune_train_batch_len):
                labels, tokens, casing, char = batch
                loss = model.test_on_batch([tokens, casing, char], labels)
                finetune_train_epoch_loss += loss
                finetune_train_batches += 1
            
            finetune_train_epoch_loss /= finetune_train_batches
            finetune_train_losses.append(finetune_train_epoch_loss)
            
            # Calculate fine-tuning validation loss for the epoch
            finetune_val_epoch_loss = 0
            finetune_val_batches = 0
            
            for batch in iterate_minibatches(finetune_dev_batch, finetune_dev_batch_len):
                labels, tokens, casing, char = batch
                loss = model.test_on_batch([tokens, casing, char], labels)
                finetune_val_epoch_loss += loss
                finetune_val_batches += 1
            
            finetune_val_epoch_loss /= finetune_val_batches
            finetune_val_losses.append(finetune_val_epoch_loss)

        # Create the learning curve graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(finetune_train_losses) + 1), finetune_train_losses, label='Finetune Training Loss')
        plt.plot(range(1, len(finetune_val_losses) + 1), finetune_val_losses, label='Finetune Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()


    return tag_dataset(test_sw_set, model)
