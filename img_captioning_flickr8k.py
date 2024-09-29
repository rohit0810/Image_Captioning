import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from PIL import Image
import nltk
import string
import warnings
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from nltk.translate.bleu_score import corpus_bleu

warnings.filterwarnings("ignore")
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

IMG_H = 224
IMG_W = 224
BATCH_SIZE = 100
BUFFER_SIZE = 1000

# Set your dataset paths
IMG_FILE_PATH = "Flickr8k_Dataset/Images/"
CAP_TEXT_PATH = "Flickr8k_Dataset/captions.txt"
TRAIN_TXT_PATH = "Flickr8k_Dataset/train_images.txt"
TEST_TXT_PATH = "Flickr8k_Dataset/test_images.txt"
VAL_TXT_PATH = "Flickr8k_Dataset/val_images.txt"
CLEAN_CAP_TEXT_PATH = "Flickr8k_Dataset/clean_captions.txt"

def cnn_model():
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-1].output)
    return model

def load_img(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img, image_path

def load_doc(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def load_set(text_file_path):
    doc = load_doc(text_file_path)
    dataset = set()
    for line in doc.strip().split('\n'):
        identifier = line.split('.')[0]
        dataset.add(identifier)
    return dataset

def img_name_2_path(image_name, img_file_path=IMG_FILE_PATH, ext=".jpg"):
    return img_file_path + str(image_name) + ext

def load_img_dataset(txt_path, batch_size=BATCH_SIZE):
    img_name_vector = load_set(txt_path)
    img_path_list = list(map(img_name_2_path, img_name_vector))
    image_dataset = tf.data.Dataset.from_tensor_slices(img_path_list)
    image_dataset = image_dataset.map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
    return image_dataset

image_features_extract_model = cnn_model()

def extract_features(image_dataset):
    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

if not os.path.exists('features_extracted.txt'):
    image_train_dataset = load_img_dataset(TRAIN_TXT_PATH)
    extract_features(image_train_dataset)
    image_test_dataset = load_img_dataset(TEST_TXT_PATH)
    extract_features(image_test_dataset)
    with open('features_extracted.txt', 'w') as f:
        f.write('Features extracted')
else:
    print("Features already extracted")

def clean_cap(caption):
    cap = ''.join([ch for ch in caption if ch not in string.punctuation])
    cap = cap.split()
    cap = [word.lower() for word in cap]
    cap = [word for word in cap if len(word) > 1]
    cap = [word for word in cap if word.isalpha()]
    lemmatizer = nltk.WordNetLemmatizer()
    cap = [lemmatizer.lemmatize(word) for word in cap]
    return ' '.join(cap)

def load_cap(caption_txt_path):
    with open(caption_txt_path, 'r', encoding='utf-8') as f:
        captions_list = f.readlines()
    mapping = {}
    for line in captions_list:
        tokens = line.strip().split('\t')
        if len(tokens) < 2:
            continue
        image_name = tokens[0].split('.')[0]
        image_caption = clean_cap(tokens[1])
        image_caption = 'startofseq ' + image_caption + ' endofseq'
        mapping.setdefault(image_name, []).append(image_caption)
    return mapping

def save_captions(mapping, filename):
    lines = []
    for key, cap_list in mapping.items():
        for cap in cap_list:
            lines.append(key + ' ' + cap)
    data = '\n'.join(lines)
    with open(filename, 'w') as file:
        file.write(data)

def load_clean_cap(caption_txt_path, dataset):
    doc = load_doc(caption_txt_path)
    clean_captions = {}
    for line in doc.strip().split('\n'):
        tokens = line.split()
        image_name, image_cap = tokens[0], " ".join(tokens[1:])
        if image_name in dataset:
            clean_captions.setdefault(image_name, []).append(image_cap)
    return clean_captions

def max_len(captions_dict):
    return max(len(caption.split()) for captions in captions_dict.values() for caption in captions)

def create_tokenizer(captions_dict, top_k=None):
    captions_list = [caption for captions in captions_dict.values() for caption in captions]
    tokenizer = Tokenizer(num_words=top_k, oov_token="<unk>")
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    tokenizer.fit_on_texts(captions_list)
    return tokenizer

def tokenize_cap(tokenizer, captions_dict, pad_len):
    pad_caps_dict = {}
    for img_name, captions in captions_dict.items():
        seqs = tokenizer.texts_to_sequences(captions)
        pad_seqs = pad_sequences(seqs, maxlen=pad_len, padding='post')
        pad_caps_dict[img_name] = pad_seqs
    return pad_caps_dict

def path_cap_list(img_names_set, tokenizer, captions_dict, pad_len):
    tokenized_caps_dict = tokenize_cap(tokenizer, captions_dict, pad_len)
    image_name_list = sorted(img_names_set)
    capt_list = []
    img_path_list = []
    for name in image_name_list:
        caps = tokenized_caps_dict[name]
        img_path = img_name_2_path(name)
        for cap in caps:
            capt_list.append(cap)
            img_path_list.append(img_path)
    return img_path_list, capt_list

def load_npy(image_path, cap):
    img_tensor = np.load(image_path.decode('utf-8') + '.npy')
    return img_tensor, cap

def create_dataset(img_path_list, cap_list):
    dataset = tf.data.Dataset.from_tensor_slices((img_path_list, cap_list))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_npy, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

cap_dict = load_cap(CAP_TEXT_PATH)
save_captions(cap_dict, CLEAN_CAP_TEXT_PATH)
MAX_CAP_LEN = max_len(cap_dict)
print("Maximum caption length:", MAX_CAP_LEN)

train_img_names = sorted(load_set(TRAIN_TXT_PATH))
train_img_cap = load_clean_cap(CLEAN_CAP_TEXT_PATH, train_img_names)
tokenizer = create_tokenizer(train_img_cap)
VOCAB_SIZE = len(tokenizer.word_index) + 1
print("Vocabulary size:", VOCAB_SIZE)

img_name_train, caption_train = path_cap_list(train_img_names, tokenizer, train_img_cap, MAX_CAP_LEN)
train_dataset = create_dataset(img_name_train, caption_train)

test_img_names = sorted(load_set(TEST_TXT_PATH))
test_img_cap = load_clean_cap(CLEAN_CAP_TEXT_PATH, test_img_names)
img_name_test, caption_test = path_cap_list(test_img_names, tokenizer, test_img_cap, MAX_CAP_LEN)
test_dataset = create_dataset(img_name_test, caption_test)

embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(train_img_names) // BATCH_SIZE
EPOCHS = 20
features_shape = 512
attention_features_shape = 49

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)
    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)
    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)

loss_plot = []

@tf.function
def train_step(img_tensor, target):
    loss = 0
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['startofseq']] * target.shape[0], 1)
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            dec_input = tf.expand_dims(target[:, i], 1)
    total_loss = loss / int(target.shape[1])
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss

EPOCHS = 20

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(train_dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
        if batch % 100 == 0:
            average_batch_loss = batch_loss.numpy()/int(target.shape[1])
            print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
    loss_plot.append(total_loss / num_steps)
    if epoch % 5 == 0:
        ckpt_manager.save()
    print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()

encoder.save_weights("encoder.h5")
decoder.save_weights("decoder.h5")

def evaluate(image):
    attention_plot = np.zeros((MAX_CAP_LEN, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_img(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['startofseq']], 0)
    result = []
    for i in range(MAX_CAP_LEN):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tokenizer.index_word.get(predicted_id, '<unk>')
        result.append(predicted_word)
        if predicted_word == 'endofseq':
            break
        dec_input = tf.expand_dims([predicted_id], 0)
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))
    fig = plt.figure(figsize=(10, 10))
    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = np.ceil(np.sqrt(len_result))
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    plt.tight_layout()
    plt.show()

# Compute BLEU scores
references = []
hypotheses = []

for image_name in tqdm(test_img_names):
    img_path = img_name_2_path(image_name)
    real_captions = test_img_cap[image_name]
    result, _ = evaluate(img_path)
    result = result[:-1] if result[-1] == 'endofseq' else result
    hypotheses.append(result)
    refs = [cap.split()[1:-1] for cap in real_captions]
    references.append(refs)

print('BLEU-1:', corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0)))
print('BLEU-2:', corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3:', corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0)))
print('BLEU-4:', corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)))

# Test on your own image
img_path = str(input("Enter the image path: "))
result, attention_plot = evaluate(img_path)
print('Prediction Caption:', ' '.join(result))
plot_attention(img_path, result, attention_plot)
Image.open(img_path)
