import os, argparse, numpy as np, tensorflow as tf, librosa, json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_wav(p, sr=16000, duration=2.5):
    y, _ = librosa.load(p, sr=sr, mono=True)
    max_len = int(sr*duration)
    if len(y) < max_len: y = np.pad(y, (0, max_len - len(y)))
    else: y = y[:max_len]
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min())/(mel_db.max()-mel_db.min()+1e-9)
    return mel_db.astype('float32')

def gather(data_dir, classes):
    X=[]; y=[]
    for c in classes:
        folder = os.path.join(data_dir, c)
        if not os.path.isdir(folder): continue
        for fn in os.listdir(folder):
            if fn.lower().endswith(('.wav','.mp3','.mp4','.flac','.ogg','.m4a')):
                p = os.path.join(folder, fn)
                mel = load_wav(p)
                X.append(mel)
                y.append(c)
    return np.array(X), np.array(y)

def build_model(input_shape, n_classes):
    inputs = tf.keras.Input(shape=input_shape+(1,))
    x = tf.keras.layers.Conv2D(16,3,padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(32,3,padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main(data_dir='datasets', classes=None, epochs=8, out='models/vehicle_classifier.h5'):
    if classes is None:
        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))])
    X, y = gather(data_dir, classes)
    if len(X)==0:
        print('No data found'); return
    X = X[..., np.newaxis]
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)
    model = build_model(X.shape[1:3], len(le.classes_))
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save(out)
    with open(os.path.join(os.path.dirname(out),'label_mapping.json'),'w') as f:
        json.dump({'classes': list(le.classes_)}, f)
    print('Saved model to', out)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--data', default='../datasets'); p.add_argument('--epochs', type=int, default=6); p.add_argument('--out', default='../models/vehicle_classifier.h5')
    args=p.parse_args(); main(data_dir=args.data, epochs=args.epochs, out=args.out)
