
###################################################################################################

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

###################################################################################################

def load_fashion_mnist():
    
    (X, y), (X_test, y_test) = fashion_mnist.load_data()
    
    # add 'channel' dimension to images

    X = np.expand_dims(X, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    # normalize images

    X = X / 255
    X_test = X_test / 255

    # transform labels into one-hot vectors

    y = np.eye(10)[y]
    y_test = np.eye(10)[y_test]
    
    # split 'X' and 'y' into 'train' and 'validation' part
    
    split_opts = dict(
        stratify=y.argmax(axis=1),
        test_size=10000, random_state=192)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, **split_opts)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

###################################################################################################

class Augmenter:
    
    def __init__(self,
                 
                 # basic augmentations
                 x_flip=False, dx=0, dy=0,
                 
                 # mixup augmentation
                 mixup=False, alpha=0.4,
                 
                 # random erasing augmentation
                 re=False, p=0.5, sl=0.02, sh=0.4, r1=3/10, r2=10/3):
        
        # footwear class ids
        self.non_flip = {5, 7, 9} 
        
        self.x_flip = x_flip
        
        self.dx = dx
        self.dy = dy
        
        self.mixup = mixup
        self.alpha = alpha
        
        self.re = re
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
    
    def _x_flip(self, img, img_class):
        
        if not self.x_flip:
            return
        
        if img_class in self.non_flip:
            return
        
        if np.random.uniform() > 0.5:
            img[:, :, :] = img[:, ::-1, :]
            
    def _x_shift(self, img):
    
        s = np.random.choice(2 * self.dx + 1)
        s -= self.dx

        if s > 0:
            img[:, s:, :] = img[:, :-s, :]
            img[:, :s, :] = 0
        elif s < 0:
            img[:, :s, :] = img[:, -s:, :]
            img[:, s:, :] = 0
            
    def _y_shift(self, img):
    
        s = np.random.choice(2 * self.dy + 1)
        s -= self.dy

        if s > 0:
            img[s:, :, :] = img[:-s, :, :]
            img[:s, :, :] = 0
        elif s < 0:
            img[:s, :, :] = img[-s:, :, :]
            img[s:, :, :] = 0
    
    def _batch_mixup(self, X_batch, y_batch):
        
        batch_size = X_batch.shape[0]
        perm = np.random.permutation(batch_size)
        
        w = np.random.beta(self.alpha, self.alpha, batch_size)
        w = np.c_[w, 1 - w].max(axis=1)
        
        wx = w.reshape(batch_size, 1, 1, 1)
        X_batch = wx * X_batch + (1 - wx) * X_batch[perm]
        
        wy = w.reshape(batch_size, 1)
        y_batch = wy * y_batch + (1 - wy) * y_batch[perm]
        
        return X_batch, y_batch
    
    def _random_erasing(self, img):
    
        if np.random.uniform() > self.p:
            return

        h, w, ch = img.shape
        s = h * w

        while True:

            se = np.random.uniform(self.sl, self.sh) * s
            re = np.random.uniform(self.r1, self.r2)

            he = int(np.sqrt(se * re) + 0.5)
            we = int(np.sqrt(se / re) + 0.5)

            xe = np.random.randint(w)
            ye = np.random.randint(h)

            if (xe + we <= w) and (ye + he <= h):
                
                noise = np.random.uniform(size=(he, we, ch))
                img[ye:ye+he, xe:xe+we] = noise
                
                return
    
    def _data_generator(self, X, y, batch_size, shuffle):
        
        m = X.shape[0]
        steps = int(np.ceil(m / batch_size))

        while True:

            perm = np.arange(m)
            
            if shuffle:
                perm = np.random.permutation(m)
            
            X_perm, y_perm = X[perm], y[perm]

            for i in range(steps):

                begin = i * batch_size
                end = min(m, begin + batch_size)

                X_batch = X_perm[begin:end]
                y_batch = y_perm[begin:end]

                for xi, yi in zip(X_batch, y_batch):
                    
                    self._x_flip(xi, yi.argmax())
                    
                    self._x_shift(xi)
                    self._y_shift(xi)

                if self.mixup:
                    X_batch, y_batch = self._batch_mixup(X_batch, y_batch)
                    
                if self.re:
                    for xi in X_batch:
                        self._random_erasing(xi)
                    
                yield X_batch, y_batch
                
    def flow(self, X, y, batch_size, shuffle=True):
        return self._data_generator(X, y, batch_size, shuffle)

###################################################################################################

