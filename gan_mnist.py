
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


#（1）必要なライブラリのインポートとMNISTデータセットの取得
import tensorflow as tf
# MNIST Data取得に向けた準備
from tensorflow.examples.tutorials.mnist import input_data
import  datetime as dt
import numpy as np
# 学習過程で生成される画像を保管するpkl形式ファイル
import pickle as pkl
# ハイパーパラメータを指定した活性化関数を指定する際に使用します
from functools import partial


# In[3]:


# 精度向上は下記パラメータを調整し行う
# エポック数(学習回数)、EPOCHSを変更した場合はビューアプログラム(gview.py)にて
# 定義しているEPOCHSも同じ値を指定してください。
EPOCHS = 100
# バッチサイズ
BATCH_SIZE=100
# 学習率
LEARNING_RATE = 0.001
# 活性化関数のハイパーパラメータ設定
ALPHA = 0.01


# In[4]:


#（2）生成モデル（Generator）を作る関数の定義
def generator(randomData, alpha, reuse=False):
      with tf.variable_scope('GAN/generator', reuse=reuse):
        # 隠れ層
        h1 = tf.layers.dense(randomData, 256,
                      activation=partial(tf.nn.leaky_relu, alpha=alpha))
        # 出力層
        o1 = tf.layers.dense(h1, 784, activation=None)
        # 活性化関数 tanh
        img = tf.tanh(o1) 

        return img


# In[5]:


#（3）識別モデル（Discriminator）を作る関数の定義
def discriminator(img, alpha, reuse=False):
    with tf.variable_scope('GAN/discriminator', reuse=reuse):
        # 隠れ層
        h1 = tf.layers.dense(img, 128, 
                      activation=partial(tf.nn.leaky_relu, alpha=alpha))
        # 出力層
        D_logits = tf.layers.dense(h1, 1, activation=None)
        # 活性化関数
        D = tf.nn.sigmoid(D_logits)

        return D, D_logits


# In[6]:


#ino
#for GeForce GTX 1080 Ti
config = tf.ConfigProto() #ino
config.gpu_options.allow_growth = True #ino


# In[8]:


if __name__ == '__main__':
    # 処理開始時刻の取得
    tstamp_s = dt.datetime.now().strftime("%H:%M:%S")
    # MNISTデータセットの取得
    mnist = input_data.read_data_sets('./MNIST_DataSet')

    # プレースホルダー
    # 本物画像データ784次元(28x28ピクセル)をバッチサイズ分保管するプレースホルダ(ph_realData)を準備する
    ph_realData = tf.placeholder(tf.float32, (BATCH_SIZE, 784))
    # 100次元の一様性乱数を保管するプレースホルダ(ph_randomData)を準備する、
    # 確保するサイズとして、学習時はバッチサイズの100件、各エポックでの画像生成は25件と、
    # 動的に変わるため、Noneを指定し実行時にサイズを決定するようにする
    ph_randomData = tf.placeholder(tf.float32, (None,100))

    # 一様性乱数を与えて画像を生成
    gimage = generator(ph_randomData, ALPHA)
    # 本物の画像を与えて判定結果を取得
    real_D, real_D_logits = discriminator(ph_realData, ALPHA)
    # 生成画像を与えて判定結果を取得
    fake_D, fake_D_logits = discriminator(gimage, ALPHA, reuse = True)

    #（3）損失関数の実装
    # 本物画像（ラベル＝１）との誤差をクロスエントロピーの平均として取得
    d_real_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits, labels=tf.ones_like(real_D))
    loss_real = tf.reduce_mean(d_real_xentropy)
    # 生成画像（ラベル=0)との誤差をクロスエントロピーの平均として取得
    d_fake_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits, labels=tf.zeros_like(fake_D))
    loss_fake = tf.reduce_mean(d_fake_xentropy)
    # Discriminatorの誤差は、本物画像、生成画像における誤差を合計した値となる
    d_loss = loss_real + loss_fake
    # Generatorの誤差を取得
    g_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits, labels=tf.ones_like(fake_D))
    g_loss = tf.reduce_mean(g_xentropy)

    # 学習によって最適化を行うパラメータ(重み、バイアス)をtf.trainable_variablesから
    # 一括して取得する。取得の際にDiscriminator用(d_training_parameter)、
    # Generator用(g_training_parameter)と取り分けて、それぞれのネットワークを
    # 最適化していく必要があるため、ネットワーク定義時に指定したスコープの名前を指定して、
    # 取り分けを行う。
    # discriminatorの最適化を行う学習パラメータを取得（一旦、trainVarに取り分けてから格納）
    d_training_parameter = [trainVar for trainVar in tf.trainable_variables()
                          if 'GAN/discriminator/' in trainVar.name]
    # generatorの最適化を行う学習パラメータを取得（一旦、trainVarに取り分けてから格納）
    g_training_parameter = [trainVar for trainVar in tf.trainable_variables()
                          if 'GAN/generator/' in trainVar.name]
    
    # オプティマイザ(AdamOptimizer)にて学習パラメータの最適化を行う
    # 一括取得したDiscriminatorのパラメータ更新
    d_optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
                d_loss, var_list=d_training_parameter)
    # 一括取得したGeneratorのパラメータ更新
    g_optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
                g_loss, var_list=g_training_parameter)

    batch = mnist.train.next_batch(BATCH_SIZE)

    # 途中経過の保存する変数定義
    save_gimage=[]
    save_loss=[]
    
    counter=0
    #（5）学習処理の実装
    with tf.Session(config=config) as sess:
      # 変数の初期化
      sess.run(tf.global_variables_initializer()) 

      # EPOCHS数ぶん繰り返す
      for e in range(EPOCHS):
        # バッチサイズ100
        for i in range(mnist.train.num_examples//BATCH_SIZE):
          #####print('********{}/{} '.format(i, mnist.train.num_examples//BATCH_SIZE))
          batch = mnist.train.next_batch(BATCH_SIZE)
          batch_images = batch[0].reshape((BATCH_SIZE, 784))
          #print(batch_images)  
          # generatorにて活性化関数tanhを使用したためレンジを合わせる
          batch_images = batch_images * 2 - 1
          # generatorに渡す一様分布のランダムノイズを生成
          # 値は-1〜1まで、サイズはbatch_size * 100
          batch_z = np.random.uniform(-1,1,size=(BATCH_SIZE, 100))
          # 最適化計算・パラメータ更新を行う
          #####print('長さ batch={}'.format(len(batch_images)))
          counter += 1
          # Discriminatorの最適化に使うデータ群をfeed_dictで与える
          sess.run(d_optimize, feed_dict = {ph_realData:batch_images, ph_randomData: batch_z})
          # Generatorの最適化と最適化に使うデータ群をfeed_dictで与える
          sess.run(g_optimize, feed_dict = {ph_randomData: batch_z})
 
        # トレーニングのロスを記録
        train_loss_d = sess.run(d_loss, {ph_randomData: batch_z, ph_realData:batch_images})
        # evalはgのロス(g_loss)を出力する命令
        train_loss_g = g_loss.eval({ph_randomData: batch_z})

        # 学習過程の表示
        print('{0} Epoch={1}/{2}, DLoss={3:.4F}, GLoss={4:.4F}'.format(
              dt.datetime.now().strftime("%H:%M:%S"),e+1,
              EPOCHS,train_loss_d,train_loss_g))

        # lossを格納するためのリストに追加する
        # train_loss_d, train_loss_gをセットでリスト追加し、
        # あとで可視化できるようにする
        save_loss.append((train_loss_d, train_loss_g))

        # 学習途中の生成モデルで画像を生成して保存する
        # 一様性乱数データを25個生成して、そのデータを使って画像を生成し保存する。
        randomData = np.random.uniform(-1, 1, size=(25, 100))
        # gen_samplesに現時点のモデルで作ったデータを読ませておく
        # ノイズ、サイズ、ユニット数(128)、reuseは状態保持、
        # データはsample_zとしてfeed_dictに指定
        gen_samples = sess.run(generator(ph_randomData, ALPHA, True),
                              feed_dict={ph_randomData: randomData})
        save_gimage.append(gen_samples)
        
    # pkl形式で生成画像を保存
    with open('save_gimage.pkl', 'wb') as f:
        pkl.dump(save_gimage, f)

    # 各エポックで得た損失関数の値を保存
    with open('save_loss.pkl', 'wb') as f:
        pkl.dump(save_loss, f)

    # 処理終了時刻の取得
    tstamp_e = dt.datetime.now().strftime("%H:%M:%S")
    
    time1 = dt.datetime.strptime(tstamp_s, "%H:%M:%S") 
    time2 = dt.datetime.strptime(tstamp_e, "%H:%M:%S") 

    # 処理時間を表示
    print("開始: {0}、終了:{1}、処理時間:{2}".format(tstamp_s, tstamp_e, (time2 - time1)))
    print('counter={}'.format(counter))

