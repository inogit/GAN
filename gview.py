import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# エポック数(学習回数)
# gan_mnist.pyのEPOCHSと合わせておく
EPOCHS = 100
#EPOCHS = 300
#EPOCHS = 5000
IMG_WH = 28
#IMG_WH = 64

# 画像の表示を行う
def display(fname):
    # 画像ファイルの読み込み
    img = Image.open(fname)
    #表示
    #img.show()

# 指定されたエポックにて、生成された画像の表示と保存を行う
#　ep: エポック
# 画像ファイルの出力と保存を行う
def result(ep):
    with open('save_gimage.pkl', 'rb') as f:
        # 生成された画像を全件読み込む
        save_gimage = pkl.load(f)
        # Generatorにて1度に25枚の画像を生成するため、表示エリアに5x5の
        # パネルを準備(axes)する。
        fig, axes = plt.subplots(5, 5, figsize=(IMG_WH, IMG_WH))

        # zipにて表示する画像 (save_gimage)と表示位置(axes.flatten)を
        # 対で取得し順に表示(imshow)する。
        for img, ax in zip(save_gimage[ep], axes.flatten()):
            ax.xaxis.set_visible(False) #xスケール非表示
            ax.yaxis.set_visible(False) #yスケール非表示
            #画像はWidth=28, Height=28のため28x28にリシェイプし、
            #グレイスケール指定にて画像化する
            ax.imshow(img.reshape((IMG_WH,IMG_WH)), cmap='gray')

        # epが-1の時は、学習最後の状態で生成された画像を対象とする
        if ep == -1:
            ep = EPOCHS - 1
        
        # ファイル名の編集
            fname='GANResult_'+format(ep, '03d')+'.png'
            print('file='+fname)
        # ファイル出力
            plt.savefig(fname)
        # ファイル表示
            display(fname)

# 10エポック毎の生成画像を表示
# 縦方向は10エポック単位、横方向は当該エポックで生成された25枚の
# 画像のうち、最初の5枚を表示
def history(): 
    with open('save_gimage.pkl', 'rb') as f:
        # エポック毎に生成された画像を全件読み込む
        save_gimage = pkl.load(f)
        # 10エポック毎に5枚の生成画像を表示するエリアを設定する
        # 画像は28x28ピクセル。
        fig, axes = plt.subplots(int(EPOCHS/10), 5, figsize=(IMG_WH, IMG_WH))
        # 10エポック単位に生成画像と表示位置を順に取得しながら処理を行う(縦方向)
        for save_gimage,  ax_row in zip(save_gimage[::10], axes):
            # 取り出したエポックには25枚の画像が含まれているため、先頭から5枚の画像を
            # 順に取り出しパネル(axes)に並べる（横方向）
            for img, ax in zip(save_gimage[::1], ax_row):
                # 画像はWidth=28, Height=28のため28x28にリシェイプし、
                # グレイスケール指定にて画像化する
                ax.imshow(img.reshape((IMG_WH,IMG_WH)), cmap='gray')
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

    fname='GANHistory.png'
    print('file='+fname)
    plt.savefig(fname)
    display(fname)

# 学習の経過をグラフ表示にて確認する
def loss():
    with open('save_loss.pkl', 'rb') as f:
        save_loss = pkl.load(f)
        
        # 学習損失の可視化
        fig, ax = plt.subplots()
        loss = np.array(save_loss)
        # 転置しDiscriminatorのロスを0番目の要素から、
        # Geneeatorのロスは1番目の要素から取得する。
        plt.plot(loss.T[0], label='Discriminator')
        plt.plot(loss.T[1], label='Generator')
        plt.title('Loss')
        plt.legend()
        fname='GANLoss.png'
        print('file='+fname)
        plt.savefig(fname)
        display(fname)

    
if __name__ == '__main__':
    args = sys.argv
    ep = 0

    if len(args) == 1:
        result(-1)
    elif args[1] == 'h':
        history()
    elif args[1] == 'l':
        loss()
    else:
        result(int(args[1]))
