import argparse, tensorflow as tf

def main():
 p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch-size",type=int,default=128); a=p.parse_args()
 (xtr,ytr),(xte,yte)=tf.keras.datasets.cifar10.load_data(); xtr=xtr.astype("float32")/255.; xte=xte.astype("float32")/255.; ytr=ytr.squeeze(); yte=yte.squeeze()
 aug=tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),tf.keras.layers.RandomTranslation(.08,.08)])
 model=tf.keras.Sequential([tf.keras.layers.Input((32,32,3)),aug,tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),tf.keras.layers.MaxPool2D(),tf.keras.layers.Conv2D(64,3,padding="same",activation="relu"),tf.keras.layers.MaxPool2D(),tf.keras.layers.Conv2D(128,3,activation="relu"),tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dropout(.25),tf.keras.layers.Dense(10)])
 model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
 model.fit(xtr,ytr,batch_size=a.batch_size,epochs=a.epochs,validation_split=.1,callbacks=[tf.keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)])
 print("Test metrics:",model.evaluate(xte,yte,verbose=0)); model.save("cifar10_keras.keras")
if __name__=="__main__":main()
