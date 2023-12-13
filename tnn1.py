import pandas as pd
data=pd.read_csv('C:/Users/Ahas Kaushik/OneDrive/Documents/AI and ML/Data_sets/survey_lung_cancer_SVM.csv')
data.head(10)
x=data.drop(['LUNG_CANCER'],axis=1)
y=data['LUNG_CANCER']
import tensorflow as tf
model=tf.keras.models.Sequential([
tf.keras.layers.Dense(512,input_dim=15,activation='relu'),
tf.keras.layers.Dense(512,activation='relu'),
tf.keras.layers.Dense(1,activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
from tensorflow.keras.callbacks import TensorBoard
log='C:/Users/Ahas Kaushik/OneDrive/Desktop/TF_NN_PD_1'
callbacks=[TensorBoard(
histogram_freq=1,
log_dir=log,
write_images=True,
write_graph=True,
update_freq='epoch',
profile_batch=2,
embeddings_freq=1)]
model.fit(x,y,epochs=5,batch_size=10,callbacks=callbacks)
model.save('m1.hs')
_,acc=model.evaluate(x,y)
print("The accuracy of the model is:%.2f"%acc)
