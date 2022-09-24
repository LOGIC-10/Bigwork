import csv
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2)

predict_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

train_generator = datagen.flow_from_directory(
        # 种类文件夹的路径
        '/Users/logic/Downloads/Bigwork/characterData',
        # 目标图片大小
        target_size=(20,20),
        # 目标颜色模式
        color_mode="grayscale",
        # 种类名字
        classes=None,
        # 种类模式：分类
        class_mode='categorical',
        # batch_size
        batch_size=2,
        # shuffle
        shuffle=True,
        # seed
        seed=None,
        # # 变换后的保存路径
        save_to_dir="/Users/logic/Downloads/Bigwork",
        # 保存的前缀
        save_prefix="c",
        # 保存的格式
        save_format="png",
        # 验证分离的设置
        subset="training"
)

val_generator = datagen.flow_from_directory(
        # 种类文件夹的路径
        '/Users/logic/Downloads/Bigwork/characterData',
        # 目标图片大小
        target_size=(20,20),
        # 目标颜色模式
        color_mode="grayscale",
        # 种类名字
        classes=None,
        # 种类模式：分类
        class_mode='categorical',
        # batch_size
        batch_size=2,
        # shuffle
        shuffle=True,
        # seed
        seed=None,
        # # 变换后的保存路径
        save_to_dir="/Users/logic/Downloads/Bigwork",
        # 保存的前缀
        save_prefix="c",
        # 保存的格式
        save_format="png",
        # 验证分离的设置
        subset="validation"
)

model = tf.keras.Sequential([
    #(-1,20,20,1)->(-1,20,20,32)
    tf.keras.layers.Conv2D(input_shape=(20, 20, 1),filters=32,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,20,20,32)->(-1,10,10,32)
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,10,10,32)->(-1,10,10,64)
    tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),
    #(-1,10,10,64)->(-1,5,5,64)
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,5,5,64)->(-1,5,5,64)
    tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),
    #(-1,5,5,64)->(-1,5*5*64)
    tf.keras.layers.Flatten(),
    #(-1,5*5*64)->(-1,256)
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    #(-1,256)->(-1,31)
    tf.keras.layers.Dense(31, activation=tf.nn.softmax)
])

print(model.summary())

lr = 0.0001
epochs = 5

# 编译模型
model.compile(optimizer=tf.optimizers.Adam(lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 拟合数据
history = model.fit(
            train_generator,
            steps_per_epoch=None,
            epochs=9,
            validation_data=val_generator,
            validation_steps=None)


plt.plot(history.epoch,history.history['loss'], label='train_loss')
plt.plot(history.epoch,history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.epoch,history.history['accuracy'], label='train_accuracy')
plt.plot(history.epoch,history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# 预测
label_index = list(train_generator.class_indices)
predict_generator = predict_datagen.flow_from_directory(
        # 种类文件夹的路径
        '/Users/logic/Downloads/Bigwork/predict',
        # 目标图片大小
        target_size=(20,20),
        # 目标颜色模式
        color_mode="grayscale",
        # 种类名字
        classes=None,
        # 种类模式：分类
        class_mode='categorical',
        # batch_size
        batch_size=1,
        # shuffle
        shuffle=False
)

# 写入CSV
file = 'results.csv'
with open(file, 'w', encoding='utf_8_sig', newline='') as f:
    print('\n--- printing results to file {}'.format(file))
    writer = csv.writer(f)
    writer.writerow(['picture', 'label'])
    predict_path = r'D:\pythonProject\base\predict\1'
    predict_dic = {}
    for picture in os.listdir(predict_path):
        y = model.predict(predict_generator.next()[0], batch_size=1)
        predict_dic[picture] = label_index[y.argmax()]
    picture_list = os.listdir(predict_path)
    picture_list.sort(key=lambda x: int(x[:-4]))
    for picture in picture_list:
        writer.writerow([picture, predict_dic[picture]])

# 计算准确率
ground_truth_csv = r'ground_truth.csv'
ground_truth_dic = {}
with open(ground_truth_csv, 'r') as f:
    print('\n--- reading ground truths from file {}'.format(ground_truth_csv))
    reader = csv.reader(f)
    for line in reader:
        ground_truth_dic[line[0][:-2]] = line[0][-1:]

    i = 0
    for picture in picture_list:
        if predict_dic[picture] == ground_truth_dic[picture]:
            i += 1

    print('predict accuracy: ', i / len(picture_list))
