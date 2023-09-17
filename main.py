import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton
from gui import Ui_MainWindow
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import PIL
import numpy as np


class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)
        # -------------------------------
        # add row and colum count
        self.uic.tableWidget.setColumnCount(1)
        self.uic.tableWidget.setRowCount(1)
        # set column width
        for m in range(0, 1):
            self.uic.tableWidget.setColumnWidth(m, 430)
        # set row height
        for n in range(1):
            self.uic.tableWidget.setRowHeight(n, 190)
        # khai bao nut chay
        self.uic.btn_predict.clicked.connect(self.insert_image)

    def insert_image(self):
        link = QFileDialog.getOpenFileName(filter="*.jpg *.jpeg *.png *.pgm")
        img = QLabel()
        img.setStyleSheet("border-image: url({});".format((link[0])))
        self.uic.tableWidget.setCellWidget(0, 0, img)
        # predict
        path_model = 'model.json'
        json_file = open(path_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights('model.h5')
        # print("Loaded model from disk")
        batch_size = 32
        img_height = 128
        img_width = 128
        label_mapping = [
            'Nhà Học C1',  # index 0
            'Khoa Khoa Học Chính Trị',  # index 1
            'Trường Công Nghệ Thông Tin',  # index 2
            'Trường Bách Khoa',  # index 3
            'Khoa Dự Bị Dân Tộc',  # index 4
            'Trường Kinh Tế',  # index 5
            'Trường Nông Nghiệp',  # index 6
            'Khoa Sư Phạm',  # index 7
            'Khoa Khoa Học Tự Nhiên'  # index 8
        ]
        img = tf.keras.utils.load_img(
            link[0], target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = loaded_model.predict(img_array)

        predictions = loaded_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        self.uic.label_predict.setText("{} - {:.2f} %".format(label_mapping[np.argmax(score)], 100 * np.max(score)))

    def show(self):
        # command to run
        self.main_win.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
