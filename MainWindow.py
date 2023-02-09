import numpy as np
import pandas as pd
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QLabel,QWidget,QSplitter,QLineEdit,QComboBox, QPushButton,QMessageBox
from PySide6.QtCore import Qt
from BCDNeuralNetwork import BCDNeuralNetwork

class MainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWindowTitle("BREAST CANCER PREDICTION")

        self.net=BCDNeuralNetwork()
        self.accuracy=self.net.best_results[1]

        age_label = QLabel("Age")
        race_label = QLabel("Race")
        ms_label = QLabel("Martial Status")
        tstage_label = QLabel("T Stage")
        nstage_label = QLabel("N Stage")
        sixstage_label = QLabel("6th Stage")
        grade_label = QLabel("Grade")
        astage_label = QLabel("A Stage")
        ts_label = QLabel("Tumor Size")
        es_label = QLabel("Estrogen Status")
        ps_label = QLabel("Progesterone Status")
        rne_label = QLabel("Regional Node Examined")
        rnp_label = QLabel("Regional Node Positive")
        sv_label = QLabel("Survival Months")

        label_layout=QVBoxLayout()

        label_layout.addWidget(age_label)
        label_layout.addWidget(race_label)
        label_layout.addWidget(ms_label)
        label_layout.addWidget(tstage_label)
        label_layout.addWidget(nstage_label)
        label_layout.addWidget(sixstage_label)
        label_layout.addWidget(grade_label)
        label_layout.addWidget(astage_label)
        label_layout.addWidget(ts_label)
        label_layout.addWidget(es_label)
        label_layout.addWidget(ps_label)
        label_layout.addWidget(rne_label)
        label_layout.addWidget(rnp_label)
        label_layout.addWidget(sv_label)

        label_widget = QWidget()
        label_widget.setLayout(label_layout)

        self.age_lineedit = QLineEdit("20")
        self.race_cb=QComboBox()
        self.race_cb.addItems(self.net.get_list("Race "))
        self.ms_cb=QComboBox()
        self.ms_cb.addItems(self.net.get_list("Marital Status"))
        self.tstage_cb=QComboBox()
        self.tstage_cb.addItems(self.net.get_list("T Stage "))
        self.nstage_cb=QComboBox()
        self.nstage_cb.addItems(self.net.get_list("N Stage"))
        self.sixstage_cb=QComboBox()
        self.sixstage_cb.addItems(self.net.get_list("6th Stage"))
        self.grade_cb=QComboBox()
        self.grade_cb.addItems(self.net.get_list("Grade"))
        self.astage_cb=QComboBox()
        self.astage_cb.addItems(self.net.get_list("A Stage"))
        self.ts_lineedit = QLineEdit("40")
        self.es_cb=QComboBox()
        self.es_cb.addItems(self.net.get_list("Estrogen Status"))
        self.ps_cb=QComboBox()
        self.ps_cb.addItems(self.net.get_list("Progesterone Status"))
        self.rne_lineedit = QLineEdit("19")
        self.rnp_lineedit = QLineEdit("11")
        self.sv_lineedit = QLineEdit("5")

        edit_layout=QVBoxLayout()

        edit_layout.addWidget(self.age_lineedit)
        edit_layout.addWidget(self.race_cb)
        edit_layout.addWidget(self.ms_cb)
        edit_layout.addWidget(self.tstage_cb)
        edit_layout.addWidget(self.nstage_cb)
        edit_layout.addWidget(self.sixstage_cb)
        edit_layout.addWidget(self.grade_cb)
        edit_layout.addWidget(self.astage_cb)
        edit_layout.addWidget(self.ts_lineedit)
        edit_layout.addWidget(self.es_cb)
        edit_layout.addWidget(self.ps_cb)
        edit_layout.addWidget(self.rne_lineedit)
        edit_layout.addWidget(self.rnp_lineedit)
        edit_layout.addWidget(self.sv_lineedit)

        edit_widget=QWidget()
        edit_widget.setLayout(edit_layout)

        check_acc_btn = QPushButton("Check accuracy")
        check_acc_btn.clicked.connect(self.check_acc_btn_OnClick)
        check_acc_btn.show()
        predict_btn=QPushButton("Predict")
        predict_btn.clicked.connect(self.predict_btn_OnClick)
        predict_btn.show()
        train_btn=QPushButton("Train net")
        train_btn.clicked.connect(self.train_btn_OnClick)
        train_btn.show()
        self.ttrain_lineedit=QLineEdit("amount of trainings")
        self.ttrain_lineedit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        action_layout=QVBoxLayout()

        action_layout.addWidget(check_acc_btn)
        action_layout.addWidget(predict_btn)
        action_layout.addWidget(train_btn)
        action_layout.addWidget(self.ttrain_lineedit)

        action_widget=QWidget()
        action_widget.setLayout(action_layout)

        splitter=QSplitter()
        splitter.addWidget(label_widget)
        splitter.addWidget(edit_widget)
        splitter.addWidget(action_widget)

        self.setCentralWidget(splitter)
        self.setFixedSize(700,400)

    def check_acc_btn_OnClick(self):
        msg = QMessageBox()
        msg.setWindowTitle("Current accuracy")
        msg.setText(str(self.accuracy))
        msg.exec_()

    def predict_btn_OnClick(self):
        assert self.age_lineedit.text()!="","Please put Age at least"
        assert self.ts_lineedit.text()!="","Please put Tumor Size at least"
        assert self.rne_lineedit.text()!="","Please put Regional Node Examined at least"
        assert self.rnp_lineedit.text()!="","Please put Regional Node Positive at least"

        arr=np.array([float(self.age_lineedit.text()),
                      float(self.ts_lineedit.text()),
                      float(self.rne_lineedit.text()),
                      float(self.rnp_lineedit.text()),
                      float(self.sv_lineedit.text()),
                      self.net.match_unique_values(["Race _embedding_0","Race _embedding_1"],"Race ",
                                                   self.race_cb.currentText())[0],
                      self.net.match_unique_values(["Race _embedding_0", "Race _embedding_1"], "Race ",
                                                   self.race_cb.currentText())[1],
                      self.net.match_unique_values(["Marital Status_embedding_0", "Marital Status_embedding_1",
                                                    "Marital Status_embedding_2"], "Marital Status",
                                                   self.ms_cb.currentText())[0],
                      self.net.match_unique_values(["Marital Status_embedding_0", "Marital Status_embedding_1",
                                                    "Marital Status_embedding_2"],"Marital Status",
                                                   self.ms_cb.currentText())[1],
                      self.net.match_unique_values(["Marital Status_embedding_0", "Marital Status_embedding_1",
                                                    "Marital Status_embedding_2"], "Marital Status",
                                                   self.ms_cb.currentText())[2],
                      self.net.match_unique_values(["T Stage _embedding_0", "T Stage _embedding_1"], "T Stage ",
                                                   self.tstage_cb.currentText())[0],
                      self.net.match_unique_values(["T Stage _embedding_0", "T Stage _embedding_1"], "T Stage ",
                                                   self.tstage_cb.currentText())[1],
                      self.net.match_unique_values(["N Stage_embedding_0", "N Stage_embedding_1"], "N Stage",
                                                   self.nstage_cb.currentText())[0],
                      self.net.match_unique_values(["N Stage_embedding_0", "N Stage_embedding_1"], "N Stage",
                                                   self.nstage_cb.currentText())[1],
                      self.net.match_unique_values(["6th Stage_embedding_0","6th Stage_embedding_1",
                                                    "6th Stage_embedding_2"],"6th Stage",
                                                   self.sixstage_cb.currentText())[0],
                      self.net.match_unique_values(["6th Stage_embedding_0", "6th Stage_embedding_1",
                                                    "6th Stage_embedding_2"], "6th Stage",
                                                   self.sixstage_cb.currentText())[1],
                      self.net.match_unique_values(["6th Stage_embedding_0", "6th Stage_embedding_1",
                                                    "6th Stage_embedding_2"], "6th Stage",
                                                   self.sixstage_cb.currentText())[2],
                      self.net.match_unique_values(["Grade_embedding_0", "Grade_embedding_1"], "Grade",
                                                   self.grade_cb.currentText())[0],
                      self.net.match_unique_values(["Grade_embedding_0", "Grade_embedding_1"], "Grade",
                                                   self.grade_cb.currentText())[1],
                      self.net.match_unique_values(["A Stage_embedding_0"],"A Stage",
                                                   self.astage_cb.currentText())[0],
                      self.net.match_unique_values(["Estrogen Status_embedding_0"], "Estrogen Status",
                                                   self.es_cb.currentText())[0],
                      self.net.match_unique_values(["Progesterone Status_embedding_0"], "Progesterone Status",
                                                   self.ps_cb.currentText())[0]
                      ])
        arr=arr.reshape(1,22)
        print(arr.shape)
        print(arr)

        prediction=self.net.best_model.predict(arr)[0]
        print(prediction)

        msg = QMessageBox()

        if prediction>0.5:
            msg.setWindowTitle("Good news")
            msg.setText("You will probably live)")
        if prediction<0.5:
            msg.setWindowTitle("Bad news")
            msg.setText("You will probably die...")
        if prediction==0.5:
            msg.setWindowTitle("ANNNNMALY")
            msg.setText("On a verge of death are thou")

        msg.exec_()

    def train_btn_OnClick(self):
        self.setDisabled(True)
        try:
            ttrain=int(self.ttrain_lineedit.text())
        except ValueError:
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Please enter an integer value")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
            self.setEnabled(True)
        else:
            self.net.train_model(times=ttrain)
            self.setEnabled(True)


