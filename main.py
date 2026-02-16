import PyQt6.QtCore
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtCore import QDateTime,QEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QInputDialog, QLineEdit\
    , QGraphicsScene, QGraphicsPixmapItem, QSizePolicy, QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap
from ui_mainwindow import Ui_MainWindow as Ui_MainWindow
from ui_mainwindow_2 import Ui_MainWindow as Ui_MainWindow_2
from PyQt6 import QtSql
import sys

import sqlite3
import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import os
import shutil

from config import parser
from utils import select_model
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from gradcam import Grad_CAM, Grad_CAMpp

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.charset")

class mainWin(QMainWindow):
    def __init__(self):
        super(QMainWindow,self).__init__()
        self.uiInit()

        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        self.root = current_dir
        self.init_table()
        self.signalInit()

        self.open_windows = []

        self.args = parser.parse_args()
        self.model = select_model(self.args)

        
        self.pathModel = os.path.join(self.root, "savedModels",
                             "ResNet18_True_MAX_IMG_SIZE_512_num_class_1_best_model.pth")
    
        modelCheckpoint = torch.load(self.pathModel, map_location=torch.device('cpu'))
        self.model.load_state_dict(modelCheckpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def uiInit(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('影像列表')

    def evaluate(self,img):
        with torch.no_grad():
            img = img.convert('RGB')
            img = self.transform(img)
            output = self.model(img.unsqueeze(0))
            output = 1/(1+np.exp(-output.item()))
            return '气胸' if output>0.5 else '正常'
    
    def heatmap_generate(self,img,method):
        img = img.convert('RGB')
        imageData = self.transform(img).unsqueeze(0)
        self.imgOriginal = imageData.squeeze(0).detach().cpu()

        model_dict = dict(type='resnet',
                                layer_name='img_model_layer4',
                                arch=self.model,
                                input_size=(self.args.img_size, self.args.img_size)
                                )

        # Function that generate the heatmap with GradCAM
        GradCAM = Grad_CAM(model_dict)
        # Function that generate the heatmap with GradCAM++
        GradCAMCPP = Grad_CAMpp(model_dict)

        heatmap = None

        if method == 'CAM':
            weights = list(self.model.img_model.fc[1].parameters())[-2].squeeze()
            output = self.model.img_model.conv1(imageData)
            output = self.model.img_model.bn1(output)
            output = self.model.img_model.relu(output)
            output = self.model.img_model.maxpool(output)
            output = self.model.img_model.layer1(output)
            output = self.model.img_model.layer2(output)
            output = self.model.img_model.layer3(output)
            output = self.model.img_model.layer4(output)
            output = F.relu(output)

            for i in range(0, len(weights)):
                map = output[0, i, :, :]
                if i == 0:
                    heatmap = weights[i] * map
                else:
                    heatmap += weights[i] * map
        
        elif method == 'GradCAM':
            imageData_temp = imageData.clone().requires_grad_(True)
            heatmap, logit = GradCAM(imageData_temp, class_idx=0)
        
        elif method == 'GradCAMpp':
            imageData_temp = imageData.clone().requires_grad_(True)
            heatmap, logit = GradCAMCPP(imageData_temp, class_idx=0)
        
        else:
            print('no such method')
            return
           
        self.heatmap = heatmap


    def heatmap_visualize(self,heatmap,save_location):
        npHeatmap = heatmap.detach().cpu().data.squeeze().numpy()

        cam = npHeatmap - np.min(npHeatmap)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, (self.args.img_size, self.args.img_size))

        factor=1.9 
        threshold_high=0.6
        thresh_low = cam.mean() * factor
        if thresh_low > threshold_high:
            thresh_low = threshold_high
        thresh_low = float(thresh_low)
        _, thresh = cv2.threshold(cam, thresh_low, 1, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
                (thresh.astype(np.uint8)),
                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = torch.from_numpy(heat).permute(2, 0, 1).float().div(255)
        b, g, r = heat.split(1)
        heat = torch.cat([r, g, b])

        #result = heat + imgOriginal
        #result = imgOriginal * (1 + heat * 0.5)
        alpha = 0.5  # 热力图透明度
        result = self.imgOriginal * (1 - alpha) + heat * alpha
        result = result - result.min()
        result = result / result.max()
        result = result.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        plt.figure()
        plt.axis('off')
        plt.imshow(result)
        ax = plt.gca()
        
        for c_0 in contours:
            area = cv2.contourArea(c_0)
            x, y, w, h = cv2.boundingRect(c_0)
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g',
                                        facecolor='none')
            ax.add_patch(rect)
        
        plt.savefig(save_location,bbox_inches='tight',pad_inches=-0.05)


    def load_file(self):

        fileDia = QFileDialog()
        file_path,selectedDic = fileDia.getOpenFileName()
        if not file_path.endswith('dcm'):
            QMessageBox.critical(self,'错误','请选择.dcm文件')
        else:      
            conn = sqlite3.connect(self.root+'/'+'database.db')
            cursor = conn.cursor()
            data = []
            
            ds = pydicom.dcmread(file_path)

            try:
                pixel_array = apply_voi_lut(ds.pixel_array, ds)
            except AttributeError:
                pixel_array = ds.pixel_array

            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
            pixel_array = pixel_array.astype('uint8')

            image = Image.fromarray(pixel_array)

            png_savepath = self.root+'/'+'Images/'+str(ds.PatientName)+'_'+str(ds.PatientID)\
                +'/'+str(ds.StudyDate)\
                    +'/'+str(ds.SOPInstanceUID)\
                        +'/'+'image.png'
            dcm_savepath = png_savepath.replace('image.png', 'image.dcm')
            heat_savepath_CAM = png_savepath.replace('image.png', 'heatmap_CAM.png')
            heat_savepath_GradCAM = png_savepath.replace('image.png', 'heatmap_GradCAM.png')
            heat_savepath_GradCAMpp = png_savepath.replace('image.png', 'heatmap_GradCAMpp.png')
            inv_savepath = png_savepath.replace('image.png', 'inverted_image.png')

            inverted_array = 255-pixel_array 
            inverted_img = Image.fromarray(inverted_array)
            
            info = {'PatientName': str(ds.PatientName), 
                    'PatientID': str(ds.PatientID),
                    'PatientSex': str(ds.PatientSex), 
                    'PatientBirthDate': str(ds.PatientBirthDate), 
                    'PatientAge': str(ds.PatientAge) if hasattr(ds, 'PatientAge') else 'unknown', 
                    'StudyDate': str(ds.StudyDate),
                    'ViewPosition': str(ds.ViewPosition) if hasattr(ds, 'ViewPosition') else 'unknown', 
                    'PatientPosition': str(ds.PatientPosition) if hasattr(ds, 'PatientPosition') else 'unknown',
                    'BodyPartExamined': str(ds.BodyPartExamined) if hasattr(ds, 'BodyPartExamined') else 'unknown',
                    'Diagnosis': self.evaluate(image),
                    'ImagePath': png_savepath,
                    'HeatmaPath': heat_savepath_CAM,
                    'DoctorNote': ''
                    }

            data.append(info)
            df = pd.DataFrame(data)
            query = 'select exists (select 1 from data where ImagePath = ?)'
            cursor.execute(query, (png_savepath,))
            result = cursor.fetchone()[0]
            if not bool(result):
                df.to_sql('data', conn, if_exists='append', index=False)
                conn.close()
            else:
                #QMessageBox.critical(self,'错误','该病例已存在') 
                conn.close()        
                return
            
            savepath = os.path.dirname(png_savepath)
            if not os.path.exists(savepath):
                os.makedirs(savepath, exist_ok=True)

            image.save(png_savepath)
            shutil.copy(file_path, dcm_savepath)
            method_dict = {'CAM': heat_savepath_CAM,
                           'GradCAM': heat_savepath_GradCAM,
                           'GradCAMpp': heat_savepath_GradCAMpp}
            for method, heat_savepath in method_dict.items():
                self.heatmap_generate(image,method)
                self.heatmap_visualize(self.heatmap,heat_savepath)
            inverted_img.save(inv_savepath)


            row = list(df.iloc[0,:])
            current_row_count = self.ui.tableWidget.rowCount()
            self.ui.tableWidget.insertRow(current_row_count)
            self.ui.tableWidget.cellChanged.disconnect(self.modifyTable)    
            for j in range(len(row)):
                if j == 3 or j == 5:
                    item = QTableWidgetItem(QDateTime.fromString(str(row[j]),"yyyyMMdd").toString("yyyy-MM-dd"))
                elif j==4:
                    item = QTableWidgetItem(str(row[j]).rstrip('Y').lstrip('0'))
                else:   
                    item = QTableWidgetItem(str(row[j]))   
                self.ui.tableWidget.setItem(current_row_count, j, item)
            self.ui.tableWidget.cellChanged.connect(self.modifyTable)

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择目录", "/") 
        if folder_path:
            self.statusBar().showMessage("正在加载文件，请稍候...")
            QApplication.processEvents()
            conn = sqlite3.connect(self.root+'/'+'database.db')
            QApplication.processEvents()
            cursor = conn.cursor()
            for root, dirs, files in os.walk(folder_path):
                for file in files: 
                    if file.endswith('.dcm'):
                        data = []
                        file_path = os.path.join(root, file)
                        ds = pydicom.dcmread(file_path)

                        try:
                            pixel_array = apply_voi_lut(ds.pixel_array, ds)
                        except AttributeError:
                            pixel_array = ds.pixel_array

                        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                        pixel_array = pixel_array.astype('uint8')

                        image = Image.fromarray(pixel_array)

                        png_savepath = self.root+'/'+'Images/'+str(ds.PatientName)+'_'+str(ds.PatientID)\
                            +'/'+str(ds.StudyDate)\
                                +'/'+str(ds.SOPInstanceUID)\
                                    +'/'+'image.png'
                        dcm_savepath = png_savepath.replace('image.png', 'image.dcm')
                        heat_savepath_CAM = png_savepath.replace('image.png', 'heatmap_CAM.png')
                        heat_savepath_GradCAM = png_savepath.replace('image.png', 'heatmap_GradCAM.png')
                        heat_savepath_GradCAMpp = png_savepath.replace('image.png', 'heatmap_GradCAMpp.png')

                        inverted_array = 255-pixel_array 
                        inverted_img = Image.fromarray(inverted_array)
                        inv_savepath = png_savepath.replace('image.png', 'inverted_image.png')


                        info = {'PatientName': str(ds.PatientName), 
                                'PatientID': str(ds.PatientID),
                                'PatientSex': str(ds.PatientSex), 
                                'PatientBirthDate': str(ds.PatientBirthDate), 
                                'PatientAge': str(ds.PatientAge) if hasattr(ds, 'PatientAge') else 'unknown', 
                                'StudyDate': str(ds.StudyDate),
                                'ViewPosition': str(ds.ViewPosition) if hasattr(ds, 'ViewPosition') else 'unknown', 
                                'PatientPosition': str(ds.PatientPosition) if hasattr(ds, 'PatientPosition') else 'unknown',
                                'BodyPartExamined': str(ds.BodyPartExamined) if hasattr(ds, 'BodyPartExamined') else 'unknown',
                                'Diagnosis': self.evaluate(image),
                                'ImagePath': png_savepath,
                                'HeatmaPath': heat_savepath_CAM,
                                'DoctorNote': ''
                                }

                        data.append(info)
                        df = pd.DataFrame(data)
                        query = 'select exists (select 1 from data where ImagePath = ?)'
                        cursor.execute(query, (png_savepath,))
                        result = cursor.fetchone()[0]
                        if not bool(result):
                            df.to_sql('data', conn, if_exists='append', index=False)
                        else:
                            continue

                        savepath = os.path.dirname(png_savepath)
                        if not os.path.exists(savepath):
                            os.makedirs(savepath, exist_ok=True)
                        

                        image.save(png_savepath)
                        shutil.copy(file_path, dcm_savepath)
                        method_dict = {'CAM': heat_savepath_CAM,
                                    'GradCAM': heat_savepath_GradCAM,
                                    'GradCAMpp': heat_savepath_GradCAMpp}
                        for method, heat_savepath in method_dict.items():
                            self.heatmap_generate(image,method)
                            self.heatmap_visualize(self.heatmap,heat_savepath)
                        inverted_img.save(inv_savepath)


                        row = list(df.iloc[0,:])
                        current_row_count = self.ui.tableWidget.rowCount()
                        self.ui.tableWidget.insertRow(current_row_count)
                        self.ui.tableWidget.cellChanged.disconnect(self.modifyTable)    
                        for j in range(len(row)):
                            if j == 3 or j == 5:
                                item = QTableWidgetItem(QDateTime.fromString(str(row[j]),"yyyyMMdd").toString("yyyy-MM-dd"))
                            elif j==4:
                                item = QTableWidgetItem(str(row[j]).rstrip('Y').lstrip('0'))   
                            else:   
                                item = QTableWidgetItem(str(row[j]))   
                            self.ui.tableWidget.setItem(current_row_count, j, item)
                        self.ui.tableWidget.cellChanged.connect(self.modifyTable)    
            
            conn.close()
            self.statusBar().showMessage("加载成功", 3000)

    def init_table(self):
        self.database = QtSql.QSqlDatabase.addDatabase("QSQLITE")
        self.database.setDatabaseName(os.path.join(self.root,'database.db'))
        self.database.open()
        query = QtSql.QSqlQuery(self.database)
        query.exec(
        'CREATE TABLE IF NOT EXISTS data (\
            PatientName TEXT,\
            PatientID TEXT,\
            PatientSex TEXT,\
            PatientBirthDate TEXT,\
            PatientAge TEXT,\
            StudyDate TEXT,\
            ViewPosition TEXT,\
            PatientPosition TEXT,\
            BodyPartExamined TEXT,\
            Diagnosis TEXT,\
            ImagePath TEXT,\
            HeatmaPath TEXT,\
            DoctorNote TEXT)'
        )
        query.exec('select * from data')
        allrows = []
        column_count = query.record().count()
        while query.next():
            row = []
            for i in range(column_count):
                row.append(query.value(i))
            allrows.append(row)
        for i in range(len(allrows)):
            self.ui.tableWidget.insertRow(i)    
            for j in range(len(allrows[0])):
                if j == 3 or j == 5:
                    item = QTableWidgetItem(QDateTime.fromString(str(allrows[i][j]),"yyyyMMdd").toString("yyyy-MM-dd"))
                elif j==4:
                    item = QTableWidgetItem(str(allrows[i][j]).rstrip('Y').lstrip('0'))
                else:   
                    item = QTableWidgetItem(str(allrows[i][j]))   
                self.ui.tableWidget.setItem(i, j, item)

    def modifyTable(self,row,column):
        if column == 10:
            QMessageBox.critical(self,'错误','禁止修改图片路径')
            win.close()
        else:
            dcm_id = self.ui.tableWidget.item(row, 10).text()  
            value = self.ui.tableWidget.item(row, column).text()
            ui_colName = self.ui.tableWidget.horizontalHeaderItem(column).text()

            column_mapping = {
                'Name': 'PatientName',
                'ID': 'PatientID',
                'Sex': 'PatientSex',
                'BirthDate': 'PatientBirthDate',
                'Age': 'PatientAge'           
            }
            
            colName = column_mapping.get(ui_colName, ui_colName)
            conn = sqlite3.connect(self.root+'/'+'database.db')
            cursor = conn.cursor()
            cursor.execute(f"select {colName} from data where ImagePath = '{dcm_id}'")
            original_value = cursor.fetchone()[0]
            conn.close()
                 
            reply = QMessageBox.question(
            self,
            '确认更改',
            '确定要改动记录吗？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

            if reply == QMessageBox.StandardButton.Yes:
                
                print(f"Updating: UI Column={ui_colName}, DB Column={colName}, Value={value}, ImagePath={dcm_id}")
                # Create a new query with parameterized SQL
                query = QtSql.QSqlQuery(self.database) 
                # 修复引号嵌套问题
                value_escaped = value.replace("'", "''")
                id_escaped = dcm_id.replace("'", "''")
                sql = f"UPDATE data SET {colName} = '{value_escaped}' WHERE ImagePath = '{id_escaped}'"
            
                # Execute the query with error handling
                if not query.exec(sql):
                    print(f"Error updating record: {query.lastError().text()}")
                    return False
                
                print(f"Successfully updated record for ImagePath {dcm_id}")
                
                return True
            else:
                self.ui.tableWidget.cellChanged.disconnect(self.modifyTable)
                self.ui.tableWidget.setItem(row,column,QTableWidgetItem(str(original_value)))
                self.ui.tableWidget.cellChanged.connect(self.modifyTable)
        
    def delete_record(self):
        reply = QMessageBox.question(
            self,
            '确认删除',
            '确定删除记录及其关联的文件夹吗？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            query = QtSql.QSqlQuery(self.database)
            seleRanges = self.ui.tableWidget.selectedRanges()

            #self.ui.tableWidget.cellChanged.disconnect(self.modifyTable)    
            rows = set()
            for j in seleRanges:
                rows.update({j.topRow()+i for i in range(j.rowCount())})
            rows = sorted(rows,reverse=True)

            for i in rows:
                dcm_id = self.ui.tableWidget.item(i,10).text()
                dcm_id_escaped = dcm_id.replace("'","''")
                sql = f'delete from data where ImagePath = "{dcm_id_escaped}"'
                query.exec(sql)
                folder_to_remove = os.path.dirname(dcm_id)
                shutil.rmtree(folder_to_remove)
                self.ui.tableWidget.removeRow(i)
            #self.ui.tableWidget.cellChanged.connect(self.modifyTable)    
            
    def signalInit(self):
        self.ui.pushButton.clicked.connect(self.search_record)
        self.ui.pushButton_2.clicked.connect(self.show_image)
        self.ui.pushButton_3.clicked.connect(self.load_file)
        self.ui.pushButton_4.clicked.connect(self.load_folder)
        self.ui.tableWidget.cellChanged.connect(self.modifyTable)
        self.ui.pushButton_5.clicked.connect(self.delete_record)
    
    def search_record(self):
        id, Pressed = QInputDialog.getText(self,"输入患者ID","ID:",QLineEdit.EchoMode.Normal,"")
        if Pressed == False:
            return
        for i in range(self.ui.tableWidget.rowCount()):
            if self.ui.tableWidget.item(i,1).text() == id:
                self.ui.tableWidget.setCurrentCell(i,1)
                break

    def show_image(self):
        ranges = self.ui.tableWidget.selectedRanges()
        rows = set()
        for j in ranges:
            rows.update({j.topRow()+i for i in range(j.rowCount())})
        if len(rows)==1:
            row = self.ui.tableWidget.currentRow()
            self.path = self.ui.tableWidget.item(row,10).text()
            win_2 = mainWin_2(self.path)
            self.open_windows.append(win_2)
            win_2.show() 
        else:
            QMessageBox.critical(self,'错误','请选择单条患者记录')
                   
   
class mainWin_2(QMainWindow):
    def __init__(self, path):
        super(QMainWindow,self).__init__()
        self.ui = Ui_MainWindow_2()
        self.ui.setupUi(self)
        self.path = path
        self.signalInit()
        self.view_dic = {0:self.ui.graphicsView, 1:self.ui.graphicsView_2}

        self.heat_savepath = path.replace('image.png', 'heatmap_CAM.png')
        
        self.image_dic = {0:self.path, 1:self.heat_savepath}
        self.image_dic_backup = {0:self.path, 1:self.heat_savepath}
        self.ratio_dic = {0:1, 1:1}
        self.scale = 0.9 * min(self.ui.graphicsView.size().width(), self.ui.graphicsView.size().height())
        #self.set_image('/Users/youpengsun/test/Squirrel.jpg',0,self.ratio_dic[0])
        self.set_image(self.image_dic[0],0,self.ratio_dic[0])
        self.set_image(self.image_dic[1],1,self.ratio_dic[1])
        self.diagnosis_label()
               
    def resizeEvent(self,event):
        super(mainWin_2, self).resizeEvent(event)
        self.scale = 0.9 * min(self.ui.graphicsView.size().width(), self.ui.graphicsView.size().height())
        #self.scale = 1/2*(self.ui.graphicsView.size().width()+self.ui.graphicsView.size().height())
        current_index = self.ui.stackedWidget_2.currentIndex()
        self.set_image(self.image_dic[current_index], current_index, self.ratio_dic[current_index])

    def signalInit(self):
        self.ui.pushButton_10.clicked.connect(self.switch_image)
        self.ui.pushButton_2.clicked.connect(self.magnify_glass)
        self.ui.pushButton.clicked.connect(self.reset_image)
        self.ui.pushButton_3.clicked.connect(self.zoom_out)
        self.ui.pushButton_8.clicked.connect(self.switch_info)
        self.ui.pushButton_4.clicked.connect(self.invert_image)

    def switch_image(self):
        if self.ui.stackedWidget_2.currentIndex() == 0:
            self.ui.stackedWidget_2.setCurrentIndex(1)
        else:
            self.ui.stackedWidget_2.setCurrentIndex(0)

    def switch_info(self):
        if self.ui.stackedWidget.currentIndex() == 0:
            self.ui.stackedWidget.setCurrentIndex(1)
        else:
            self.ui.stackedWidget.setCurrentIndex(0)

    def set_image(self, image_path, index, ratio):
        scene = QGraphicsScene()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(int(ratio * self.scale),int(ratio * self.scale), Qt.AspectRatioMode.KeepAspectRatio)
        scene.addItem(QGraphicsPixmapItem(pixmap))
        view = self.view_dic[index]
        view.setScene(scene)

    def diagnosis_label(self):
        pixmap = QPixmap('/Users/youpengsun/Downloads/lightning.png')
        scale = int(1.2 * max(self.ui.label_3.size().width(), self.ui.label_3.size().height()))
        pixmap = pixmap.scaled(scale,scale, Qt.AspectRatioMode.KeepAspectRatio)
        self.ui.label_3.setPixmap(pixmap)

    def magnify_glass(self):
        index = self.ui.stackedWidget_2.currentIndex()
        self.set_image(self.image_dic[index],index,self.ratio_dic[index]*1.2)
        self.ratio_dic[index] = self.ratio_dic[index]*1.2

    def reset_image(self):
        index = self.ui.stackedWidget_2.currentIndex()
        self.image_dic[index]=self.image_dic_backup[index] 
        self.set_image(self.image_dic[index],index,1)
        self.ratio_dic[index] = 1

    def zoom_out(self):
        index = self.ui.stackedWidget_2.currentIndex()
        self.set_image(self.image_dic[index],index,self.ratio_dic[index]/1.2)
        self.ratio_dic[index] = self.ratio_dic[index]/1.2

    def invert_image(self):
        index = self.ui.stackedWidget_2.currentIndex()
        inv_savepath = self.path.replace('image.png', 'inverted_image.png')
        if self.image_dic[index] != inv_savepath:
            self.image_dic[index] = inv_savepath      
        elif index==0:
            self.image_dic[index] = self.path
        else:
            self.image_dic[index] = self.heat_savepath
        self.set_image(self.image_dic[index],index,self.ratio_dic[index])




if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = mainWin()
    win.show()
    sys.exit(app.exec())
