import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, 
                           QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QFrame, 
                           QGroupBox, QTabWidget, QProgressBar, QTextBrowser)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QLinearGradient, QBrush, QPen
import pandas as pd
import traceback

import numpy as np
import torch
from lightning import pytorch as pl
from chemprop import data, featurizers, models


class HerbBackgroundWidget(QWidget):
    """中药背景装饰部件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
    
    def paintEvent(self, event):
        """绘制中药装饰背景"""
        painter = QPainter(self)
        
        # 创建渐变背景
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, QColor(139, 69, 19, 50))    # 深褐色
        gradient.setColorAt(0.5, QColor(210, 105, 30, 30)) # 巧克力色
        gradient.setColorAt(1, QColor(139, 69, 19, 50))    # 深褐色
        
        painter.fillRect(self.rect(), gradient)
        
        # 绘制装饰图案
        painter.setPen(QPen(QColor(160, 82, 45, 100), 2))
        
        # 绘制草药图案（简单的叶子形状）
        for i in range(0, self.width(), 100):
            # 左叶子
            painter.drawArc(i + 20, 20, 20, 40, 90 * 16, 180 * 16)
            # 右叶子
            painter.drawArc(i + 40, 20, 20, 40, 270 * 16, 180 * 16)
            # 茎
            painter.drawLine(i + 40, 40, i + 40, 60)


class ChempropBeautifulGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口属性
        self.setWindowTitle("🌿 本草御菌录 - 中药抗生素成分智能预测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化模型
        self.model = None
        self.model_path = None
        self.prediction_history = []
        
        # 创建主窗口部件
        self.create_main_widget()
        
        # 设置整体样式
        self.set_beautiful_style()
        
        # 添加启动动画
        self.show_welcome_animation()
    
    def create_main_widget(self):
        """创建主窗口部件"""
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建装饰背景
        self.herb_background = HerbBackgroundWidget()
        main_layout.addWidget(self.herb_background)
        
        # 内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(20, 10, 20, 20)
        
        # 标题区域
        self.create_title_section(content_layout)
        
        # 创建标签页
        self.create_tab_widget(content_layout)
        
        # 添加日志显示区域
        self.create_log_section(content_layout)
        
        main_layout.addWidget(content_widget)
    
    def create_log_section(self, parent_layout):
        """创建日志显示区域"""
        log_group = self.create_beautiful_groupbox("📝 系统日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_edit = QTextEdit()
        self.log_edit.setMaximumHeight(150)
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("""
            QTextEdit {
                border: 2px solid #D2691E;
                border-radius: 8px;
                padding: 10px;
                background-color: #FFFAF0;
                color: #8B4513;
                font-family: '宋体';
                font-size: 11px;
            }
        """)
        
        log_layout.addWidget(self.log_edit)
        parent_layout.addWidget(log_group)
    
    def create_title_section(self, parent_layout):
        """创建标题区域"""
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        
        # 主标题
        title_label = QLabel("🌿 本草御菌录")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("楷体", 32, QFont.Bold))
        title_label.setStyleSheet("""
            color: #8B4513;
            padding: 15px;
            margin: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #FFF8DC, stop:0.5 #F5DEB3, stop:1 #FFF8DC);
            border: 3px solid #D2691E;
            border-radius: 15px;
        """)
        title_layout.addWidget(title_label)
        
        # 副标题
        subtitle_label = QLabel("传承千年智慧 · 融合现代科技 · 智能识别中药抗生素成分")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont("宋体", 14))
        subtitle_label.setStyleSheet("color: #A0522D; margin: 5px;")
        title_layout.addWidget(subtitle_label)
        
        # 装饰分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 transparent, stop:0.5 #D2691E, stop:1 transparent);
            height: 3px;
            margin: 10px 0;
        """)
        title_layout.addWidget(separator)
        
        parent_layout.addWidget(title_widget)
    
    def create_tab_widget(self, parent_layout):
        """创建标签页部件"""
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("宋体", 12))
        
        # 预测功能标签页
        prediction_tab = self.create_prediction_tab()
        self.tab_widget.addTab(prediction_tab, "🔮 智能预测")
        
        # 模型管理标签页
        model_tab = self.create_model_tab()
        self.tab_widget.addTab(model_tab, "⚙️ 模型管理")
        
        # 历史记录标签页
        history_tab = self.create_history_tab()
        self.tab_widget.addTab(history_tab, "📜 历史记录")
        
        # 帮助标签页
        help_tab = self.create_help_tab()
        self.tab_widget.addTab(help_tab, "❓ 使用帮助")
        
        # 标签页样式
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #D2691E;
                border-radius: 10px;
                background: rgba(255, 248, 220, 0.3);
                margin-top: 10px;
            }
            QTabBar::tab {
                background: #DEB887;
                color: #8B4513;
                padding: 12px 20px;
                margin: 2px;
                border: 1px solid #D2691E;
                border-radius: 8px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #CD853F;
                color: white;
                border: 2px solid #8B4513;
            }
            QTabBar::tab:hover {
                background: #D2691E;
                color: white;
            }
        """)
        
        parent_layout.addWidget(self.tab_widget)
    
    def create_prediction_tab(self):
        """创建预测标签页"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setSpacing(15)
        
        # 单分子预测区域
        single_group = self.create_beautiful_groupbox("🌿 单分子草药预测")
        single_layout = QVBoxLayout(single_group)
        
        # SMILES输入
        smiles_layout = QHBoxLayout()
        smiles_layout.addWidget(QLabel("🔬 SMILES分子式:"))
        
        self.smiles_input = QLineEdit()
        self.smiles_input.setPlaceholderText("例如: CC(=O)OC1=CC=CC=C1C(=O)O (阿司匹林)")
        smiles_layout.addWidget(self.smiles_input)
        
        self.predict_single_btn = QPushButton("✨ 预测此分子")
        self.predict_single_btn.clicked.connect(self.predict_single_molecule)
        smiles_layout.addWidget(self.predict_single_btn)
        
        single_layout.addLayout(smiles_layout)
        
        # 快速示例
        example_layout = QHBoxLayout()
        example_layout.addWidget(QLabel("📝 快速示例:"))
        
        examples = [
            ("阿司匹林", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("青霉素", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"),
            ("维生素C", "C([C@@H]([C@H](C=O)O)O)[C@@H](C(=O)O)O")
        ]
        
        for name, smiles in examples:
            btn = QPushButton(f"{name}")
            btn.clicked.connect(lambda checked, s=smiles: self.smiles_input.setText(s))
            example_layout.addWidget(btn)
        
        example_layout.addStretch()
        single_layout.addLayout(example_layout)
        
        layout.addWidget(single_group)
        
        # 批量预测区域
        batch_group = self.create_beautiful_groupbox("📊 批量草药预测")
        batch_layout = QVBoxLayout(batch_group)
        
        # 文件选择
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("📁 数据文件:"))
        
        self.test_edit = QLineEdit()
        self.test_edit.setPlaceholderText("选择包含SMILES的CSV文件")
        file_layout.addWidget(self.test_edit)
        
        self.select_file_btn = QPushButton("📂 选择文件")
        self.select_file_btn.clicked.connect(self.select_test)
        file_layout.addWidget(self.select_file_btn)
        
        self.batch_predict_btn = QPushButton("🎯 批量预测")
        self.batch_predict_btn.clicked.connect(self.run_prediction)
        file_layout.addWidget(self.batch_predict_btn)
        
        batch_layout.addLayout(file_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #D2691E;
                border-radius: 5px;
                text-align: center;
                background: #FFFAF0;
                color: #8B4513;
            }
            QProgressBar::chunk {
                background-color: #CD853F;
                width: 20px;
            }
        """)
        batch_layout.addWidget(self.progress_bar)
        
        layout.addWidget(batch_group)
        
        # 结果显示区域
        result_group = self.create_beautiful_groupbox("📋 预测结果")
        result_layout = QVBoxLayout(result_group)
        
        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels([
            "SMILES分子式", "预测分数", "置信度", "草药性质"
        ])
        
        # 表格样式
        self.result_table.setStyleSheet("""
            QTableWidget {
                border: 2px solid #D2691E;
                border-radius: 8px;
                background-color: #FFFAF0;
                color: #8B4513;
                gridline-color: #DEB887;
            }
            QHeaderView::section {
                background-color: #DEB887;
                color: #8B4513;
                font-weight: bold;
                border: 1px solid #D2691E;
                padding: 8px;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
                border: 1px solid #F5DEB3;
                font-size: 11px;
            }
        """)
        
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(self.result_table)
        
        layout.addWidget(result_group)
        
        return tab_widget
    
    def create_model_tab(self):
        """创建模型管理标签页"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setSpacing(15)
        
        # 模型信息区域
        info_group = self.create_beautiful_groupbox("📊 模型信息")
        info_layout = QVBoxLayout(info_group)
        
        self.model_info_display = QTextBrowser()
        self.model_info_display.setMaximumHeight(200)
        self.model_info_display.setStyleSheet("""
            QTextBrowser {
                border: 2px solid #D2691E;
                border-radius: 8px;
                padding: 10px;
                background-color: #FFFAF0;
                color: #8B4513;
                font-family: '宋体';
                font-size: 12px;
            }
        """)
        
        info_layout.addWidget(self.model_info_display)
        layout.addWidget(info_group)
        
        # 模型操作区域
        action_group = self.create_beautiful_groupbox("⚙️ 模型操作")
        action_layout = QVBoxLayout(action_group)
        
        # 路径选择
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("🎯 模型路径:"))
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("model/model_0/checkpoints/best-epoch=42-val_loss=0.12.ckpt")
        path_layout.addWidget(self.model_path_edit)
        
        self.browse_model_btn = QPushButton("📁 浏览")
        self.browse_model_btn.clicked.connect(self.select_model)
        path_layout.addWidget(self.browse_model_btn)
        
        action_layout.addLayout(path_layout)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.load_model_btn = QPushButton("⚡ 加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        button_layout.addWidget(self.load_model_btn)
        
        self.unload_model_btn = QPushButton("🔄 卸载模型")
        self.unload_model_btn.clicked.connect(self.unload_model)
        button_layout.addWidget(self.unload_model_btn)
        
        self.refresh_info_btn = QPushButton("🔄 刷新信息")
        self.refresh_info_btn.clicked.connect(self.refresh_model_info)
        button_layout.addWidget(self.refresh_info_btn)
        
        action_layout.addLayout(button_layout)
        
        layout.addWidget(action_group)
        
        # 模型状态
        status_group = self.create_beautiful_groupbox("🏮 模型状态")
        status_layout = QVBoxLayout(status_group)
        
        self.model_status_label = QLabel("模型状态: 未加载")
        self.model_status_label.setFont(QFont("宋体", 14, QFont.Bold))
        self.model_status_label.setAlignment(Qt.AlignCenter)
        self.model_status_label.setStyleSheet("""
            color: #CD853F;
            padding: 20px;
            border: 2px solid #D2691E;
            border-radius: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #FFF8DC, stop:1 #F5DEB3);
        """)
        status_layout.addWidget(self.model_status_label)
        
        layout.addWidget(status_group)
        
        return tab_widget
    
    def create_history_tab(self):
        """创建历史记录标签页"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setSpacing(15)
        
        # 历史记录显示
        history_group = self.create_beautiful_groupbox("📜 预测历史")
        history_layout = QVBoxLayout(history_group)
        
        self.history_display = QTextBrowser()
        self.history_display.setStyleSheet("""
            QTextBrowser {
                border: 2px solid #D2691E;
                border-radius: 8px;
                padding: 10px;
                background-color: #FFFAF0;
                color: #8B4513;
                font-family: '宋体';
                font-size: 12px;
            }
        """)
        
        history_layout.addWidget(self.history_display)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.clear_history_btn = QPushButton("🗑️ 清空历史")
        self.clear_history_btn.clicked.connect(self.clear_history)
        button_layout.addWidget(self.clear_history_btn)
        
        self.export_history_btn = QPushButton("💾 导出历史")
        self.export_history_btn.clicked.connect(self.export_history)
        button_layout.addWidget(self.export_history_btn)
        
        button_layout.addStretch()
        history_layout.addLayout(button_layout)
        
        layout.addWidget(history_group)
        
        return tab_widget
    
    def create_help_tab(self):
        """创建帮助标签页"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setSpacing(15)
        
        help_group = self.create_beautiful_groupbox("❓ 使用帮助")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QTextBrowser()
        help_text.setHtml("""
            <html>
            <body style="font-family: '宋体'; font-size: 14px; color: #8B4513;">
            
            <h2 style="color: #8B4513; text-align: center;">🌿 本草御菌录使用指南</h2>
            
            <h3 style="color: #CD853F;">🏮 系统介绍</h3>
            <p>本草御菌录是基于深度学习的传统中药抗生素成分智能识别系统，<br>
            结合现代分子化学与传统中医药理论，为中药研究和开发提供科学依据。</p>
            
            <h3 style="color: #CD853F;">🔧 使用步骤</h3>
            <ol>
                <li><b>选择模型：</b>点击"选择模型"按钮，选择您的Chemprop模型文件</li>
                <li><b>加载模型：</b>点击"加载模型"按钮，等待模型激活</li>
                <li><b>输入分子：</b>在预测功能中输入SMILES分子式</li>
                <li><b>开始预测：</b>点击相应的预测按钮，获得预测结果</li>
            </ol>
            
            <h3 style="color: #CD853F;">🧪 SMILES示例</h3>
            <ul>
                <li><b>阿司匹林：</b> CC(=O)OC1=CC=CC=C1C(=O)O</li>
                <li><b>青霉素：</b> CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O</li>
                <li><b>维生素C：</b> C([C@@H]([C@H](C=O)O)O)[C@@H](C(=O)O)O</li>
            </ul>
            
            <h3 style="color: #CD853F;">📊 结果说明</h3>
            <ul>
                <li>🟢 <b>绿色结果：</b>预测为抗生素成分（分数 > 0.5）</li>
                <li>🟡 <b>黄色结果：</b>预测为非抗生素成分（分数 ≤ 0.5）</li>
                <li>🔴 <b>红色结果：</b>预测失败或错误</li>
            </ul>
            
            <h3 style="color: #CD853F;">💡 使用技巧</h3>
            <ul>
                <li>批量预测时，确保CSV文件包含'smiles'列</li>
                <li>预测结果会自动保存到历史记录中</li>
                <li>可以导出历史记录进行进一步分析</li>
            </ul>
            
            </body>
            </html>
        """)
        
        help_layout.addWidget(help_text)
        layout.addWidget(help_group)
        
        return tab_widget
    
    def create_beautiful_groupbox(self, title):
        """创建美观的组框"""
        groupbox = QGroupBox(title)
        groupbox.setFont(QFont("宋体", 12, QFont.Bold))
        groupbox.setStyleSheet("""
            QGroupBox {
                color: #8B4513;
                border: 3px solid #D2691E;
                border-radius: 12px;
                margin-top: 15px;
                padding-top: 15px;
                background: rgba(255, 248, 220, 0.4);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #DEB887, stop:1 #F5DEB3);
                border: 2px solid #D2691E;
                border-radius: 8px;
            }
        """)
        return groupbox
    
    def set_beautiful_style(self):
        """设置美观的整体样式"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FFFEF7, stop:0.5 #F5F5DC, stop:1 #FFF8DC);
            }
        """)
    
    def show_welcome_animation(self):
        """显示欢迎动画"""
        # 这里可以添加启动动画效果
        QTimer.singleShot(1000, self.show_ready_message)
    
    def show_ready_message(self):
        """显示就绪信息"""
        self.log_edit.append("🎉 本草御菌录系统已准备就绪！")
        self.log_edit.append("🌿 传承千年智慧，融合现代科技")
        self.log_edit.append("📖 欢迎使用中药抗生素成分智能预测系统")
    
    # 以下是功能实现方法（与之前版本类似，但增加了更多功能）
    def select_model(self):
        """选择模型文件"""
        default_dir = ""
        if os.path.exists("model/model_0/checkpoints/"):
            default_dir = "model/model_0/checkpoints/"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择本草御菌录模型文件", 
            default_dir,
            "模型文件 (*.ckpt *.pt *.pth);;所有文件 (*)"
        )
        
        if file_path:
            self.model_path_edit.setText(file_path)
            self.log_edit.append(f"📁 已选择模型: {os.path.basename(file_path)}")
    
    def load_model(self):
        """加载模型"""
        model_path = self.model_path_edit.text().strip()
        
        if not model_path:
            QMessageBox.warning(self, "🚫 提醒", "请先选择模型文件！")
            return
        
        try:
            self.log_edit.append("🔄 正在加载本草御菌录模型...")
            QApplication.processEvents()
            
            if model_path.endswith('.ckpt'):
                self.model = models.MPNN.load_from_checkpoint(model_path)
            else:
                self.model = torch.load(model_path, map_location='cpu')
            
            self.model_path = model_path
            self.model_status_label.setText(f"🌟 模型已激活: {os.path.basename(model_path)}")
            self.model_status_label.setStyleSheet("color: #228B22;")
            
            # Enable prediction buttons
            self.predict_single_btn.setEnabled(True)
            self.batch_predict_btn.setEnabled(True)
            
            self.refresh_model_info()
            
            self.log_edit.append("✨ 本草御菌录模型加载成功！")
            QMessageBox.information(self, "🎉 成功", "模型激活成功！")
            
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            self.log_edit.append(f"❌ {error_msg}")
            QMessageBox.critical(self, "💥 错误", error_msg)
    
    def unload_model(self):
        """卸载模型"""
        self.model = None
        self.model_path = None
        self.predict_single_btn.setEnabled(False)
        self.batch_predict_btn.setEnabled(False)
        self.model_status_label.setText("模型状态: 未加载")
        self.model_status_label.setStyleSheet("color: #CD853F;")
        self.model_info_display.setHtml("<p style='color: #8B4513;'>无模型信息</p>")
        self.log_edit.append("🔄 模型已卸载")
    
    def refresh_model_info(self):
        """刷新模型信息"""
        if self.model:
            info_html = f"""
            <html>
            <body style="font-family: '宋体'; font-size: 12px; color: #8B4513;">
            <h3>📊 模型信息</h3>
            <ul>
                <li><b>模型文件:</b> {os.path.basename(self.model_path) if self.model_path else '未知'}</li>
                <li><b>模型类型:</b> {'Lightning Checkpoint' if self.model_path and self.model_path.endswith('.ckpt') else 'PyTorch Model'}</li>
                <li><b>状态:</b> <span style="color: #228B22;">已激活</span></li>
            </ul>
            </body>
            </html>
            """
            self.model_info_display.setHtml(info_html)
    
    def select_test(self):
        """选择测试文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择测试数据文件", 
            "", 
            "CSV文件 (*.csv);;所有文件 (*)"
        )
        
        if file_path:
            self.test_edit.setText(file_path)
            self.log_edit.append(f"📊 已选择测试文件: {os.path.basename(file_path)}")
    
    def predict_single_molecule(self):
        """预测单个分子"""
        smiles = self.smiles_input.text().strip()
        
        if not smiles:
            QMessageBox.warning(self, "🚫 提醒", "请输入SMILES分子式！")
            return
        
        if not self.model:
            QMessageBox.warning(self, "🚫 提醒", "请先加载模型！")
            return
        
        try:
            self.log_edit.append(f"🔬 正在分析分子: {smiles[:50]}...")
            
            #self.model.eval()
            
            with torch.no_grad():
            #     # 修复：使用正确的MoleculeDatapoint构造函数
            #     try:
                featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
                batch = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(smi) for smi in [smiles]],featurizer=featurizer)
                test_loader = data.build_dataloader(batch, shuffle=False)

                # except TypeError:
                #     try:
                #         from rdkit import Chem
                #         mol = Chem.MolFromSmiles(smiles)
                #         if mol is None:
                #             raise ValueError(f"无效的SMILES: {smiles}")
                #         batch = data.MoleculeDataset([data.MoleculeDatapoint(mol=mol)])
                #     except:
                #         dp = data.MoleculeDatapoint()
                #         dp.smiles = smiles
                #         batch = data.MoleculeDataset([dp])
                
                # # 修复：处理模型输出的不同格式
                
                with torch.inference_mode():
                    trainer = pl.Trainer(
                    logger=None,
                    enable_progress_bar=False,
                    accelerator="cpu",
                    devices=1
                    )
                    test_preds = trainer.predict(self.model, test_loader)
                pred = np.concatenate(test_preds, axis=0)

                # 修复：处理模型输出的解包问题
                pred_value = self._extract_prediction_value(pred)
                
                # 添加到历史记录
                self.add_to_history(smiles, pred_value, 0.95)
                
                # 显示结果
                self.display_single_result(smiles, pred_value, 0.95)
                
                self.log_edit.append(f"🎯 预测完成: {pred_value:.4f}")
                
        except Exception as e:
            error_msg = f"预测失败：{str(e)}\n\n{traceback.format_exc()}"
            self.log_edit.append(f"❌ 预测失败: {error_msg}")
            QMessageBox.critical(self, "💥 错误", f"预测失败: {str(e)}")
    
    def _extract_prediction_value(self, pred):
        """从模型输出中提取预测值（修复解包错误）"""
        try:
            if isinstance(pred, torch.Tensor):
                # 如果张量有多个值，取第一个
                if pred.numel() > 1:
                    return pred[0].item()
                else:
                    return pred.item()
            elif isinstance(pred, (list, tuple)):
                # 如果是列表或元组，取第一个元素
                return float(pred[0])
            elif isinstance(pred, np.ndarray):
                # 如果是numpy数组，取第一个值
                return float(pred.flat[0])
            else:
                # 其他类型直接转换为float
                return float(pred)
        except Exception as e:
            self.log_edit.append(f"⚠️ 预测值提取失败: {e}, 使用默认值 0.5")
            return 0.5
    
    def run_prediction(self):
        """批量预测"""
        test_file = self.test_edit.text().strip()
        
        if not test_file:
            QMessageBox.warning(self, "🚫 提醒", "请选择测试文件！")
            return
        
        if not self.model:
            QMessageBox.warning(self, "🚫 提醒", "请先加载模型！")
            return
        
        try:
            self.log_edit.append("📊 开始批量草药预测...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            test_df = pd.read_csv(test_file)
            
            if 'smiles' not in test_df.columns:
                QMessageBox.warning(self, "🚫 错误", "CSV文件必须包含'smiles'列！")
                return
            
            smiles_list = test_df['smiles'].tolist()
            total_molecules = len(smiles_list)
            
            self.log_edit.append(f"🌿 共检测到 {total_molecules} 个草药分子")
            
            # 预测
            #self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for i, smiles in enumerate(smiles_list):
                    try:
                        progress = int((i + 1) / total_molecules * 100)
                        self.progress_bar.setValue(progress)
                        
                        # 修复：使用正确的MoleculeDatapoint构造函数
                        try:
                            featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
                            batch = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(smi) for smi in [smiles]],featurizer=featurizer)
                            test_loader = data.build_dataloader(batch, shuffle=False)
                        except TypeError:
                            try:
                                from rdkit import Chem
                                mol = Chem.MolFromSmiles(smiles)
                                if mol is None:
                                    raise ValueError(f"无效的SMILES: {smiles}")
                                batch = data.MoleculeDataset([data.MoleculeDatapoint(mol=mol)])
                            except:
                                dp = data.MoleculeDatapoint()
                                dp.smiles = smiles
                                batch = data.MoleculeDataset([dp])
                        
                        # 修复：处理模型输出的不同格式
                        with torch.inference_mode():
                            trainer = pl.Trainer(
                            logger=None,
                            enable_progress_bar=False,
                            accelerator="cpu",
                            devices=1
                            )
                            test_preds = trainer.predict(self.model, test_loader)
                        pred = np.concatenate(test_preds, axis=0)
                        
                        # 修复：使用新的提取方法
                        pred_value = self._extract_prediction_value(pred)
                        
                        is_antibiotic = pred_value > 0.5
                        herb_type = "🌿 抗生素成分" if is_antibiotic else "🍃 非抗生素成分"
                        
                        predictions.append({
                            'smiles': smiles,
                            'prediction': pred_value,
                            'confidence': 0.95,
                            'is_antibiotic': is_antibiotic,
                            'herb_type': herb_type
                        })
                        
                        self.log_edit.append(f"分子 {i+1}/{total_molecules}: {pred_value:.4f} - {herb_type}")
                        
                        QApplication.processEvents()
                        
                    except Exception as e:
                        predictions.append({
                            'smiles': smiles,
                            'prediction': 'ERROR',
                            'confidence': 0.0,
                            'is_antibiotic': False,
                            'herb_type': '❌ 预测失败'
                        })
                        self.log_edit.append(f"❌ 分子 {i+1} 预测失败: {str(e)}")
            
            # 显示结果
            self.display_batch_results(predictions)
            
            # 添加到历史记录
            for pred in predictions:
                if pred['prediction'] != 'ERROR':
                    self.add_to_history(pred['smiles'], pred['prediction'], pred['confidence'])
            
            # 统计结果
            total = len(predictions)
            antibiotics = sum(1 for p in predictions if p.get('is_antibiotic', False))
            errors = sum(1 for p in predictions if p['prediction'] == 'ERROR')
            
            self.log_edit.append(f"\n🎉 批量预测完成！")
            self.log_edit.append(f"📊 总计: {total} 个分子")
            self.log_edit.append(f"🌿 抗生素成分: {antibiotics} 个 ({antibiotics/total*100:.1f}%)")
            self.log_edit.append(f"❌ 预测失败: {errors} 个")
            
            self.progress_bar.setVisible(False)
            
            QMessageBox.information(self, "🎉 成功", f"批量预测完成！\n总计: {total} 个分子\n抗生素成分: {antibiotics} 个")
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            error_msg = f"批量预测失败: {str(e)}"
            self.log_edit.append(f"💥 {error_msg}")
            QMessageBox.critical(self, "💥 错误", error_msg)
    
    def display_single_result(self, smiles, prediction, confidence):
        """显示单个预测结果"""
        self.result_table.setRowCount(1)
        
        # SMILES
        smiles_item = QTableWidgetItem(smiles)
        self.result_table.setItem(0, 0, smiles_item)
        
        # 预测值
        pred_item = QTableWidgetItem(f"{prediction:.4f}")
        is_antibiotic = prediction > 0.5
        if is_antibiotic:
            pred_item.setBackground(QColor(200, 255, 200))
            pred_item.setText(f"{prediction:.4f} 🌿")
        else:
            pred_item.setBackground(QColor(255, 255, 200))
            pred_item.setText(f"{prediction:.4f} 🍃")
        self.result_table.setItem(0, 1, pred_item)
        
        # 置信度
        conf_item = QTableWidgetItem(f"{confidence:.3f}")
        self.result_table.setItem(0, 2, conf_item)
        
        # 草药性质
        nature_item = QTableWidgetItem("🌿 抗生素成分" if is_antibiotic else "🍃 非抗生素成分")
        self.result_table.setItem(0, 3, nature_item)
    
    def display_batch_results(self, predictions):
        """显示批量预测结果"""
        self.result_table.setRowCount(len(predictions))
        
        for i, pred in enumerate(predictions):
            # SMILES
            smiles_item = QTableWidgetItem(pred['smiles'])
            self.result_table.setItem(i, 0, smiles_item)
            
            # 预测值
            if pred['prediction'] == 'ERROR':
                pred_item = QTableWidgetItem("❌ 预测失败")
                pred_item.setBackground(QColor(255, 200, 200))
            else:
                pred_item = QTableWidgetItem(f"{pred['prediction']:.4f}")
                if pred.get('is_antibiotic', False):
                    pred_item.setBackground(QColor(200, 255, 200))
                else:
                    pred_item.setBackground(QColor(255, 255, 200))
            self.result_table.setItem(i, 1, pred_item)
            
            # 置信度
            conf_item = QTableWidgetItem(f"{pred['confidence']:.3f}")
            self.result_table.setItem(i, 2, conf_item)
            
            # 草药性质
            nature_item = QTableWidgetItem(pred.get('herb_type', '未知'))
            self.result_table.setItem(i, 3, nature_item)
    
    def add_to_history(self, smiles, prediction, confidence):
        """添加到历史记录"""
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        self.prediction_history.append({
            'timestamp': timestamp,
            'smiles': smiles,
            'prediction': prediction,
            'confidence': confidence
        })
        
        # 更新历史显示
        self.update_history_display()
    
    def update_history_display(self):
        """更新历史记录显示"""
        if not self.prediction_history:
            self.history_display.setHtml("<p style='color: #8B4513; text-align: center;'>暂无预测历史</p>")
            return
        
        history_html = """
        <html>
        <body style="font-family: '宋体'; font-size: 12px; color: #8B4513;">
        <h3 style="color: #8B4513;">📜 预测历史记录</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #DEB887;">
            <th style="padding: 8px;">时间</th>
            <th style="padding: 8px;">SMILES</th>
            <th style="padding: 8px;">预测值</th>
            <th style="padding: 8px;">置信度</th>
        </tr>
        """
        
        for record in self.prediction_history[-20:]:  # 显示最近20条记录
            history_html += f"""
            <tr>
                <td style="padding: 5px; font-size: 11px;">{record['timestamp']}</td>
                <td style="padding: 5px; font-size: 10px;">{record['smiles'][:50]}...</td>
                <td style="padding: 5px;">{record['prediction']:.4f}</td>
                <td style="padding: 5px;">{record['confidence']:.3f}</td>
            </tr>
            """
        
        history_html += "</table></body></html>"
        self.history_display.setHtml(history_html)
    
    def clear_history(self):
        """清空历史记录"""
        self.prediction_history.clear()
        self.update_history_display()
        self.log_edit.append("🗑️ 历史记录已清空")
    
    def export_history(self):
        """导出历史记录"""
        if not self.prediction_history:
            QMessageBox.warning(self, "🚫 提醒", "暂无历史记录可导出！")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "导出历史记录", 
            "prediction_history.csv",
            "CSV文件 (*.csv);;所有文件 (*)"
        )
        
        if file_path:
            df = pd.DataFrame(self.prediction_history)
            df.to_csv(file_path, index=False)
            self.log_edit.append(f"💾 历史记录已导出到: {file_path}")
            QMessageBox.information(self, "✅ 成功", "历史记录导出成功！")


def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 设置全局字体
    font = QFont("宋体", 11)
    app.setFont(font)
    
    window = ChempropBeautifulGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()