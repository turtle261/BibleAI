import sys
import json
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QTextEdit, 
    QTabWidget, QComboBox, QCheckBox, QMessageBox, 
    QProgressBar, QSplitter, QFrame, QStyleFactory,
    QScrollArea, QMainWindow, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, QRunnable, QThreadPool, pyqtSignal, QObject, pyqtSlot, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QTextCursor
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomTextEdit(QTextEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QTextEdit {
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
                background-color: palette(base);
                selection-background-color: #666;
            }
        """)
        # Enable rich text rendering
        self.setAcceptRichText(True)
        # Set default font
        font = self.font()
        font.setFamily("Arial")
        self.setFont(font)
        
    def append_markdown(self, text: str):
        """Convert markdown to HTML and append to the display."""
        # Convert basic markdown to HTML
        html = self._markdown_to_html(text)
        # Append the HTML
        self.append(html)
        
    def _markdown_to_html(self, text: str) -> str:
        """Convert markdown formatting to HTML."""
        # Headers with improved styling
        text = re.sub(r'^# (.*?)$', r'<h1 style="color: #2b5b84; margin: 20px 0 10px 0; font-size: 24px;">\1</h1>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', r'<h2 style="color: #2b5b84; margin: 15px 0 8px 0; font-size: 20px;">\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.*?)$', r'<h3 style="color: #2b5b84; margin: 12px 0 6px 0; font-size: 16px;">\1</h3>', text, flags=re.MULTILINE)
        
        # Bold with color
        text = re.sub(r'\*\*(.*?)\*\*', r'<b style="color: #2b5b84;">\1</b>', text)
        text = re.sub(r'__(.*?)__', r'<b style="color: #2b5b84;">\1</b>', text)
        
        # Italic with subtle color
        text = re.sub(r'\*(.*?)\*', r'<i style="color: #666;">\1</i>', text)
        text = re.sub(r'_(.*?)_', r'<i style="color: #666;">\1</i>', text)
        
        # Code blocks with improved styling
        text = re.sub(r'`(.*?)`', r'<code style="background-color: rgba(43, 91, 132, 0.1); padding: 2px 4px; border-radius: 3px; font-family: monospace;">\1</code>', text)
        
        # Lists with proper spacing and bullets
        text = re.sub(r'^\* (.*?)$', r'<ul style="margin: 5px 0;"><li style="margin: 3px 0;">\1</li></ul>', text, flags=re.MULTILINE)
        text = re.sub(r'^\d\. (.*?)$', r'<ol style="margin: 5px 0;"><li style="margin: 3px 0;">\1</li></ol>', text, flags=re.MULTILINE)
        
        # Links with hover effect
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" style="color: #2b5b84; text-decoration: none; border-bottom: 1px solid #2b5b84;">\1</a>', text)
        
        # Blockquotes with improved styling
        text = re.sub(r'^> (.*?)$', r'<blockquote style="margin: 10px 0 10px 20px; padding: 10px; border-left: 3px solid #2b5b84; background-color: rgba(43, 91, 132, 0.05); color: #666;">\1</blockquote>', text, flags=re.MULTILINE)
        
        return text

class CustomLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QLineEdit {
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px;
                background-color: palette(base);
                selection-background-color: #666;
            }
            QLineEdit:focus {
                border: 1px solid #888;
            }
        """)

class CustomButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2b5b84;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #3d7ab8;
            }
            QPushButton:pressed {
                background-color: #1d4d74;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)

class CustomCheckBox(QCheckBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #555;
                background-color: palette(base);
            }
            QCheckBox::indicator:checked {
                background-color: #2b5b84;
                border: 2px solid #2b5b84;
            }
        """)

class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))

class BibleAIGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.conversation_id = None
        self.setWindowTitle("Bible AI - Intelligent Bible Study Assistant")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_ui()
        self.threadpool = QThreadPool()
        self.load_settings()
        self.init_connectors()
        
        # Set default theme
        self.apply_theme("Dark")

    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Header section
        header_layout = QVBoxLayout()
        
        # Input section with shadow effect
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_frame.setStyleSheet("""
            QFrame#inputFrame {
                background-color: palette(base);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        input_layout = QVBoxLayout(input_frame)

        # Query input
        self.input_field = CustomLineEdit()
        self.input_field.setPlaceholderText("Enter your Bible study question here...")
        self.input_field.returnPressed.connect(self.send_query)
        
        # Send button
        self.send_button = CustomButton("Send Query")
        self.send_button.clicked.connect(self.send_query)
        
        # Input row
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_field, stretch=1)
        input_row.addWidget(self.send_button)
        
        # Options row
        options_layout = QHBoxLayout()
        self.greek_checkbox = CustomCheckBox("Include Greek Analysis")
        self.complexity_checkbox = CustomCheckBox("Include Complexity Analysis")
        options_layout.addWidget(self.greek_checkbox)
        options_layout.addWidget(self.complexity_checkbox)
        options_layout.addStretch()

        input_layout.addLayout(input_row)
        input_layout.addLayout(options_layout)
        header_layout.addWidget(input_frame)

        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Q&A Section
        qna_widget = QWidget()
        qna_layout = QVBoxLayout(qna_widget)
        qna_layout.setContentsMargins(0, 0, 0, 0)
        
        qna_label = QLabel("Questions & Answers")
        qna_label.setStyleSheet("font-weight: bold; font-size: 14px; color: palette(text);")
        self.qna_display = CustomTextEdit()
        self.qna_display.setReadOnly(True)
        
        qna_layout.addWidget(qna_label)
        qna_layout.addWidget(self.qna_display)
        content_splitter.addWidget(qna_widget)

        # Bible Quotes Section
        quotes_widget = QWidget()
        quotes_layout = QVBoxLayout(quotes_widget)
        quotes_layout.setContentsMargins(0, 0, 0, 0)
        
        quotes_label = QLabel("Relevant Bible Quotes")
        quotes_label.setStyleSheet("font-weight: bold; font-size: 14px; color: palette(text);")
        self.quotes_display = CustomTextEdit()
        self.quotes_display.setReadOnly(True)
        
        quotes_layout.addWidget(quotes_label)
        quotes_layout.addWidget(self.quotes_display)
        content_splitter.addWidget(quotes_widget)

        content_splitter.setSizes([600, 600])
        content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555;
                width: 2px;
            }
        """)

        # Settings section
        self.setup_settings_panel()

        # Add all components to main layout
        main_layout.addLayout(header_layout)
        main_layout.addWidget(content_splitter, stretch=1)
        main_layout.addWidget(self.settings_panel)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                border-top: 1px solid #555;
                padding: 5px;
                background-color: palette(base);
                color: palette(text);
            }
        """)

    def setup_settings_panel(self):
        self.settings_panel = QFrame()
        self.settings_panel.setObjectName("settingsPanel")
        self.settings_panel.setStyleSheet("""
            QFrame#settingsPanel {
                background-color: palette(base);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        settings_layout = QHBoxLayout(self.settings_panel)
        
        # Theme selection
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setStyleSheet("color: palette(text);")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["System", "Light", "Dark"])
        self.theme_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        
        # Font size selection
        font_layout = QHBoxLayout()
        font_label = QLabel("Font Size:")
        font_label.setStyleSheet("color: palette(text);")
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Small", "Medium", "Large"])
        self.font_combo.setStyleSheet(self.theme_combo.styleSheet())
        font_layout.addWidget(font_label)
        font_layout.addWidget(self.font_combo)
        
        # Save button
        self.save_button = CustomButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        
        # Add layouts to settings panel
        settings_layout.addLayout(theme_layout)
        settings_layout.addSpacing(20)
        settings_layout.addLayout(font_layout)
        settings_layout.addSpacing(20)
        settings_layout.addWidget(self.save_button)
        settings_layout.addStretch()

    def apply_theme(self, theme_name):
        if theme_name == "Dark":
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
            dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
            dark_palette.setColor(QPalette.Active, QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
            dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(128, 128, 128))
            dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
            dark_palette.setColor(QPalette.Disabled, QPalette.Light, QColor(53, 53, 53))
            self.setPalette(dark_palette)
        elif theme_name == "Light":
            self.setPalette(self.style().standardPalette())
        else:
            self.setPalette(QApplication.style().standardPalette())

    def init_connectors(self):
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        self.font_combo.currentTextChanged.connect(self.change_font_size)

    def load_settings(self):
        try:
            with open("data/settings.json", "r") as f:
                settings_data = json.load(f)
                self.theme_combo.setCurrentText(settings_data.get("theme", "Dark"))
                self.font_combo.setCurrentText(settings_data.get("font_size", "Medium"))
                self.api_url = settings_data.get("api_url", "http://localhost:8000")
                self.apply_theme(settings_data.get("theme", "Dark"))
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            self.api_url = "http://localhost:8000"
            self.apply_theme("Dark")

    def save_settings(self):
        settings_data = {
            "theme": self.theme_combo.currentText(),
            "font_size": self.font_combo.currentText(),
            "api_url": self.api_url
        }
        try:
            with open("data/settings.json", "w") as f:
                json.dump(settings_data, f, indent=4)
            self.status_bar.showMessage("Settings saved successfully", 3000)
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            QMessageBox.critical(self, "Error", "Failed to save settings.")

    def change_font_size(self):
        size_map = {'Small': 10, 'Medium': 12, 'Large': 14}
        size = size_map.get(self.font_combo.currentText(), 12)
        
        font = QFont()
        font.setPointSize(size)
        QApplication.setFont(font)
        
        # Update font size for text displays
        self.qna_display.setFont(font)
        self.quotes_display.setFont(font)

    def send_query(self):
        question = self.input_field.text().strip()
        if not question:
            QMessageBox.warning(self, "Input Error", "Please enter a question.")
            return

        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.status_bar.showMessage("Processing query...")

        include_greek = self.greek_checkbox.isChecked()
        analyze_complexity = self.complexity_checkbox.isChecked()

        worker = Worker(self.query_api, question, include_greek, analyze_complexity)
        worker.signals.finished.connect(self.on_query_finished)
        worker.signals.error.connect(self.on_query_error)
        self.threadpool.start(worker)

    def query_api(self, question, include_greek, analyze_complexity):
        payload = {
            "question": question,
            "include_greek": include_greek,
            "analyze_complexity": analyze_complexity
        }

        response = requests.post(f"{self.api_url}/query", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(response.text)

    def on_query_finished(self, result):
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.status_bar.showMessage("Query completed", 3000)

        # Format and display the answer
        answer = result.get("answer", "")
        verses = result.get("verses", [])
        thoughts = result.get("thoughts", [])
        
        # Display agent thoughts
        thought_text = (
            '<div style="margin-bottom: 15px; padding: 10px; background-color: rgba(43, 91, 132, 0.1); '
            'border-radius: 5px;">'
            '<p style="color: #2b5b84;"><b>Agent Thoughts:</b></p>'
        )
        for thought in thoughts:
            thought_text += f'<p>• {thought}</p>'
        thought_text += '</div>'
        
        # Update Q&A display with formatted text
        qa_text = (
            f'{thought_text}'
            f'<div style="margin-bottom: 10px;">'
            f'<p style="color: #2b5b84;"><b>Q: {self.input_field.text()}</b></p>'
            f'<p><b>A:</b></p>'
        )
        qa_text += self.qna_display._markdown_to_html(answer)
        qa_text += '</div>'
        
        self.qna_display.append(qa_text)
        self.qna_display.moveCursor(QTextCursor.End)
        
        # Update quotes display
        self.update_quotes_display(verses)
        
        self.input_field.clear()

    def update_quotes_display(self, verses):
        """Update quotes display in a more efficient way."""
        self.quotes_display.clear()
        html_chunks = []
        
        for verse in verses:
            chunk = [
                f'<div style="margin-bottom: 20px; padding: 15px; border-radius: 5px; '
                f'background-color: rgba(43, 91, 132, 0.02);">'
                f'<p style="color: #2b5b84; font-size: 16px; margin-bottom: 10px;">'
                f'<b>{verse["book"]} {verse["chapter"]}:{verse["verse"]}</b></p>'
                f'<p style="margin-bottom: 10px;">{self.quotes_display._markdown_to_html(verse["text"])}</p>'
            ]
            
            if verse.get("greek_text"):
                chunk.append(
                    f'<div style="margin: 10px 0; padding: 10px; background-color: rgba(43, 91, 132, 0.05); '
                    f'border-left: 3px solid #2b5b84; border-radius: 0 5px 5px 0;">'
                    f'<p style="color: #2b5b84;"><b>Greek Text:</b></p>'
                    f'<p style="font-style: italic;">{self.quotes_display._markdown_to_html(verse["greek_text"])}</p>'
                    f'</div>'
                )
                
            if verse.get("complexity_score") is not None:
                chunk.append(
                    f'<div style="margin: 15px 0; padding: 10px; background-color: rgba(43, 91, 132, 0.05); '
                    f'border-radius: 5px;">'
                    f'<p style="color: #2b5b84; margin-bottom: 10px;"><b>Complexity Analysis</b></p>'
                    f'<p>Complexity Score: <b>{verse["complexity_score"]:.3f}</b></p>'
                )
                
                if verse.get("similar_by_complexity"):
                    chunk.append(
                        f'<div style="margin-top: 10px;">'
                        f'<p style="color: #2b5b84;"><b>Similar Verses by Complexity:</b></p>'
                    )
                    
                    eng_verses = [v for v in verse["similar_by_complexity"] if v.get("match_type") == "english"]
                    greek_verses = [v for v in verse["similar_by_complexity"] if v.get("match_type") == "greek"]
                    
                    if eng_verses:
                        chunk.append(
                            f'<div style="margin: 10px 0;">'
                            f'<p style="color: #2b5b84;"><i>Similar in English Structure:</i></p>'
                        )
                        for similar in eng_verses:
                            chunk.append(
                                f'<div style="margin: 5px 0 10px 15px; padding: 8px; '
                                f'background-color: rgba(43, 91, 132, 0.03); border-radius: 3px;">'
                                f'<p style="color: #2b5b84;"><b>{similar["book"]} {similar["chapter"]}:'
                                f'{similar["verse"]}</b></p>'
                                f'<p>{self.quotes_display._markdown_to_html(similar["text"])}</p>'
                                f'</div>'
                            )
                        chunk.append('</div>')
                        
                    if greek_verses:
                        chunk.append(
                            f'<div style="margin: 10px 0;">'
                            f'<p style="color: #2b5b84;"><i>Similar in Greek Structure:</i></p>'
                        )
                        for similar in greek_verses:
                            chunk.append(
                                f'<div style="margin: 5px 0 10px 15px; padding: 8px; '
                                f'background-color: rgba(43, 91, 132, 0.03); border-radius: 3px;">'
                                f'<p style="color: #2b5b84;"><b>{similar["book"]} {similar["chapter"]}:'
                                f'{similar["verse"]}</b></p>'
                                f'<p>{self.quotes_display._markdown_to_html(similar["text"])}</p>'
                                f'</div>'
                            )
                        chunk.append('</div>')
                    
                    chunk.append('</div>')
                chunk.append('</div>')
                    
            chunk.append('</div>')
            html_chunks.append(''.join(chunk))
        
        self.quotes_display.setHtml(''.join(html_chunks))
        self.quotes_display.moveCursor(QTextCursor.Start)

    def on_query_error(self, error):
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.status_bar.showMessage("Error processing query", 3000)
        QMessageBox.critical(self, "Error", str(error))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    ex = BibleAIGUI()
    ex.show()
    sys.exit(app.exec_())
