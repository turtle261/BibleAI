import sys
import json
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QTextEdit, 
    QScrollArea, QTabWidget, QComboBox, QCheckBox, 
    QMessageBox, QFileDialog, QProgressBar, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, QRunnable, QThreadPool, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPalette, QColor
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkerSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(tuple)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit((e, traceback.format_exc()))
        else:
            self.signals.result.emit(result)

class VerseWidget(QWidget):
    def __init__(self, verse):
        super().__init__()
        layout = QVBoxLayout()
        reference = QLabel(f"<b>{verse['book']} {verse['chapter']}:{verse['verse']}</b>")
        text = QLabel(verse['text'])
        text.setWordWrap(True)
        layout.addWidget(reference)
        layout.addWidget(text)
        self.setLayout(layout)
       
class BibleAIGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.conversation_id = None
        self.setWindowTitle("Optimized FOSS Bible AI")
        self.setGeometry(100, 100, 1000, 600)
        self.setup_ui()
        self.threadpool = QThreadPool()
        self.load_settings()
        self.init_connectors()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        # Top layout for input and buttons
        top_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter your question here...")
        self.input_field.returnPressed.connect(self.send_query)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_query)
        top_layout.addWidget(self.input_field)
        top_layout.addWidget(self.send_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Splitter for Q&A and Relevant Quotes
        splitter = QSplitter(Qt.Horizontal)

        # Q&A Section
        qna_layout = QVBoxLayout()
        self.qna_display = QTextEdit()
        self.qna_display.setReadOnly(True)
        qna_layout.addWidget(QLabel("<b>Q&A</b>"))
        qna_layout.addWidget(self.qna_display)
        qna_widget = QWidget()
        qna_widget.setLayout(qna_layout)
        splitter.addWidget(qna_widget)

        # Relevant Quotes Section
        quotes_layout = QVBoxLayout()
        self.quotes_display = QTextEdit()
        self.quotes_display.setReadOnly(True)
        quotes_layout.addWidget(QLabel("<b>Relevant Bible Quotes</b>"))
        quotes_layout.addWidget(self.quotes_display)
        quotes_widget = QWidget()
        quotes_widget.setLayout(quotes_layout)
        splitter.addWidget(quotes_widget)

        splitter.setSizes([600, 400])

        # Tabs for Q&A and Settings
        self.tabs = QTabWidget()
        self.tab_qna = QWidget()
        self.tab_settings = QWidget()
        self.setup_qna_tab(splitter)
        self.setup_settings_tab()
        self.tabs.addTab(self.tab_qna, "Q&A & Quotes")
        self.tabs.addTab(self.tab_settings, "Settings")

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def setup_qna_tab(self, splitter):
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.tab_qna.setLayout(layout)
       
    def setup_settings_tab(self):
        layout = QVBoxLayout()

        # Theme selection
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["System Default", "Light", "Dark"])
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        layout.addLayout(theme_layout)

        # Font size selection
        font_layout = QHBoxLayout()
        font_label = QLabel("Font Size:")
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Small", "Medium", "Large"])
        font_layout.addWidget(font_label)
        font_layout.addWidget(self.font_combo)
        layout.addLayout(font_layout)

        # Save settings button
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        layout.addWidget(self.save_button)

        # Spacer to push elements to the top
        layout.addStretch()

        self.tab_settings.setLayout(layout)

    def init_connectors(self):
        # Connect theme and font size changes
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        self.font_combo.currentTextChanged.connect(self.change_font_size)

    def send_query(self):
        question = self.input_field.text().strip()
        if not question:
            QMessageBox.warning(self, "Input Error", "Please enter a question.")
            return

        self.qna_display.append(f"<p><b>You:</b> {question}</p>")
        self.input_field.clear()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        worker = Worker(self.query_api, question)
        worker.signals.result.connect(self.display_response)
        worker.signals.error.connect(self.handle_error)
        self.threadpool.start(worker)
       
    def query_api(self, question):
        payload = {"question": question}
        try:
            # Increase the timeout duration to 120 seconds
            response = requests.post("http://localhost:8000/query", json=payload, timeout=150)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise Exception("The request timed out. Please try again later or check the server's performance.")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to the backend API. Ensure it's running.")
        except Exception as e:
            raise Exception(f"An error occurred: {e}")

    def display_response(self, data):
        answer = data.get("answer", "No answer provided.")
        verses = data.get("verses", [])
        formatted_answer = f"<div style='background-color:#e0e0e0; padding:10px; border-radius:5px; margin:10px 0;'><b>AI:</b> {answer}</div>"
        self.qna_display.append(formatted_answer)

        # Clear previous quotes
        self.quotes_display.clear()

        for verse in verses:
            verse_html = (
                f"<div style='background-color:#f0f8ff; padding:10px; border-radius:5px; margin:5px 0;'>"
                f"<b>{verse['book']} {verse['chapter']}:{verse['verse']}</b>: {verse['text']}"
                f"</div>"
            )
            self.quotes_display.append(verse_html)

        self.progress_bar.setVisible(False)
        self.tabs.setCurrentWidget(self.tab_qna)

    def handle_error(self, error):
        e, traceback_str = error
        logger.error(f"Error in GUI thread: {e}\n{traceback_str}")
        QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        self.progress_bar.setVisible(False)

    def change_theme(self):
        theme = self.theme_combo.currentText()
        palette = QPalette()
        if theme == "Dark":
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.black)
        elif theme == "Light":
            palette = QApplication.style().standardPalette()
        else:
            palette = QApplication.style().standardPalette()
        QApplication.setPalette(palette)

    def change_font_size(self):
        size = self.font_combo.currentText()
        if size == 'Small':
            font_size = 10
        elif size == 'Medium':
            font_size = 12
        else:  # Large
            font_size = 14

        font = QFont()
        font.setPointSize(font_size)
        QApplication.setFont(font)

    def save_settings(self):
        settings = {
            "theme": self.theme_combo.currentText(),
            "font_size": self.font_combo.currentText()
        }
        try:
            with open("settings.json", "w", encoding='utf-8') as f:
                json.dump(settings, f)
            QMessageBox.information(self, "Settings Saved", "Your settings have been saved.")
            self.tabs.setCurrentWidget(self.tab_qna)
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Could not save settings: {e}")

    def load_settings(self):
        try:
            with open("settings.json", "r", encoding='utf-8') as f:
                settings = json.load(f)
            self.theme_combo.setCurrentText(settings.get("theme", "System Default"))
            self.font_combo.setCurrentText(settings.get("font_size", "Medium"))
            self.change_theme()
            self.change_font_size()
        except FileNotFoundError:
            pass  # Use default settings if file not found
        except Exception as e:
            QMessageBox.warning(self, "Load Failed", f"Could not load settings: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BibleAIGUI()
    ex.show()
    sys.exit(app.exec_())
