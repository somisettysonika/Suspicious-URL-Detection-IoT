# ==========================================================
#  Enhanced Url_Detection Dashboard
#  Url_Detection Monitoring with Advanced Features
#  Developed by RAJ SHARMA - Enhanced Version
# ==========================================================
import sys
import os
import re
import random
import datetime
import smtplib
import ssl
import socket
import whois
import requests
from urllib.parse import urlparse
from email.message import EmailMessage

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import joblib
import pyttsx3
import speech_recognition as sr

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import (
    Qt,
    QTimer,
    QUrl,
    QThread,
    pyqtSignal,
    QEasingCurve,
    QPropertyAnimation,
)
from PyQt6.QtGui import (
    QFont,
    QAction,
    QPixmap,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTabWidget,
    QFileDialog,
    QMessageBox,
    QTextEdit,
    QScrollArea,
    QFrame,
    QComboBox,
    QDialog,
    QFormLayout,
    QGraphicsOpacityEffect,
)

# -------------------------------------------------------------
# Optional video support
# -------------------------------------------------------------
try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    HAS_VIDEO = True
except Exception:
    HAS_VIDEO = False

sns.set_style("whitegrid")

# =============================================================
#  TTS Thread
# =============================================================
class TTSWorker(QThread):
    text = ""

    def run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.setProperty("volume", 1.0)
            engine.say(self.text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"TTS Error: {e}")


# =============================================================
#  Voice Command Worker
# =============================================================
class VoiceWorker(QThread):
    commandDetected = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        recognizer = sr.Recognizer()
        try:
            mic = sr.Microphone()
        except Exception as e:
            self.errorOccurred.emit(f"Microphone error: {e}")
            return

        while self._running:
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

                command = recognizer.recognize_google(audio, language="en-IN").lower()
                self.commandDetected.emit(command)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                self.errorOccurred.emit("Could not understand. Try again.")
            except Exception as e:
                self.errorOccurred.emit(f"Voice error: {e}")
                break


# =============================================================
#  URL FEATURE EXTRACTION ENGINE
# =============================================================
class URLFeatureExtractor:

    def extract(self, url: str) -> dict:
        """Extract full feature set from a given URL."""

        parsed = urlparse(url)

        # -------------------------------
        # Feature 1 – Total URL Length
        # -------------------------------
        length = len(url)

        # -------------------------------
        # Feature 2 – Count Digits
        # -------------------------------
        num_digits = sum(c.isdigit() for c in url)

        # -------------------------------
        # Feature 3 – Count Special Characters
        # -------------------------------
        num_special = sum(not c.isalnum() for c in url)

        # -------------------------------
        # Feature 4 – Contains HTTPS
        # -------------------------------
        contains_https = 1 if url.startswith("https") else 0

        # -------------------------------
        # Feature 5 – Domain has IP?
        # -------------------------------
        domain = parsed.netloc
        has_ip = 1 if self._looks_like_ip(domain) else 0

        # -------------------------------
        # Feature 6 – Domain Age
        # -------------------------------
        domain_age = self._get_domain_age(domain)

        # -------------------------------
        # Feature 7 – Alexa Rank
        # -------------------------------
        alexa = self._get_alexa_rank(domain)

        return {
            "length": length,
            "num_digits": num_digits,
            "num_special_chars": num_special,
            "contains_https": contains_https,
            "has_ip": has_ip,
            "domain_age": domain_age,
            "alexa_rank": alexa,
        }

    # ---------------------------------------------------------
    # Helper: Check if domain looks like an IP
    # ---------------------------------------------------------
    def _looks_like_ip(self, domain):
        pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
        return 1 if re.match(pattern, domain) else 0

    # ---------------------------------------------------------
    # Helper: Domain age in days
    # ---------------------------------------------------------
    def _get_domain_age(self, domain):
        try:
            info = whois.whois(domain)
            creation = info.creation_date
            if creation is None:
                return 0
            if isinstance(creation, list):
                creation = creation[0]
            age_days = (datetime.datetime.now() - creation).days
            return max(1, age_days)
        except:
            return 0

    # ---------------------------------------------------------
    # Helper: Alexa Rank (dummy fallback)
    # ---------------------------------------------------------
    def _get_alexa_rank(self, domain):
        """Alexa API is discontinued, so return synthetic values."""
        try:
            # Fake simulation:
            if "." not in domain:
                return 999999
            return random.randint(1000, 900000)
        except:
            return 999999


# =============================================================
#  HYBRID FEATURE SELECTION (Filter + Wrapper + GA)
# =============================================================
class HybridFeatureSelectorGA:

    def __init__(self, population=12, generations=10, mutation=0.2):
        self.population = population
        self.generations = generations
        self.mutation_rate = mutation

    # ---------------------------------------------------------
    # Fit selector
    # ---------------------------------------------------------
    def select(self, X, y):
        feature_names = list(X.columns)
        num_features = X.shape[1]

        # ---------------------------------------------
        # FILTER STAGE (Top-K selection)
        # ---------------------------------------------
        chi_scores, _ = chi2(X, y)
        mi_scores = mutual_info_classif(X, y)

        combined_score = chi_scores + mi_scores
        top_k = np.argsort(combined_score)[::-1][:num_features]   # keep all for now

        X_filtered = X.iloc[:, top_k]

        # ---------------------------------------------
        # WRAPPER STAGE (RFE)
        # ---------------------------------------------
        model = RandomForestClassifier(n_estimators=50)
        rfe = RFE(model, n_features_to_select=int(num_features * 0.7))
        rfe.fit(X_filtered, y)

        rfe_mask = rfe.support_
        X_rfe = X_filtered.loc[:, rfe_mask]
        rfe_features = list(X_rfe.columns)

        # ---------------------------------------------
        # GENETIC ALGORITHM STAGE
        # ---------------------------------------------
        best_features = self._genetic_algorithm(X_rfe, y)

        return best_features

    # ---------------------------------------------------------
    # Genetic Algorithm
    # ---------------------------------------------------------
    def _genetic_algorithm(self, X, y):

        feature_names = list(X.columns)
        num_features = len(feature_names)

        # Population initialization (binary masks)
        population = [
            np.random.randint(0, 2, num_features) for _ in range(self.population)
        ]

        for gen in range(self.generations):

            fitness_scores = [
                self._fitness(ind, X, y) for ind in population
            ]

            # Select best 50%
            sorted_idx = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_idx[: len(population)//2]]

            # Crossover
            offspring = []
            while len(offspring) < self.population - len(population):
                parent1, parent2 = random.sample(population, 2)
                cross_point = random.randint(1, num_features - 1)
                child = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                offspring.append(child)

            population.extend(offspring)

            # Mutation
            for ind in population:
                if random.random() < self.mutation_rate:
                    m = random.randint(0, num_features - 1)
                    ind[m] = 1 - ind[m]

        # Final best feature mask
        final_scores = [
            self._fitness(ind, X, y) for ind in population
        ]
        best = population[np.argmax(final_scores)]

        return [feature_names[i] for i in range(num_features) if best[i] == 1]

    # ---------------------------------------------------------
    # Fitness = ML accuracy
    # ---------------------------------------------------------
    def _fitness(self, mask, X, y):
        selected_cols = [col for col, m in zip(X.columns, mask) if m == 1]
        if len(selected_cols) == 0:
            return 0
        X_sel = X[selected_cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(n_estimators=40)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return accuracy_score(y_test, pred)

# =============================================================
# Part 2/5 – GUI Layout: Main Window, Header, Left Panel
# =============================================================

class SuspiciousURLDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window
        self.setWindowTitle("Suspicious URL Detection in IoT Using Hybrid Feature Selection")
        self.resize(1600, 900)
        self.setMinimumSize(1200, 700)

        # State
        self.current_dataset = None
        self.model = None
        self.scaler = None
        self.feature_selector = HybridFeatureSelectorGA(population=12, generations=10, mutation=0.2)
        self.extractor = URLFeatureExtractor()
        self.history = []  # rows of predictions
        self.feature_order = [
            "length",
            "num_digits",
            "num_special_chars",
            "has_ip",
            "contains_https",
            "domain_age",
            "alexa_rank",
        ]

        # UI containers (will be created in _init_ui)
        self.url_field = None
        self.feature_fields = {}
        self.result_label = None
        self.status_label = None
        self.tabs = None

        self._init_ui()
        self._apply_style()

    # ------------------- Build UI -------------------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = self._build_header()
        main_layout.addWidget(header)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(10, 10, 10, 10)
        body_layout.setSpacing(10)

        self.left_panel = self._build_left_panel()
        self.center_panel = self._build_center_panel()
        self.right_panel = self._build_right_panel()

        body_layout.addWidget(self.left_panel, 0)
        body_layout.addWidget(self.center_panel, 3)
        body_layout.addWidget(self.right_panel, 0)

        main_layout.addWidget(body)

        self._create_menu_bar()

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(90)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(12)

        # Logo
        logo = QLabel()
        logo.setFixedSize(64, 64)
        pix = QPixmap("assets/cloud_shield.png")
        if not pix.isNull():
            logo.setPixmap(pix.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        else:
            logo.setText("🛡️")
            logo.setFont(QFont("Segoe UI Emoji", 30))
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Titles
        title_container = QWidget()
        t_layout = QVBoxLayout(title_container)
        t_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Suspicious URL Detection in IoT Using Hybrid Feature Selection")
        title.setObjectName("HeaderTitle")
        subtitle = QLabel("IoT URL security · Hybrid Feature Selection · GA-Optimized Model")
        subtitle.setObjectName("HeaderSubtitle")

        t_layout.addWidget(title)
        t_layout.addWidget(subtitle)

        # Meta
        meta = QLabel("PyQt6 · Research Prototype")
        meta.setObjectName("HeaderMeta")

        layout.addWidget(logo)
        layout.addWidget(title_container, 1)
        layout.addWidget(meta, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        return header

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("LeftPanel")
        panel.setMinimumWidth(360)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Data & Model management
        section_title = QLabel("📊 DATA & MODEL")
        section_title.setObjectName("SectionTitle")
        layout.addWidget(section_title)

        def add_primary(text, slot):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btn.setObjectName("PrimaryButton")
            layout.addWidget(btn)

        add_primary("📁 Load Dataset", self.load_dataset)
        add_primary("🧠 Load Model", self.load_model)
        add_primary("🎯 Train New Model (Hybrid FS + GA)", self.train_model_dialog)
        add_primary("💾 Save Model", self.save_model)

        layout.addSpacing(12)

        # URL input
        url_title = QLabel("🔗 URL ANALYSIS")
        url_title.setObjectName("SectionTitle")
        layout.addWidget(url_title)

        url_row = QWidget()
        url_row_layout = QHBoxLayout(url_row)
        url_row_layout.setContentsMargins(0, 0, 0, 0)

        url_label = QLabel("URL:")
        url_label.setFixedWidth(40)
        self.url_field = QLineEdit()
        self.url_field.setPlaceholderText("http://example.com/path...")

        url_row_layout.addWidget(url_label)
        url_row_layout.addWidget(self.url_field)
        layout.addWidget(url_row)

        extract_btn = QPushButton("⚙️ Extract Features")
        extract_btn.setObjectName("SecondaryButton")
        extract_btn.clicked.connect(self._on_extract_clicked)
        layout.addWidget(extract_btn)

        layout.addSpacing(8)

        feat_title = QLabel("🔍 Extracted Features (editable)")
        feat_title.setObjectName("SectionTitle")
        layout.addWidget(feat_title)

        # Feature fields
        self.feature_fields = {}
        for f in self.feature_order:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f + ":")
            label.setFixedWidth(150)
            entry = QLineEdit()
            entry.setPlaceholderText("Auto-filled after extraction")
            row_layout.addWidget(label)
            row_layout.addWidget(entry)

            self.feature_fields[f] = entry
            layout.addWidget(row)

        layout.addSpacing(10)

        # Result / status
        self.result_label = QLabel("Ready")
        self.result_label.setObjectName("ResultLabel")
        layout.addWidget(self.result_label)

        self.status_label = QLabel("STATUS: Idle")
        self.status_label.setObjectName("StatusLabel")
        layout.addWidget(self.status_label)

        layout.addSpacing(10)

        # Actions
        actions_title = QLabel("⚡ ACTIONS")
        actions_title.setObjectName("SectionTitle")
        layout.addWidget(actions_title)

        def add_action(text, slot):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btn.setObjectName("SecondaryButton")
            layout.addWidget(btn)

        add_action("🔍 Predict URL Safety", self.predict_url)
        add_action("📁 Export CSV", self.export_csv)
        add_action("📤 Email Report", self.send_email_popup)
        add_action("🔊 Speak Result", self.speak_result)
        add_action("🎤 Voice Command", self.start_voice_mode)

        layout.addStretch(1)
        v.addWidget(scroll)
        return panel

    # ---------------- Center & Right placeholders (will be filled in Part 3)
    def _build_center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("CenterPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        # create tab placeholders; actual plotting setup in Part 3
        self.tab_time_series = QWidget()
        self.tab_pie = QWidget()
        self.tab_bar = QWidget()
        self.tab_heatmap = QWidget()

        self.tabs.addTab(self.tab_time_series, "📈 Time Series")
        self.tabs.addTab(self.tab_pie, "🥧 Distribution")
        self.tabs.addTab(self.tab_bar, "📊 Features")
        self.tabs.addTab(self.tab_heatmap, "🔥 Correlation")

        layout.addWidget(self.tabs)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("RightPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("📊 STATISTICS")
        title.setObjectName("SectionTitleBig")
        layout.addWidget(title)

        self.stats_container = QWidget()
        stats_layout = QVBoxLayout(self.stats_container)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(6)

        self.stat_total = self._create_stat_card(stats_layout, "Total Predictions", "0", "📈")
        self.stat_latest = self._create_stat_card(stats_layout, "Latest Result", "N/A", "🛡️")
        self.stat_susp_rate = self._create_stat_card(stats_layout, "Suspicious Rate", "0.0%", "🚨")

        layout.addWidget(self.stats_container)

        history_title = QLabel("📜 HISTORY")
        history_title.setObjectName("SectionTitle")
        layout.addWidget(history_title)

        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setObjectName("HistoryText")
        layout.addWidget(self.history_text, 1)

        return panel

    def _create_stat_card(self, parent_layout, title, value, icon) -> QLabel:
        card = QFrame()
        card.setObjectName("StatCard")
        grid = QGridLayout(card)
        grid.setContentsMargins(10, 8, 10, 8)

        label_title = QLabel(f"{icon} {title}")
        label_title.setObjectName("StatTitle")
        label_value = QLabel(value)
        label_value.setObjectName("StatValue")

        grid.addWidget(label_title, 0, 0)
        grid.addWidget(label_value, 1, 0)

        parent_layout.addWidget(card)
        return label_value

    # ----------------- Menu & Style helpers -----------------
    def _create_menu_bar(self):
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0E1117; }
            #Header { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1F2937, stop:1 #111827); border-bottom: 1px solid #111827; }
            #HeaderTitle { color: #E5E7EB; font-size: 18px; font-weight: 700; }
            #HeaderSubtitle { color: #9CA3AF; font-size: 12px; }
            #HeaderMeta { color: #6EE7B7; font-size: 11px; }
            #LeftPanel, #RightPanel, #CenterPanel { background-color: #111827; border-radius: 12px; }
            QLabel#SectionTitle { color: #9CA3AF; font-size: 12px; font-weight: 600; }
            QLabel#SectionTitleBig { color: #E5E7EB; font-size: 14px; font-weight: 700; }
            QPushButton#PrimaryButton { background-color: #10B981; color: white; border-radius: 8px; padding: 6px 10px; font-size: 12px; }
            QPushButton#PrimaryButton:hover { background-color: #059669; }
            QPushButton#SecondaryButton { background-color: #1F2937; color: #E5E7EB; border-radius: 8px; padding: 6px 10px; font-size: 12px; border: 1px solid #374151; }
            QLineEdit { background-color: #020617; border-radius: 6px; border: 1px solid #374151; color: #E5E7EB; padding: 4px 6px; font-size: 11px; }
            QLabel#ResultLabel { color: #22C55E; font-size: 14px; font-weight: 700; }
            QLabel#StatusLabel { color: #38BDF8; font-size: 11px; }
            #StatCard { background-color: #020617; border-radius: 10px; border: 1px solid #111827; }
            QLabel#StatTitle { color: #9CA3AF; font-size: 11px; }
            QLabel#StatValue { color: #F9FAFB; font-size: 16px; font-weight: 700; }
            QTextEdit#HistoryText { background-color: #020617; color: #E5E7EB; border-radius: 8px; border: 1px solid #111827; font-family: Consolas, monospace; font-size: 10px; }
            QTabWidget::pane { border: none; }
            QTabBar::tab { background-color: #020617; color: #9CA3AF; padding: 6px 10px; margin-right: 2px; border-top-left-radius: 8px; border-top-right-radius: 8px; }
            QTabBar::tab:selected { background-color: #111827; color: #E5E7EB; }
        """)
# =============================================================
# Part 3/5 – Plots, Dataset/Model I/O, Training, Prediction, Stats
# =============================================================

    # ----------------- Plot setup (called from _build_center_panel) -----------------
    def _setup_plots_widgets(self):
        # Time series tab
        v_ts = QVBoxLayout(self.tab_time_series)
        self.fig_ts, self.ax_ts = plt.subplots(figsize=(6, 4))
        self.fig_ts.patch.set_facecolor("#2B2D3A")
        self.ax_ts.set_facecolor("#1E1E26")
        self.canvas_ts = FigureCanvas(self.fig_ts)
        v_ts.addWidget(self.canvas_ts)

        # Distribution (pie)
        v_pie = QVBoxLayout(self.tab_pie)
        self.fig_pie, self.ax_pie = plt.subplots(figsize=(6, 4))
        self.fig_pie.patch.set_facecolor("#2B2D3A")
        self.ax_pie.set_facecolor("#2B2D3A")
        self.canvas_pie = FigureCanvas(self.fig_pie)
        v_pie.addWidget(self.canvas_pie)

        # Feature bar chart
        v_bar = QVBoxLayout(self.tab_bar)
        self.fig_bar, self.ax_bar = plt.subplots(figsize=(6, 4))
        self.fig_bar.patch.set_facecolor("#2B2D3A")
        self.ax_bar.set_facecolor("#1E1E26")
        self.canvas_bar = FigureCanvas(self.fig_bar)
        v_bar.addWidget(self.canvas_bar)

        # Correlation heatmap
        v_heat = QVBoxLayout(self.tab_heatmap)
        self.fig_heat, self.ax_heat = plt.subplots(figsize=(6, 4))
        self.fig_heat.patch.set_facecolor("#2B2D3A")
        self.canvas_heat = FigureCanvas(self.fig_heat)
        v_heat.addWidget(self.canvas_heat)

    # ----------------- Called after UI init to finalize plots -----------------
    def showEvent(self, event):
        # ensure plots are created once widgets exist
        try:
            self._setup_plots_widgets()
        except Exception:
            pass
        super().showEvent(event)

    # ----------------- Extract Button handler -----------------
    def _on_extract_clicked(self):
        url = self.url_field.text().strip()
        if not url:
            QMessageBox.warning(self, "Warning", "Please enter a URL first.")
            return

        try:
            feats = self.extractor.extract(url)
            # Fill fields (editable)
            for k, v in feats.items():
                if k in self.feature_fields:
                    self.feature_fields[k].setText(str(v))
            self.status_label.setText("STATUS: Features extracted")
            self.status_label.setStyleSheet("color: #22C55E;")
            self.speak("Features extracted")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Feature extraction failed:\n{e}")
            self.speak("Feature extraction failed")

    # ----------------- Dataset loading -----------------
    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load IoT URL Dataset", "", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            # If url column exists but features missing, compute them
            required = set(self.feature_order + ["url", "label"])
            missing = [c for c in required if c not in df.columns]
            if missing:
                # extract features for rows missing
                extractor = self.extractor
                rows = []
                for _, r in df.iterrows():
                    url = r.get("url", "")
                    feats = extractor.extract(url)
                    feats["url"] = url
                    feats["label"] = r.get("label", r.get("label", "benign"))
                    rows.append(feats)
                df = pd.DataFrame(rows)
            self.current_dataset = df
            QMessageBox.information(self, "Success", f"Dataset loaded: {len(df)} rows")
            self.speak(f"Dataset loaded with {len(df)} rows")
            # Optionally update distribution plot
            self._refresh_distribution()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{e}")

    # ----------------- Load model -----------------
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "PKL Files (*.pkl *.joblib);;All Files (*)")
        if not path:
            return
        try:
            obj = joblib.load(path)
            if isinstance(obj, dict) and "model" in obj:
                self.model = obj["model"]
                self.scaler = obj.get("scaler", None)
                self.selected_features = obj.get("features", self.feature_order)
            else:
                # assume raw model, ask for scaler
                self.model = obj
                s_path, _ = QFileDialog.getOpenFileName(self, "Select Scaler File (if any)", "", "PKL Files (*.pkl);;All Files (*)")
                if s_path:
                    self.scaler = joblib.load(s_path)
                self.selected_features = self.feature_order
            QMessageBox.information(self, "Success", "Model loaded successfully")
            self.speak("Model loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    # ----------------- Save model -----------------
    def save_model(self):
        if self.model is None or self.scaler is None:
            QMessageBox.warning(self, "Warning", "No model/scaler to save")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "PKL Files (*.pkl)")
        if not path:
            return
        try:
            payload = {
                "model": self.model,
                "scaler": self.scaler,
                "features": getattr(self, "selected_features", self.feature_order)
            }
            joblib.dump(payload, path)
            QMessageBox.information(self, "Success", "Model saved")
            self.speak("Model saved")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model:\n{e}")

    # ----------------- Training dialog -----------------
    def train_model_dialog(self):
        if self.current_dataset is None:
            QMessageBox.warning(self, "Warning", "Load dataset before training")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Train Hybrid Model (Filter + RFE + GA)")
        dialog.resize(420, 220)
        layout = QVBoxLayout(dialog)

        form = QFormLayout()
        test_edit = QLineEdit("0.2")
        n_est_edit = QLineEdit("200")
        pop_edit = QLineEdit("12")
        gen_edit = QLineEdit("10")
        mut_edit = QLineEdit("0.2")

        form.addRow("Test size (0.1-0.5):", test_edit)
        form.addRow("N estimators:", n_est_edit)
        form.addRow("GA population:", pop_edit)
        form.addRow("GA generations:", gen_edit)
        form.addRow("GA mutation rate:", mut_edit)
        layout.addLayout(form)

        progress_label = QLabel("")
        layout.addWidget(progress_label)

        def on_start():
            try:
                test_size = float(test_edit.text())
                n_est = int(n_est_edit.text())
                pop = int(pop_edit.text())
                gens = int(gen_edit.text())
                mut = float(mut_edit.text())
            except Exception:
                QMessageBox.warning(self, "Warning", "Invalid training parameters")
                return

            progress_label.setText("Training... this may take a while")
            QApplication.processEvents()
            # Run training
            try:
                selected_feats, model_obj, scaler_obj, acc = self._train_hybrid_model(
                    test_size=test_size,
                    n_estimators=n_est,
                    pop=pop,
                    gens=gens,
                    mut=mut
                )
                self.model = model_obj
                self.scaler = scaler_obj
                self.selected_features = selected_feats
                QMessageBox.information(self, "Success", f"Training done. Accuracy: {acc:.2%}\nSelected features: {selected_feats}")
                self.speak(f"Model trained with {acc:.0%} accuracy")
                dialog.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Training failed:\n{e}")
                progress_label.setText("Training failed")

        start_btn = QPushButton("🚀 Start Training")
        start_btn.clicked.connect(on_start)
        layout.addWidget(start_btn)

        dialog.exec()

    # ----------------- Hybrid training core -----------------
    def _train_hybrid_model(self, test_size=0.2, n_estimators=200, pop=12, gens=10, mut=0.2):
        # Prepare dataset
        df = self.current_dataset.copy()
        # Expect label column mapping
        if "label" not in df.columns:
            raise ValueError("Dataset must have a 'label' column (benign/suspicious)")

        df = df.reset_index(drop=True)
        df["label_num"] = df["label"].map({"benign": 0, "suspicious": 1})
        y = df["label_num"]

        # ensure features exist, otherwise compute
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = df["url"].apply(lambda u: self.extractor.extract(u).get(col, 0))

        X = df[self.feature_order].astype(float)

        # set GA params
        self.feature_selector = HybridFeatureSelectorGA(population=pop, generations=gens, mutation=mut)

        # Filter+Wrapper+GA selection
        selected = self.feature_selector.select(X, y)
        if not selected:
            selected = self.feature_order.copy()

        # Final training
        scaler = StandardScaler()
        X_sel = X[selected]
        X_scaled = scaler.fit_transform(X_sel)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Save locally to attributes
        return selected, model, scaler, acc

    # ----------------- Prediction -----------------
    def predict_url(self):
        if self.model is None or self.scaler is None:
            QMessageBox.warning(self, "Warning", "Load or train a model first")
            return

        try:
            # Read feature fields (editable)
            vals = []
            for f in self.selected_features:
                txt = self.feature_fields[f].text().strip()
                if txt == "":
                    QMessageBox.warning(self, "Warning", f"Feature '{f}' is empty. Extract or fill it.")
                    return
                vals.append(float(txt))
            X = pd.DataFrame([vals], columns=self.selected_features)
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            prob = None
            try:
                prob = self.model.predict_proba(X_scaled)[0][int(pred)]
            except Exception:
                prob = None

            label = "Benign" if int(pred) == 0 else "Suspicious"
            score_text = f"{label}"
            if prob is not None:
                score_text += f" (confidence {prob:.2f})"

            # Update result label & history
            is_attack = (int(pred) == 1)
            self.result_label.setText(f"STATUS: {'🚨 SUSPICIOUS' if is_attack else '✅ BENIGN'}  |  {score_text}")
            self.result_label.setStyleSheet("color: #F97373;" if is_attack else "color: #22C55E;")

            timestamp = datetime.datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # store history row: timestamp, url, features..., label
            feat_vals = [float(self.feature_fields[f].text()) for f in self.selected_features]
            row = {"timestamp": timestamp, "url": self.url_field.text().strip(), "pred": label}
            for k, v in zip(self.selected_features, feat_vals):
                row[k] = v
            self.history.append(row)

            self._update_stats_after_prediction()
            self._refresh_plots()
            self._append_history_text(timestamp, label)
            self.speak_result()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{e}")

    # ----------------- Update stats + history -----------------
    def _update_stats_after_prediction(self):
        if not self.history:
            return
        total = len(self.history)
        self.stat_total.setText(str(total))
        last = self.history[-1]
        self.stat_latest.setText(last["pred"])
        susp_count = sum(1 for r in self.history if r["pred"] == "Suspicious")
        rate = (susp_count / total) * 100
        self.stat_susp_rate.setText(f"{rate:.1f}%")

    def _append_history_text(self, timestamp, label):
        self.history_text.insertPlainText(f"[{timestamp}] {label} - {self.url_field.text().strip()}\n")
        self.history_text.moveCursor(QTextCursor.MoveOperation.End)

    # ----------------- Plots refresh -----------------
    def _refresh_plots(self):
        # Time series: count suspicious vs benign over time
        if not self.history:
            return
        df = pd.DataFrame(self.history)
        # convert timestamp to datetime if possible
        try:
            df["ts"] = pd.to_datetime(df["timestamp"])
        except Exception:
            df["ts"] = pd.RangeIndex(len(df))

        # time series: cumulative suspicious count
        df_sorted = df.sort_values("ts")
        df_sorted["susp"] = (df_sorted["pred"] == "Suspicious").astype(int)
        cum = df_sorted.set_index("ts")["susp"].cumsum()

        self.ax_ts.clear()
        self.ax_ts.plot(cum.index, cum.values, label="Cumulative Suspicious", linewidth=2)
        self.ax_ts.set_title("Cumulative Suspicious Detections Over Time", color="white")
        self.ax_ts.tick_params(colors="white")
        self.ax_ts.legend(facecolor="#2B2D3A")
        self.canvas_ts.draw()

        # Pie chart: distribution
        self.ax_pie.clear()
        counts = df["pred"].value_counts()
        labels = counts.index.tolist()
        values = counts.values.tolist()
        self.ax_pie.pie(values, labels=labels, autopct="%1.1f%%", textprops={"color":"white"})
        self.ax_pie.set_title("Prediction Distribution", color="white")
        self.canvas_pie.draw()

        # Feature bar: show latest feature values
        self.ax_bar.clear()
        latest = self.history[-1]
        keys = self.selected_features
        values = [latest.get(k, 0) for k in keys]
        self.ax_bar.bar(range(len(keys)), values)
        self.ax_bar.set_xticks(range(len(keys)))
        self.ax_bar.set_xticklabels(keys, rotation=45, color="white")
        self.ax_bar.set_title("Latest Feature Values", color="white")
        self.canvas_bar.draw()

        # Heatmap: correlation from last up to 100 history rows
        self.ax_heat.clear()
        if len(self.history) > 1:
            heat_df = pd.DataFrame(self.history).tail(100)
            # pick only numeric feature cols
            num_df = heat_df[[c for c in self.selected_features if c in heat_df.columns]].astype(float)
            if num_df.shape[0] > 1:
                corr = num_df.corr()
                sns.heatmap(corr, annot=True, fmt=".2f", ax=self.ax_heat, cbar_kws={"label":"corr"})
                self.ax_heat.set_title("Feature Correlation (recent)", color="white")
        self.canvas_heat.draw()

    def _refresh_distribution(self):
        # refresh distribution pie from dataset
        if self.current_dataset is None:
            return
        try:
            counts = self.current_dataset["label"].value_counts()
            self.ax_pie.clear()
            self.ax_pie.pie(counts.values, labels=counts.index, autopct="%1.1f%%", textprops={"color":"white"})
            self.ax_pie.set_title("Dataset Distribution", color="white")
            self.canvas_pie.draw()
        except Exception:
            pass

    # ----------------- Export CSV (history) -----------------
    def export_csv(self):
        if not self.history:
            QMessageBox.warning(self, "Warning", "No history to export")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export History", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.DataFrame(self.history)
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Success", "Exported history CSV")
            self.speak("History exported")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

    # ----------------- Email report (history CSV attached) -----------------
    def send_email_popup(self):
        # ⚠️ replace these with your credentials if you intend to send
        EMAIL_HOST = "smtp.gmail.com"
        EMAIL_PORT = 587
        EMAIL_HOST_USER = "rajsharma63829@gmail.com"
        EMAIL_HOST_PASSWORD = "roiu olje mpwg zbra"

        dialog = QDialog(self)
        dialog.setWindowTitle("Send Report by Email")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        to_edit = QLineEdit()
        form.addRow("To:", to_edit)
        layout.addLayout(form)

        status_lbl = QLabel("")
        layout.addWidget(status_lbl)

        def do_send():
            to_addr = to_edit.text().strip()
            if not to_addr:
                QMessageBox.warning(self, "Warning", "Enter recipient email")
                return
            # create temp csv
            try:
                tmp = "report_history.csv"
                pd.DataFrame(self.history).to_csv(tmp, index=False)
                msg = EmailMessage()
                msg["From"] = EMAIL_HOST_USER
                msg["To"] = to_addr
                msg["Subject"] = "IoT URL Detection Report"
                msg.set_content("Attached is the URL detection history report.")
                with open(tmp, "rb") as f:
                    msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=tmp)
                context = ssl.create_default_context()
                with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
                    server.starttls(context=context)
                    server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
                    server.send_message(msg)
                status_lbl.setText("Email sent")
                QMessageBox.information(self, "Success", "Email sent")
                dialog.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Email failed:\n{e}")

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(do_send)
        layout.addWidget(send_btn)
        dialog.exec()
# =============================================================
# Part 4/5 – Voice Control, TTS, Animations, Utility Functions
# =============================================================

    # ----------------- Text-to-Speech -----------------
    def speak(self, text: str):
        worker = TTSWorker(self)
        worker.text = text
        worker.start()   # runs in background thread

    def speak_result(self):
        txt = self.result_label.text()
        if txt and txt != "Ready":
            self.speak(txt)
        else:
            self.speak("No prediction result available")

    # ----------------- Voice Command Mode -----------------
    def start_voice_mode(self):
        if hasattr(self, "voice_thread") and self.voice_thread and self.voice_thread.isRunning():
            # Turn off
            self.voice_thread.stop()
            self.voice_thread = None
            self.status_label.setText("STATUS: Idle")
            self.status_label.setStyleSheet("color:#38BDF8;")
            self.speak("Voice mode deactivated")
            return

        # Activate voice
        self.status_label.setText("STATUS: Voice Mode Active")
        self.status_label.setStyleSheet("color:#FACC15;")
        self.speak("Voice command mode activated. Say a command.")

        self.voice_thread = VoiceWorker(self)
        self.voice_thread.commandDetected.connect(self._handle_voice_command)
        self.voice_thread.errorOccurred.connect(self._handle_voice_error)
        self.voice_thread.start()

    def _handle_voice_command(self, command: str):
        cmd = command.lower()

        if "predict" in cmd:
            self.speak("Predicting URL status")
            self.predict_url()

        elif "extract" in cmd:
            self.speak("Extracting features")
            self._on_extract_clicked()

        elif "speak" in cmd:
            self.speak_result()

        elif "load model" in cmd:
            self.speak("Open model selection window")
            self.load_model()

        elif "train" in cmd:
            self.speak("Opening training dialog")
            self.train_model_dialog()

        elif "email" in cmd:
            self.speak("Opening email report")
            self.send_email_popup()

        elif "exit" in cmd or "quit" in cmd:
            self.speak("Exiting voice mode")
            if self.voice_thread:
                self.voice_thread.stop()
                self.voice_thread = None
            self.status_label.setText("STATUS: Idle")
            self.status_label.setStyleSheet("color:#38BDF8;")

        else:
            self.speak("I did not understand that command")

    def _handle_voice_error(self, msg: str):
        print("Voice error:", msg)
        self.speak(msg)

    # ----------------- Label Fade Animation -----------------
    def _animate_result_label(self):
        effect = QGraphicsOpacityEffect(self.result_label)
        self.result_label.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(600)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    # After prediction, animate the label
    def update_result_with_animation(self, text, color_good=True):
        self.result_label.setText(text)
        self.result_label.setStyleSheet(
            "color:#22C55E; font-weight:700;" if color_good else "color:#F97373; font-weight:700;"
        )
        self._animate_result_label()

    # ----------------- Utility -----------------
    def closeEvent(self, event):
        # stop voice worker safely
        if hasattr(self, "voice_thread") and self.voice_thread and self.voice_thread.isRunning():
            self.voice_thread.stop()
        event.accept()
        self.speak("Closing application")

# =============================================================
# Part 5/5 – Final fixes, corrected predict method, video tab, app launcher
# =============================================================

    # ----------------- Corrected Prediction (replaces earlier definition) -----------------
    def predict_url(self):
        """
        Corrected and robust predict method.
        Uses self.selected_features (falls back to full feature_order).
        """
        if self.model is None or self.scaler is None:
            QMessageBox.warning(self, "Warning", "Load or train a model first")
            return

        try:
            # ensure selected_features exists
            selected = getattr(self, "selected_features", None)
            if not selected:
                selected = self.feature_order.copy()
                self.selected_features = selected

            # Read feature fields (editable) but only for selected features
            vals = []
            for f in selected:
                widget = self.feature_fields.get(f)
                if widget is None:
                    QMessageBox.warning(self, "Warning", f"Feature widget for '{f}' not found.")
                    return
                txt = widget.text().strip()
                if txt == "":
                    QMessageBox.warning(self, "Warning", f"Feature '{f}' is empty. Extract or fill it.")
                    return
                vals.append(float(txt))

            X = pd.DataFrame([vals], columns=selected)
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]

            prob = None
            try:
                if hasattr(self.model, "predict_proba"):
                    prob = self.model.predict_proba(X_scaled)[0][int(pred)]
            except Exception:
                prob = None

            label = "Benign" if int(pred) == 0 else "Suspicious"
            score_text = f"{label}"
            if prob is not None:
                score_text += f" (confidence {prob:.2f})"

            # Update result label & history
            is_attack = (int(pred) == 1)
            color_good = not is_attack
            self.update_result_with_animation(f"STATUS: {'🚨 SUSPICIOUS' if is_attack else '✅ BENIGN'}  |  {score_text}", color_good=color_good)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            feat_vals = [float(v) for v in vals]
            row = {"timestamp": timestamp, "url": self.url_field.text().strip(), "pred": label}
            for k, v in zip(selected, feat_vals):
                row[k] = v
            self.history.append(row)

            self._update_stats_after_prediction()
            self._refresh_plots()
            self._append_history_text(timestamp, label)
            self.speak_result()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{e}")

    # ----------------- Video tab setup (optional) -----------------
    def _setup_video_tab(self):
        if not HAS_VIDEO:
            return
        try:
            self.tab_video = QWidget()
            self.tabs.addTab(self.tab_video, "🎬 Video")
            layout = QVBoxLayout(self.tab_video)
            layout.setContentsMargins(10, 10, 10, 10)

            info = QLabel("Overview Video (optional)")
            info.setObjectName("SectionTitleBig")
            layout.addWidget(info)

            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumHeight(280)
            layout.addWidget(self.video_widget, 1)

            controls = QHBoxLayout()
            self.btn_play_video = QPushButton("▶ Play")
            self.btn_stop_video = QPushButton("⏹ Stop")
            controls.addWidget(self.btn_play_video)
            controls.addWidget(self.btn_stop_video)
            controls.addStretch(1)
            layout.addLayout(controls)

            self.media_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_widget)

            # NOTE: set your local video file path here if you want it to play
            video_path = r"C:\Users\rajsh\OneDrive\Desktop\IOT SUSPICIOUS URL DETECTION\assets\An Efficient Hybrid Feature Selection Technique towards Prediction of Suspicious URLs in IoT Environment.mp4"
            if os.path.exists(video_path):
                self.media_player.setSource(QUrl.fromLocalFile(video_path))

            self.btn_play_video.clicked.connect(lambda: self.media_player.play())
            self.btn_stop_video.clicked.connect(lambda: self.media_player.stop())
        except Exception:
            # Non-fatal: continue without video
            pass

# ----------------- End of SuspiciousURLDashboard class -----------------

# =============================================================
# Application Launcher
# =============================================================
def main():
    app = QApplication(sys.argv)

    # Nice DPI scaling on some platforms
    try:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    except Exception:
        pass

    window = SuspiciousURLDashboard()

    # ensure default selected_features exists to avoid errors
    if not hasattr(window, "selected_features") or not window.selected_features:
        window.selected_features = window.feature_order.copy()

    # show window
    window.show()

    # If video support is available, setup tab
    try:
        if HAS_VIDEO:
            window._setup_video_tab()
    except Exception:
        pass

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
