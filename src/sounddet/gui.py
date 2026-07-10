from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from . import CLASS_NAMES
from .config import load_config, resolve_app_path
from .engine import DetectionEngine
from .event_store import EventStore
from .model_registry import registry
from .reporting import export_session_report
from .system_monitor import read_runtime_metrics


class EvalWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        model_key: str,
        quick: bool = True,
        class_thresholds: tuple[float, float, float] | None = None,
        ema_alpha: float = 0.6,
        confirm_frames: int = 1,
        merge_gap_sec: float = 1.0,
        trial_mode: str = "stream",
        input_mode: str = "online_replay",
        device: int | None = None,
        channels: int = 1,
        duration_sec: float = 20.0,
    ):
        super().__init__()
        self.model_key = model_key
        self.quick = quick
        self.class_thresholds = class_thresholds
        self.ema_alpha = ema_alpha
        self.confirm_frames = confirm_frames
        self.merge_gap_sec = merge_gap_sec
        self.trial_mode = trial_mode
        self.input_mode = input_mode
        self.device = device
        self.channels = channels
        self.duration_sec = duration_sec

    def run(self):
        try:
            cfg = load_config()
            cfg.class_thresholds = self.class_thresholds
            cfg.ema_alpha = self.ema_alpha
            cfg.confirm_frames = self.confirm_frames
            cfg.merge_gap_sec = self.merge_gap_sec
            cfg.trial_mode = self.trial_mode
            store = EventStore(cfg)
            engine = DetectionEngine(cfg, model_key=self.model_key, store=store)
            out_root = resolve_app_path(cfg, "outputs") / "gui_sessions"
            if self.input_mode == "microphone":
                summary = engine.run_microphone(
                    duration_sec=self.duration_sec,
                    device=self.device,
                    channels=self.channels,
                    out_dir=out_root,
                )
            else:
                summary = engine.run_streaming_trial(
                    minutes=1.0 if self.quick else cfg.stream_minutes,
                    streams_per_dataset=1 if self.quick else cfg.streams_per_dataset,
                    trial_mode=cfg.trial_mode,
                    out_dir=out_root,
                )
            session_dirs = [p for p in out_root.glob("*") if p.is_dir()]
            latest = max(session_dirs, key=lambda p: p.stat().st_mtime) if session_dirs else out_root
            self.finished.emit(summary, str(latest))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    _fallback_app = None

    def __new__(cls, *args, **kwargs):
        if QtWidgets.QApplication.instance() is None:
            cls._fallback_app = QtWidgets.QApplication(sys.argv)
        return super().__new__(cls)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sound Target Detection")
        self.resize(1480, 900)
        self.models = registry()
        self.cfg = load_config()
        self.store = EventStore(self.cfg)
        self.audio_devices: list[dict] = []
        self.obs_labels: dict[str, QtWidgets.QLabel] = {}
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        self.setStyleSheet(self._style())
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Acoustic Sentinel")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel("online sound target detection console")
        subtitle.setObjectName("Subtitle")
        header_text = QtWidgets.QVBoxLayout()
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header.addLayout(header_text)
        header.addStretch(1)
        layout.addLayout(header)
        status_cards = QtWidgets.QHBoxLayout()
        self.card_model = QtWidgets.QLabel("Model: -")
        self.card_events = QtWidgets.QLabel("Events: 0")
        self.card_latency = QtWidgets.QLabel("Latency p95: -")
        self.card_direction = QtWidgets.QLabel("Direction: -")
        self.card_db = QtWidgets.QLabel(f"DB: {self.store.db_path}")
        for card in [self.card_model, self.card_events, self.card_latency, self.card_direction, self.card_db]:
            card.setObjectName("StatusCard")
            card.setMinimumHeight(54)
            card.setWordWrap(True)
            status_cards.addWidget(card)
        layout.addLayout(status_cards)
        layout.addWidget(self._make_observability_panel())
        controls_panel = QtWidgets.QVBoxLayout()
        controls_top = QtWidgets.QHBoxLayout()
        controls_bottom = QtWidgets.QHBoxLayout()
        self.input_combo = QtWidgets.QComboBox()
        self.input_combo.addItem("Online replay trial", "online_replay")
        self.input_combo.addItem("Live microphone", "microphone")
        self.device_combo = QtWidgets.QComboBox()
        self.refresh_devices_btn = QtWidgets.QPushButton("Refresh")
        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setRange(1, 8)
        self.channel_spin.setValue(1)
        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(2.0, 3600.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setValue(20.0)
        self.duration_spin.setSuffix(" s")
        self.model_combo = QtWidgets.QComboBox()
        for key, spec in self.models.items():
            suffix = "" if spec.available else " (unavailable)"
            self.model_combo.addItem(spec.label + suffix, key)
        self.quick_box = QtWidgets.QCheckBox("Quick online replay")
        self.quick_box.setChecked(True)
        self.trial_combo = QtWidgets.QComboBox()
        self.trial_combo.addItem("Stream", "stream")
        self.trial_combo.addItem("Aligned", "aligned")
        self.gun_spin = self._threshold_box(0.25)
        self.glass_spin = self._threshold_box(0.25)
        self.baby_spin = self._threshold_box(0.35)
        self.ema_spin = self._threshold_box(0.60)
        self.merge_spin = self._threshold_box(1.00)
        self.confirm_spin = QtWidgets.QSpinBox()
        self.confirm_spin.setRange(1, 8)
        self.confirm_spin.setValue(1)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.report_btn = QtWidgets.QPushButton("Export latest report")
        self.devices_btn = QtWidgets.QPushButton("Audio devices")
        self.stop_btn.setEnabled(False)
        controls_top.addWidget(QtWidgets.QLabel("Input"))
        controls_top.addWidget(self.input_combo, 1)
        controls_top.addWidget(QtWidgets.QLabel("Mic"))
        controls_top.addWidget(self.device_combo, 2)
        controls_top.addWidget(self.refresh_devices_btn)
        controls_top.addWidget(QtWidgets.QLabel("Ch"))
        controls_top.addWidget(self.channel_spin)
        controls_top.addWidget(QtWidgets.QLabel("Duration"))
        controls_top.addWidget(self.duration_spin)
        controls_top.addWidget(QtWidgets.QLabel("Model"))
        controls_top.addWidget(self.model_combo, 2)
        controls_top.addWidget(self.start_btn)
        controls_top.addWidget(self.stop_btn)
        controls_bottom.addWidget(QtWidgets.QLabel("Trial"))
        controls_bottom.addWidget(self.trial_combo)
        controls_bottom.addWidget(self.quick_box)
        controls_bottom.addWidget(QtWidgets.QLabel("Gun"))
        controls_bottom.addWidget(self.gun_spin)
        controls_bottom.addWidget(QtWidgets.QLabel("Glass"))
        controls_bottom.addWidget(self.glass_spin)
        controls_bottom.addWidget(QtWidgets.QLabel("Baby"))
        controls_bottom.addWidget(self.baby_spin)
        controls_bottom.addWidget(QtWidgets.QLabel("EMA"))
        controls_bottom.addWidget(self.ema_spin)
        controls_bottom.addWidget(QtWidgets.QLabel("Merge"))
        controls_bottom.addWidget(self.merge_spin)
        controls_bottom.addWidget(QtWidgets.QLabel("Confirm frames"))
        controls_bottom.addWidget(self.confirm_spin)
        controls_bottom.addStretch(1)
        controls_bottom.addWidget(self.report_btn)
        controls_bottom.addWidget(self.devices_btn)
        controls_panel.addLayout(controls_top)
        controls_panel.addLayout(controls_bottom)
        layout.addLayout(controls_panel)

        plots = QtWidgets.QHBoxLayout()
        self.wave_plot = pg.PlotWidget(title="Online event timeline")
        self.wave_plot.setMinimumHeight(300)
        self.wave_plot.setLabel("bottom", "Time / event index")
        self.conf_plot = pg.PlotWidget(title="Latest summary")
        self.conf_plot.setMinimumHeight(300)
        self.conf_plot.setLabel("bottom", "Metric")
        self.direction_plot = pg.PlotWidget(title="Source direction")
        self.direction_plot.setMinimumHeight(300)
        self.direction_plot.setXRange(-1.1, 1.1)
        self.direction_plot.setYRange(-0.1, 1.1)
        self.direction_plot.hideAxis("bottom")
        self.direction_plot.hideAxis("left")
        plots.addWidget(self.wave_plot, 2)
        plots.addWidget(self.conf_plot, 1)
        plots.addWidget(self.direction_plot, 1)
        layout.addLayout(plots, 2)

        tables = QtWidgets.QHBoxLayout()
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setMinimumHeight(210)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Output", "Model"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.event_table = QtWidgets.QTableWidget(0, 6)
        self.event_table.setMinimumHeight(210)
        self.event_table.setHorizontalHeaderLabels(["ID", "Start", "End", "Class", "Confidence", "Review"])
        self.event_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        tables.addWidget(self.table, 1)
        tables.addWidget(self.event_table, 1)
        layout.addLayout(tables, 1)
        review_controls = QtWidgets.QHBoxLayout()
        self.tp_btn = QtWidgets.QPushButton("Mark TP")
        self.fp_btn = QtWidgets.QPushButton("Mark FP")
        self.fn_btn = QtWidgets.QPushButton("Mark FN")
        self.note_edit = QtWidgets.QLineEdit()
        self.note_edit.setPlaceholderText("review note")
        review_controls.addWidget(self.tp_btn)
        review_controls.addWidget(self.fp_btn)
        review_controls.addWidget(self.fn_btn)
        review_controls.addWidget(self.note_edit)
        layout.addLayout(review_controls)
        self.status = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status)
        self.setCentralWidget(root)
        self.start_btn.clicked.connect(self.start_eval)
        self.stop_btn.clicked.connect(self.stop_eval)
        self.report_btn.clicked.connect(self.export_report)
        self.devices_btn.clicked.connect(self.show_devices)
        self.refresh_devices_btn.clicked.connect(self.refresh_devices)
        self.tp_btn.clicked.connect(lambda: self.review_selected("TP"))
        self.fp_btn.clicked.connect(lambda: self.review_selected("FP"))
        self.fn_btn.clicked.connect(lambda: self.review_selected("FN"))
        self.thread = None
        self.worker = None
        self.latest_out_dir = ""
        self.refresh_devices()
        self._draw_direction(None)
        self.obs_timer = QtCore.QTimer(self)
        self.obs_timer.timeout.connect(self._refresh_observability)
        self.obs_timer.start(2000)
        self._refresh_observability()

    def _style(self) -> str:
        return """
        QMainWindow, QWidget { background: #10151b; color: #eef4f7; font-family: "Segoe UI"; font-size: 10pt; }
        QLabel#Title { font-size: 24pt; font-weight: 700; color: #f6fbff; letter-spacing: 0px; }
        QLabel#Subtitle { color: #94a8b8; font-size: 10pt; }
        QLabel#StatusCard { background: #17212b; border: 1px solid #263645; border-radius: 8px; padding: 10px; color: #e7f0f5; }
        QTableWidget { background: #151f28; alternate-background-color: #1a2630; color: #e7edf4; gridline-color: #2c3a46; border: 1px solid #263645; border-radius: 6px; }
        QHeaderView::section { background: #20303b; color: #cbd7df; padding: 6px; border: 0px; }
        QGroupBox { border: 1px solid #263645; border-radius: 8px; margin-top: 10px; padding: 10px; color: #dce8ef; font-weight: 700; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #9fb6c8; }
        QPushButton { background: #1f6f78; color: white; padding: 7px 12px; border: 0px; border-radius: 6px; font-weight: 600; }
        QPushButton:hover { background: #268895; }
        QPushButton:disabled { background: #303841; color: #8794a0; }
        QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { background: #17212b; color: #eef4f7; border: 1px solid #344556; border-radius: 5px; padding: 5px; }
        QCheckBox { color: #d8e3ea; spacing: 6px; }
        """

    def _threshold_box(self, value: float) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(0.00, 1.00)
        box.setSingleStep(0.05)
        box.setDecimals(2)
        box.setValue(value)
        box.setMaximumWidth(72)
        return box

    def _make_observability_panel(self) -> QtWidgets.QGroupBox:
        panel = QtWidgets.QGroupBox("Runtime observability")
        grid = QtWidgets.QGridLayout(panel)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        items = [
            ("run", "Run state"),
            ("cpu", "CPU"),
            ("memory", "Memory"),
            ("gpu", "GPU"),
            ("gpu_mem", "GPU memory"),
            ("gpu_temp", "GPU temp"),
            ("disk", "Disk free"),
            ("windows", "Windows"),
            ("events", "Events"),
            ("p95", "Latency P95"),
            ("p99", "Latency P99"),
            ("queue", "Queue / dropped"),
        ]
        for i, (key, title) in enumerate(items):
            cell = QtWidgets.QWidget()
            cell_layout = QtWidgets.QVBoxLayout(cell)
            cell_layout.setContentsMargins(10, 8, 10, 8)
            cell.setObjectName("StatusCard")
            label = QtWidgets.QLabel(title)
            label.setStyleSheet("color: #91a6b6; font-size: 8.5pt;")
            value = QtWidgets.QLabel("-")
            value.setStyleSheet("color: #f5fbff; font-size: 12pt; font-weight: 700;")
            value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            cell_layout.addWidget(label)
            cell_layout.addWidget(value)
            self.obs_labels[key] = value
            grid.addWidget(cell, i // 6, i % 6)
        return panel

    def _set_obs(self, key: str, value: str) -> None:
        label = self.obs_labels.get(key)
        if label is not None:
            label.setText(value)

    def _refresh_observability(self) -> None:
        try:
            runtime = read_runtime_metrics(self.cfg.app_root)
            running = self.thread is not None and self.thread.isRunning()
            event_rows = self.store.query("SELECT COUNT(*) AS n FROM events")
            window_rows = self.store.query("SELECT COUNT(*) AS n FROM window_predictions")
            lat_rows = self.store.query("SELECT latency_ms FROM window_predictions ORDER BY id DESC LIMIT 2000")
            latencies = np.array([float(row["latency_ms"]) for row in lat_rows], dtype=float)
            p95 = float(np.percentile(latencies, 95)) if latencies.size else 0.0
            p99 = float(np.percentile(latencies, 99)) if latencies.size else 0.0
            events = int(event_rows[0]["n"]) if event_rows else 0
            windows = int(window_rows[0]["n"]) if window_rows else 0
            self._set_obs("run", "Running" if running else "Idle")
            self._set_obs("cpu", f"{runtime.cpu_percent:.0f}%")
            self._set_obs("memory", f"{runtime.memory_percent:.0f}%")
            if runtime.gpu_util_percent is None:
                self._set_obs("gpu", "not available")
                self._set_obs("gpu_mem", "-")
                self._set_obs("gpu_temp", "-")
            else:
                self._set_obs("gpu", f"{runtime.gpu_util_percent:.0f}%")
                self._set_obs("gpu_mem", f"{runtime.gpu_memory_used_mb:.0f}/{runtime.gpu_memory_total_mb:.0f} MB")
                self._set_obs("gpu_temp", f"{runtime.gpu_temperature_c:.0f} C")
            self._set_obs("disk", f"{runtime.disk_free_gb:.1f} GB")
            self._set_obs("windows", str(windows))
            self._set_obs("events", str(events))
            self._set_obs("p95", f"{p95:.2f} ms")
            self._set_obs("p99", f"{p99:.2f} ms")
            self._set_obs("queue", "0 / 0")
        except Exception as exc:
            self._set_obs("run", f"metrics error: {exc}")

    def start_eval(self):
        key = self.model_combo.currentData()
        spec = self.models[key]
        if not spec.available:
            QtWidgets.QMessageBox.warning(self, "Model unavailable", f"Checkpoint not found:\n{spec.checkpoint}")
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        mode = "live microphone session" if self.input_combo.currentData() == "microphone" else "online replay evaluation"
        self.status.setText(f"Running {mode}...")
        self.thread = QtCore.QThread()
        device = self.device_combo.currentData()
        if device == "none":
            device = None
        self.worker = EvalWorker(
            key,
            quick=self.quick_box.isChecked(),
            class_thresholds=(float(self.gun_spin.value()), float(self.glass_spin.value()), float(self.baby_spin.value())),
            ema_alpha=float(self.ema_spin.value()),
            confirm_frames=int(self.confirm_spin.value()),
            merge_gap_sec=float(self.merge_spin.value()),
            trial_mode=str(self.trial_combo.currentData()),
            input_mode=str(self.input_combo.currentData()),
            device=device,
            channels=int(self.channel_spin.value()),
            duration_sec=float(self.duration_spin.value()),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def stop_eval(self):
        self.status.setText("Stop requested; current batch will finish.")
        self.stop_btn.setEnabled(False)

    def on_finished(self, summary: dict, out_dir: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText(f"Complete. Output: {out_dir}")
        self.latest_out_dir = out_dir
        self._show_summary(summary, out_dir)

    def on_failed(self, message: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("Failed")
        QtWidgets.QMessageBox.critical(self, "Evaluation failed", message)

    def _show_summary(self, summary: dict, out_dir: str):
        self.table.setRowCount(0)
        model = self.model_combo.currentText()
        self.card_model.setText(f"Model: {model}")
        self.card_events.setText(f"Events: {summary.get('n_pred_events', 0)}")
        self.card_latency.setText(f"Latency p95: {summary.get('latency_p95_ms', 0):.3f} ms" if "latency_p95_ms" in summary else "Latency p95: -")
        self._refresh_observability()
        if "direction_azimuth_median_deg" in summary:
            self.card_direction.setText(f"Direction: {summary['direction_azimuth_median_deg']:.1f} deg")
        else:
            self.card_direction.setText("Direction: -")
        for k, v in summary.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(k))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{v:.6f}" if isinstance(v, float) else str(v)))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(out_dir))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(model))
        self.conf_plot.clear()
        keys = ["window_accuracy", "window_macro_f1", "event_precision", "event_recall", "event_f1"]
        vals = [float(summary.get(k, 0) or 0) for k in keys]
        bg = pg.BarGraphItem(x=np.arange(len(vals)), height=vals, width=0.6, brush="#2f7f7b")
        self.conf_plot.addItem(bg)
        self.conf_plot.getAxis("bottom").setTicks([[(i, k.replace("_", "\n")) for i, k in enumerate(keys)]])
        self.wave_plot.clear()
        pred_path = Path(out_dir) / "events_pred.csv"
        gt_path = Path(out_dir) / "events_gt.csv"
        if pred_path.exists() and gt_path.exists():
            pred = pd.read_csv(pred_path)
            gt = pd.read_csv(gt_path)
            for label in range(len(CLASS_NAMES)):
                gp = gt[gt["label"] == label]
                pp = pred[pred["label"] == label]
                self.wave_plot.plot(gp["start"].to_numpy(), np.full(len(gp), label - 0.12), pen=None, symbol="o", symbolBrush="#3a9d78", symbolSize=7)
                self.wave_plot.plot(pp["start"].to_numpy(), np.full(len(pp), label + 0.12), pen=None, symbol="t", symbolBrush="#d65f5f", symbolSize=8)
            self.wave_plot.getAxis("left").setTicks([[(i, n) for i, n in enumerate(CLASS_NAMES)]])
            self._show_events(pred)
        direction_path = Path(out_dir) / "direction_estimates.csv"
        if direction_path.exists():
            ddf = pd.read_csv(direction_path)
            if not ddf.empty:
                self._draw_direction(float(ddf["azimuth_deg"].median()), float(ddf["confidence"].mean()))
        else:
            self._draw_direction(None)

    def _draw_direction(self, azimuth: float | None, confidence: float = 0.0):
        self.direction_plot.clear()
        theta = np.linspace(0, np.pi, 120)
        self.direction_plot.plot(np.cos(theta), np.sin(theta), pen=pg.mkPen("#365160", width=2))
        for deg in [-90, -45, 0, 45, 90]:
            rad = np.deg2rad(90 - deg)
            self.direction_plot.plot([0, np.cos(rad)], [0, np.sin(rad)], pen=pg.mkPen("#22323d", width=1))
        left_label = pg.TextItem("L", color="#91a6b6", anchor=(0.5, 0.5))
        center_label = pg.TextItem("C", color="#91a6b6", anchor=(0.5, 0.5))
        right_label = pg.TextItem("R", color="#91a6b6", anchor=(0.5, 0.5))
        self.direction_plot.addItem(left_label)
        self.direction_plot.addItem(center_label)
        self.direction_plot.addItem(right_label)
        left_label.setPos(-1.02, 0.02)
        center_label.setPos(0.0, 1.05)
        right_label.setPos(1.02, 0.02)
        if azimuth is not None:
            rad = np.deg2rad(90 - azimuth)
            length = 0.35 + 0.55 * max(0.0, min(1.0, confidence))
            x, y = length * np.cos(rad), length * np.sin(rad)
            self.direction_plot.plot([0, x], [0, y], pen=pg.mkPen("#f0b45a", width=5))
            dot = pg.ScatterPlotItem([x], [y], brush="#f0b45a", pen="#fff4d6", size=13)
            self.direction_plot.addItem(dot)

    def _show_events(self, pred: pd.DataFrame):
        self.event_table.setRowCount(0)
        if pred.empty:
            return
        for _, ev in pred.head(200).iterrows():
            row = self.event_table.rowCount()
            self.event_table.insertRow(row)
            self.event_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(ev.get("id", row))))
            self.event_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{float(ev['start']):.2f}"))
            self.event_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{float(ev['end']):.2f}"))
            self.event_table.setItem(row, 3, QtWidgets.QTableWidgetItem(CLASS_NAMES[int(ev["label"])]))
            self.event_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{float(ev['confidence']):.3f}"))
            self.event_table.setItem(row, 5, QtWidgets.QTableWidgetItem(str(ev.get("review_status", "unreviewed"))))

    def selected_event_id(self) -> int | None:
        row = self.event_table.currentRow()
        if row < 0:
            return None
        try:
            return int(self.event_table.item(row, 0).text())
        except Exception:
            return None

    def review_selected(self, status: str):
        event_id = self.selected_event_id()
        if event_id is None:
            QtWidgets.QMessageBox.information(self, "No event selected", "Select an event row first.")
            return
        self.store.review_event(event_id, status, self.note_edit.text())
        self.status.setText(f"Event {event_id} marked {status}")

    def export_report(self):
        sessions = self.store.query("SELECT session_id FROM sessions ORDER BY started_at DESC LIMIT 1")
        if not sessions:
            QtWidgets.QMessageBox.information(self, "No session", "No database session is available yet.")
            return
        package = export_session_report(self.store, sessions[0]["session_id"], resolve_app_path(self.cfg, self.cfg.report_dir) / sessions[0]["session_id"])
        self.status.setText(f"Report exported: {package}")

    def show_devices(self):
        from .audio_stream import list_audio_devices

        devices = list_audio_devices()
        text = "\n".join(str(d) for d in devices) if devices else "No microphone backend available. Install sounddevice for live microphone input."
        QtWidgets.QMessageBox.information(self, "Audio devices", text[:4000])

    def refresh_devices(self):
        from .audio_stream import list_audio_devices

        self.audio_devices = [d for d in list_audio_devices() if int(d.get("max_input_channels", 0) or 0) > 0]
        self.device_combo.clear()
        self.device_combo.addItem("Default input", "none")
        for d in self.audio_devices:
            name = str(d.get("name", f"Device {d.get('index')}"))
            ch = int(d.get("max_input_channels", 0) or 0)
            self.device_combo.addItem(f"{d.get('index')}: {name} ({ch} ch)", int(d["index"]))


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = MainWindow()
    win.show()
    return app.exec_()
