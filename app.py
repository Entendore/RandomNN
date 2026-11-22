"""
Neural Network Visualizer — Loss, PCA, Derivatives, Integrals
--------------------------------------------------------------
Randomizes weights of a feedforward NN over n iterations and visualizes:
  - MSE loss per iteration
  - PCA projections (1D/2D/3D) of the flattened weight vectors
  - 1st through 4th derivatives of the network output w.r.t. input
  - 1st through 4th cumulative integrals of the network output

Uses tanh activations so derivatives are smooth and meaningful.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QPushButton, QDoubleSpinBox, QGroupBox,
    QFormLayout, QComboBox, QProgressBar, QTextEdit, QTabWidget
)


# ---------------------------------------------------------------------------
# 1) Neural Network (smooth tanh activations so derivatives exist)
# ---------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.initialize_weights()

    def initialize_weights(self):
        self.weights, self.biases = [], []
        for i in range(len(self.layer_sizes) - 1):
            fan_in, fan_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            std = np.sqrt(2.0 / fan_in)
            self.weights.append(np.random.randn(fan_in, fan_out) * std)
            self.biases.append(np.zeros((1, fan_out)))

    def randomize_weights(self):
        self.weights, self.biases = [], []
        for i in range(len(self.layer_sizes) - 1):
            fan_in, fan_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            self.weights.append(np.random.uniform(-1.5, 1.5, (fan_in, fan_out)))
            self.biases.append(np.random.uniform(-1.5, 1.5, (1, fan_out)))

    def flatten_params(self):
        """Flatten all weights and biases into a single 1D vector."""
        return np.concatenate(
            [w.ravel() for w in self.weights] +
            [b.ravel() for b in self.biases]
        )

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    def forward(self, X):
        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            a = self.tanh(z) if i < len(self.weights) - 1 else z
        return a

    def compute_loss(self, y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))


# ---------------------------------------------------------------------------
# 2) Dataset
# ---------------------------------------------------------------------------
def make_dataset(n_samples=200, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n_samples, 1))
    y = np.sin(X) + rng.normal(0, noise, size=X.shape)
    return X, y


# ---------------------------------------------------------------------------
# 3) Numerical calculus helpers
# ---------------------------------------------------------------------------
def numerical_derivative(y, dx, order=1):
    """Compute the n-th order derivative using central finite differences."""
    out = y.copy()
    for _ in range(order):
        d = np.zeros_like(out)
        d[1:-1] = (out[2:] - out[:-2]) / (2 * dx)
        d[0] = (out[1] - out[0]) / dx
        d[-1] = (out[-1] - out[-2]) / dx
        out = d
    return out


def cumulative_integral(y, dx, order=1):
    """Repeated cumulative trapezoidal integration."""
    out = y.copy()
    for _ in range(order):
        # Trapezoidal rule cumulative
        avg = (out[:-1] + out[1:]) / 2.0
        cum = np.concatenate([[0.0], np.cumsum(avg) * dx])
        # Re-center so integral starts at 0
        cum -= cum[0]
        out = cum
    return out


def pca_project(X, n_components):
    """PCA projection (manual, no sklearn). Returns projected data and eigenvalues."""
    X_c = X - X.mean(axis=0)
    cov = np.cov(X_c, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    proj = X_c @ eigvecs[:, :n_components]
    return proj, eigvals[:n_components]


# ---------------------------------------------------------------------------
# 4) Worker thread
# ---------------------------------------------------------------------------
class Worker(QObject):
    iteration_done = Signal(int, float, object)   # (i, loss, flattened_weights)
    finished = Signal()

    def __init__(self, network, X, y, x_dense, n_iterations):
        super().__init__()
        self.network = network
        self.X, self.y = X, y
        self.x_dense = x_dense
        self.n_iterations = n_iterations
        self._stop = False

    def run(self):
        for i in range(self.n_iterations):
            if self._stop:
                break
            self.network.randomize_weights()
            y_pred = self.network.forward(self.X)
            loss = self.network.compute_loss(y_pred, self.y)
            # Also evaluate on dense grid for derivatives/integrals
            y_dense = self.network.forward(self.x_dense)
            flat = self.network.flatten_params()
            self.iteration_done.emit(i, loss, (flat, y_dense))
        self.finished.emit()

    def stop(self):
        self._stop = True


# ---------------------------------------------------------------------------
# 5) Matplotlib canvases
# ---------------------------------------------------------------------------
class LossCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Iteration"); self.ax.set_ylabel("MSE Loss")
        self.ax.set_title("Loss vs. Iteration (random weights)")
        self.ax.grid(True, alpha=0.3)
        (self.line,) = self.ax.plot([], [], "-", color="#1f77b4")
        self.fig.tight_layout()

    def update(self, iters, losses):
        self.line.set_data(iters, losses)
        self.ax.relim(); self.ax.autoscale_view(); self.draw_idle()

    def clear(self):
        self.line.set_data([], []); self.ax.relim(); self.ax.autoscale_view(); self.draw_idle()


class PCACanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

    def update(self, proj, n_components, eigvals):
        self.fig.clear()
        if n_components == 1:
            ax = self.fig.add_subplot(111)
            ax.hist(proj[:, 0], bins=40, color="#ff7f0e", alpha=0.8, edgecolor="white")
            ax.set_xlabel(f"PC1 (λ={eigvals[0]:.2f})")
            ax.set_ylabel("Count")
            ax.set_title("PCA 1D — Histogram of PC1")
        elif n_components == 2:
            ax = self.fig.add_subplot(111)
            ax.scatter(proj[:, 0], proj[:, 1], s=12, alpha=0.6, c="#2ca02c")
            ax.set_xlabel(f"PC1 (λ={eigvals[0]:.2f})")
            ax.set_ylabel(f"PC2 (λ={eigvals[1]:.2f})")
            ax.set_title("PCA 2D — Weight space projection")
            ax.grid(True, alpha=0.3)
        else:  # 3
            ax = self.fig.add_subplot(111, projection="3d")
            ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                       s=10, alpha=0.6, c="#d62728")
            ax.set_xlabel(f"PC1 (λ={eigvals[0]:.2f})")
            ax.set_ylabel(f"PC2 (λ={eigvals[1]:.2f})")
            ax.set_zlabel(f"PC3 (λ={eigvals[2]:.2f})")
            ax.set_title("PCA 3D — Weight space projection")
        self.fig.tight_layout()
        self.draw_idle()

    def clear(self):
        self.fig.clear(); self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(); self.draw_idle()


class FunctionCanvas(FigureCanvas):
    """Shared canvas for derivatives and integrals — plots f and orders 1..4."""
    def __init__(self, title, ylabel):
        self.fig = Figure(figsize=(6, 5))
        super().__init__(self.fig)
        self.title = title
        self.ylabel = ylabel
        self.fig.tight_layout()

    def update(self, x, curves, labels, colors):
        """curves: list of 5 arrays [f, f', f'', f''', f''''] (or integrals)."""
        self.fig.clear()
        n = len(curves)
        for i, (curve, label, color) in enumerate(zip(curves, labels, colors)):
            ax = self.fig.add_subplot(n, 1, i + 1)
            ax.plot(x, curve, color=color, linewidth=1.2)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(self.title)
            if i == n - 1:
                ax.set_xlabel("x")
        self.fig.tight_layout()
        self.draw_idle()

    def clear(self):
        self.fig.clear(); self.fig.tight_layout(); self.draw_idle()


# ---------------------------------------------------------------------------
# 6) Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NN Visualizer — Loss · PCA · Derivatives · Integrals")
        self.resize(1200, 750)

        self.X, self.y = make_dataset(n_samples=200, noise=0.2)
        self.x_dense = np.linspace(-3.0, 3.0, 300).reshape(-1, 1)
        self.dx = float(self.x_dense[1] - self.x_dense[0])
        self.network = NeuralNetwork([1, 16, 16, 1])

        # Collected per iteration
        self.iterations, self.losses = [], []
        self.weight_snapshots = []    # list of flattened weight vectors
        self.last_y_dense = None      # predictions on dense grid (last iter)

        self.thread = None
        self.worker = None
        self._build_ui()

    # ----- UI -----
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: controls
        controls = QGroupBox("Configuration")
        form = QFormLayout(controls)

        self.spin_input = QSpinBox(); self.spin_input.setRange(1, 100); self.spin_input.setValue(1)
        self.spin_hidden = QSpinBox(); self.spin_hidden.setRange(1, 128); self.spin_hidden.setValue(16)
        self.spin_hidden_layers = QSpinBox(); self.spin_hidden_layers.setRange(1, 5); self.spin_hidden_layers.setValue(2)
        self.spin_output = QSpinBox(); self.spin_output.setRange(1, 100); self.spin_output.setValue(1)
        self.spin_iters = QSpinBox(); self.spin_iters.setRange(1, 100_000); self.spin_iters.setValue(300)
        self.spin_iters.setSingleStep(50)
        self.spin_samples = QSpinBox(); self.spin_samples.setRange(10, 10_000); self.spin_samples.setValue(200)
        self.spin_noise = QDoubleSpinBox(); self.spin_noise.setRange(0.0, 5.0); self.spin_noise.setValue(0.2)
        self.spin_noise.setSingleStep(0.05)

        self.combo_pca = QComboBox()
        self.combo_pca.addItems(["1", "2", "3"])
        self.combo_pca.setCurrentIndex(1)  # default 2D

        form.addRow("Input neurons:", self.spin_input)
        form.addRow("Hidden neurons:", self.spin_hidden)
        form.addRow("Hidden layers:", self.spin_hidden_layers)
        form.addRow("Output neurons:", self.spin_output)
        form.addRow("Iterations (n):", self.spin_iters)
        form.addRow("Dataset size:", self.spin_samples)
        form.addRow("Noise level:", self.spin_noise)
        form.addRow("PCA dimensions:", self.combo_pca)

        self.btn_start = QPushButton("▶  Start")
        self.btn_stop = QPushButton("■  Stop")
        self.btn_clear = QPushButton("✕  Clear")
        self.btn_stop.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start); btn_row.addWidget(self.btn_stop); btn_row.addWidget(self.btn_clear)
        form.addRow(btn_row)

        self.progress = QProgressBar()
        form.addRow(self.progress)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(140)
        form.addRow("Log:", self.log)

        # Right: tabs
        tabs = QTabWidget()
        self.loss_canvas = LossCanvas()
        self.pca_canvas = PCACanvas()
        self.deriv_canvas = FunctionCanvas("Network Output & Derivatives (1st–4th)", "y")
        self.integ_canvas = FunctionCanvas("Cumulative Integrals (1st–4th)", "∫y")

        tabs.addTab(self.loss_canvas, "Loss")
        tabs.addTab(self.pca_canvas, "PCA")
        tabs.addTab(self.deriv_canvas, "Derivatives")
        tabs.addTab(self.integ_canvas, "Integrals")

        self.lbl_current = QLabel("Current loss: —")
        self.lbl_current.setAlignment(Qt.AlignCenter)
        self.lbl_current.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")

        right = QVBoxLayout()
        right.addWidget(tabs, stretch=1)
        right.addWidget(self.lbl_current)

        root.addWidget(controls, stretch=0)
        root.addLayout(right, stretch=1)

        # Signals
        self.btn_start.clicked.connect(self.start_run)
        self.btn_stop.clicked.connect(self.stop_run)
        self.btn_clear.clicked.connect(self.clear_run)
        self.spin_samples.valueChanged.connect(self._regen_dataset)
        self.spin_noise.valueChanged.connect(self._regen_dataset)
        for s in (self.spin_input, self.spin_hidden, self.spin_hidden_layers, self.spin_output):
            s.valueChanged.connect(self._rebuild_network)

    # ----- Actions -----
    def _regen_dataset(self):
        self.X, self.y = make_dataset(self.spin_samples.value(), self.spin_noise.value())
        self._log(f"Dataset: {self.X.shape[0]} samples, noise={self.spin_noise.value():.2f}")

    def _rebuild_network(self):
        layers = [self.spin_input.value()]
        layers += [self.spin_hidden.value()] * self.spin_hidden_layers.value()
        layers.append(self.spin_output.value())
        self.network = NeuralNetwork(layers)
        self._log(f"Network: {' → '.join(map(str, layers))}")

    def _log(self, msg):
        self.log.append(msg)

    def start_run(self):
        self._rebuild_network()
        self.iterations.clear(); self.losses.clear()
        self.weight_snapshots.clear(); self.last_y_dense = None
        self.loss_canvas.clear(); self.pca_canvas.clear()
        self.deriv_canvas.clear(); self.integ_canvas.clear()

        n = self.spin_iters.value()
        self.progress.setRange(0, n); self.progress.setValue(0)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)

        self.thread = QThread()
        self.worker = Worker(self.network, self.X, self.y, self.x_dense, n)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.iteration_done.connect(self._on_iteration)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self._cleanup_thread)

        self._log(f"Starting {n} iterations...")
        self.thread.start()

    def stop_run(self):
        if self.worker:
            self.worker.stop()
        self._log("Stop requested.")

    def clear_run(self):
        self.iterations.clear(); self.losses.clear()
        self.weight_snapshots.clear(); self.last_y_dense = None
        self.loss_canvas.clear(); self.pca_canvas.clear()
        self.deriv_canvas.clear(); self.integ_canvas.clear()
        self.lbl_current.setText("Current loss: —"); self.progress.setValue(0)

    def _on_iteration(self, i, loss, payload):
        flat, y_dense = payload
        self.iterations.append(i)
        self.losses.append(loss)
        self.weight_snapshots.append(flat)
        self.last_y_dense = y_dense
        self.progress.setValue(i + 1)
        self.lbl_current.setText(f"Current loss: {loss:.6f}")

        n = self.spin_iters.value()
        if n <= 500 or (i + 1) % max(1, n // 500) == 0:
            self.loss_canvas.update(self.iterations, self.losses)

    def _on_finished(self):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        if not self.losses:
            self._log("Run cancelled."); return

        self._log(f"Done. Loss min={min(self.losses):.4f}, "
                  f"max={max(self.losses):.4f}, mean={np.mean(self.losses):.4f}")

        # --- PCA ---
        W = np.vstack(self.weight_snapshots)  # (n_iters, n_params)
        n_comp = int(self.combo_pca.currentText())
        if W.shape[0] >= n_comp and W.shape[1] >= n_comp:
            proj, eigvals = pca_project(W, n_comp)
            self.pca_canvas.update(proj, n_comp, eigvals)
            self._log(f"PCA{n_comp} computed. Top eigenvalues: {np.round(eigvals, 2)}")
        else:
            self._log(f"Not enough data for PCA{n_comp} (need ≥{n_comp} iters & params).")

        # --- Derivatives ---
        if self.last_y_dense is not None:
            f = self.last_y_dense.ravel()
            derivs = [f] + [numerical_derivative(f, self.dx, k) for k in range(1, 5)]
            labels = ["f(x)", "f'(x)", "f''(x)", "f'''(x)", "f⁽⁴⁾(x)"]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
            self.deriv_canvas.update(self.x_dense.ravel(), derivs, labels, colors)

            # --- Integrals ---
            integrals = [f] + [cumulative_integral(f, self.dx, k) for k in range(1, 5)]
            ilabels = ["f(x)", "∫f", "∫∫f", "∫∫∫f", "∫⁴f"]
            icolors = ["#1f77b4", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
            self.integ_canvas.update(self.x_dense.ravel(), integrals, ilabels, icolors)

            self._log("Derivatives & integrals computed for last network snapshot.")

    def _cleanup_thread(self):
        self.thread = None; self.worker = None


# ---------------------------------------------------------------------------
# 7) Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()