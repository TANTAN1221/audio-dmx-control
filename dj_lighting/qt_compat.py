"""
Qt compatibility layer: PyQt6 preferred, fallback to PyQt5.
Keeps enum differences behind small helpers.
"""
from __future__ import annotations

QT6: bool

try:
    from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer, QRect, QRectF, QPoint, QPointF, QSize, QEvent, QMimeData
    from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QPolygonF, QRadialGradient, QLinearGradient, QBrush, QDrag, QAction
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QToolButton, QMenu,
        QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QSizePolicy, QButtonGroup,
        QSlider, QLineEdit, QMessageBox, QFileDialog, QAbstractButton, QColorDialog,
        QBoxLayout
    )
    QT6 = True
except Exception:  # pragma: no cover
    from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer, QRect, QRectF, QPoint, QPointF, QSize, QEvent, QMimeData
    from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QPolygonF, QRadialGradient, QLinearGradient, QBrush, QDrag, QAction
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QToolButton, QMenu,
        QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QSizePolicy, QButtonGroup,
        QSlider, QLineEdit, QMessageBox, QFileDialog, QAbstractButton, QColorDialog,
        QBoxLayout
    )
    QT6 = False


def hbox_dir_left_to_right():
    return QBoxLayout.Direction.LeftToRight if QT6 else QBoxLayout.LeftToRight


def hbox_dir_top_to_bottom():
    return QBoxLayout.Direction.TopToBottom if QT6 else QBoxLayout.TopToBottom


def align_hcenter():
    return Qt.AlignmentFlag.AlignHCenter if QT6 else Qt.AlignHCenter


def no_pen():
    return Qt.PenStyle.NoPen if QT6 else Qt.NoPen


def orientation_vertical():
    return Qt.Orientation.Vertical if QT6 else Qt.Vertical


def orientation_horizontal():
    return Qt.Orientation.Horizontal if QT6 else Qt.Horizontal


def cursor_pointing_hand():
    return Qt.CursorShape.PointingHandCursor if QT6 else Qt.PointingHandCursor


def renderhint_antialiasing():
    return QPainter.RenderHint.Antialiasing if QT6 else QPainter.Antialiasing
