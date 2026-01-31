from __future__ import annotations

from dj_lighting.qt_compat import QApplication, QT6
from dj_lighting.style import DARK_QSS
from dj_lighting.main_window import MainWindow


def main():
    app = QApplication([])
    app.setStyleSheet(DARK_QSS)
    w = MainWindow()
    w.show()  # MainWindow will lock itself to maximized.
    if QT6:
        app.exec()
    else:
        app.exec_()


if __name__ == "__main__":
    main()
