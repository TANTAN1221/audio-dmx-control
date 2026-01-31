DARK_QSS = """
* { color: #E6E6E6; font-family: Segoe UI, Arial; }
QMainWindow { background: #15171A; }
QWidget { background: #15171A; }

QFrame#Panel {
    background: #1C1F23;
    border: 1px solid #2E333A;
    border-radius: 12px;
}

QLabel#PanelTitle {
    background: transparent;
    color: #D6D9DE;
    font-weight: 700;
    padding: 2px 2px;
}

QLabel#Subtle { color: #A9B0BA; }

QToolButton, QPushButton {
    background: #23272D;
    border: 1px solid #323844;
    border-radius: 12px;
    padding: 10px 14px;
}
QToolButton:hover, QPushButton:hover { border-color: #3C4452; }
QToolButton:checked {
    background: #2D3642;
    border-color: #5B89B8;
}

QPushButton#Primary { border-color: #5B89B8; }
QPushButton#Danger  { border-color: #B85B5B; }

QLineEdit {
    background: #121417;
    border: 1px solid #323844;
    border-radius: 10px;
    padding: 8px 10px;
}

/* DJ Fader (vertical) */
QSlider::groove:vertical {
    background: #0E1013;
    border: 1px solid #2E333A;
    width: 12px;
    border-radius: 6px;
}
QSlider::sub-page:vertical {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #7FB0E3, stop:1 #3E6EA6);
    border-radius: 6px;
}
QSlider::add-page:vertical { background: #0E1013; border-radius: 6px; }
QSlider::handle:vertical {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #3B424D, stop:1 #222830);
    border: 1px solid #3A4350;
    height: 32px;
    margin: 0 -10px;
    border-radius: 10px;
}

/* Player scrubber (horizontal) */
QSlider::groove:horizontal {
    background: #0E1013;
    border: 1px solid #2E333A;
    height: 10px;
    border-radius: 5px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7FB0E3, stop:1 #3E6EA6);
    border-radius: 5px;
}
QSlider::add-page:horizontal { background: #0E1013; border-radius: 5px; }
QSlider::handle:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #E6EBF2, stop:1 #AAB2BC);
    border: 1px solid #3A4350;
    width: 18px;
    margin: -6px 0;
    border-radius: 9px;
}

/* Color button */
QPushButton#ColorSwatch {
    border-radius: 12px;
    border: 1px solid #323844;
    padding: 12px;
    text-align: left;
}
QPushButton#ColorSwatch:hover { border-color: #3C4452; }
"""
