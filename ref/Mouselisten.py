import pynput
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener, Key

def mouse_click(x, y, button, pressed):
    print(x, y, button, pressed)
    if pressed and (button == pynput.mouse.Button.x2 or button == Key.shift):
        print('Pressed X2 or Shift')

def on_press(key):
    if key == Key.shift:
        print('Shift key pressed')

def on_release(key):
    if key == Key.shift:
        print('Shift key released')

with MouseListener(on_click=mouse_click) as mouse_listener:
    with KeyboardListener(on_press=on_press, on_release=on_release) as keyboard_listener:
        mouse_listener.join()
