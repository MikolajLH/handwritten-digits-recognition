import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import cv2
from neural_network import NeuralNetwork, relu, softmax


class PaintApp:
    def __init__(self, root, model_path: str):
        self.root = root
        self.root.title("Handwritten digits recognition demo")
        
        self.canvas = tk.Canvas(root, width=400, height=400, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.img = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.img)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.check_button = tk.Button(root, text="Check", command=self.check_digit)
        self.check_button.pack()
        
        self.x0 = None
        self.y0 = None

        self.model = NeuralNetwork()
        self.model.add_input_layer(784)
        self.model.add_hidden_layer(40, relu)
        self.model.add_output_layer(10, softmax)
        self.model.load_from_nn(model_path)

    def reset(self, event):
        self.x0 = None
        self.y0 = None

    def paint(self, event):
        x1, y1 = event.x, event.y
        if self.x0 and self.y0:
            self.canvas.create_line(self.x0, self.y0, x1, y1, width=6)
            self.draw.line([self.x0, self.y0, x1, y1], fill='black',width=6)

        self.x0 = x1
        self.y0 = y1

    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.img = Image.new("RGB", (400, 400), "white")
        self.draw = ImageDraw.Draw(self.img)

    def check_digit(self):
        I = np.array(self.img)
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        I = I.astype(float) / 255
        I = -I + 1

        c_row = np.nonzero(np.sum(I > 0.7, 0))
        c_col = np.nonzero(np.sum(I > 0.7, 1))

        min_x, max_x = c_row[0][0], c_row[0][-1]
        min_y, max_y = c_col[0][0], c_col[0][-1]
        
        dx = 40
        dy = 40
        min_x = max(0, min_x - dx)
        max_x = min(400, max_x + dx)

        min_y = max(0, min_y - dy)
        max_y = min(400, max_y + dy)

        I = I[min_y:max_y,min_x:max_x]

        I = cv2.dilate(I, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        I = cv2.dilate(I, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        kernel = np.ones((2,2)) / 4
        I = cv2.filter2D(I, -1, kernel)
        I = cv2.filter2D(I, -1, kernel)


        I = cv2.resize(I, (28, 28))

        for r in range(28):
            for c in range(28):
                print(float_to_char(1 - I[r,c]), end='')
            print()

        X = I.flatten().astype(np.float64)
        Y = self.model(X)
        d = np.argmax(Y)
        print(f"{d}: {100 * Y[d]:.2f}%")

        print("|", end=' ')
        for i, p in enumerate(Y):
            print(f"{i}: {100 * p:.2f}",end=' | ')
        print()
                    


def float_to_char(v):
    alfa = bytes([219, 178, 177, 176]).decode('cp437')
    if v <= 0.30:
        return alfa[0]
    if v <= 0.50:
        return alfa[1]
    if v <= 0.70:
        return alfa[2]
    if v <= 0.90:
        return alfa[3]
    return ' '


def main():
    root = tk.Tk()
    app = PaintApp(root, 'models/nn_H500_B100_MB5.nn')
    root.mainloop()

if __name__ == "__main__":
    main()