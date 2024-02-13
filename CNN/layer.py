import numpy as np

class ConvLayer():
    def __init__(self, num_kernels, kernel_size, input_shape, padding = 0, stride = 1):
        input_width, input_height, input_depth = input_shape
        self.input_depth = input_depth
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.input_shape = input_shape # 255, 255, 1
        self.output_shape = (num_kernels, ((input_width - kernel_size + 2*padding)/stride)+1, ((input_height - kernel_size + 2 * padding)/stride) + 1)
        self.kernels = np.random.randn(num_kernels, kernel_size, kernel_size, input_depth)
        
    def convolve2d(self, images):
        output = np.zeros(self.output_shape)
        for image in images:
            self.image = image
            for i in range(self.num_kernels):
                for j in range(self.input_depth):
                    output[i, j] = sum(self.kernels *image[i: i+self.kernel_size, j: j + self.kernel_size])
        return output
