import numpy as np
import torch
import torch.nn as nn

# relationship: n=s×(m−1)+k−2p
def deconv(input, kernel, padding=0,stride=1):
    input_height,input_width=input.shape
    k_height,k_width=kernel.shape
    output_height,output_width=stride*(input_height-1)+k_height,stride*(input_width-1)+k_width
    output=np.zeros((output_height,output_width))
    for i in range(input_height):
        for j in range(input_width):
            output[i*stride:i*stride+k_height,j*stride:j*stride+k_width]+=input[i,j]*kernel

    output=output[padding:output_height-padding,padding:output_width-padding]
    return output

def test_deconv(num_tests, max_dim=10, max_val=10, tol=1e-6):
    for i in range(num_tests):
        # Randomly generate dimensions and values for input and kernel
        input_dim = np.random.randint(1, max_dim)
        kernel_dim = np.random.randint(1, max_dim)
        stride = np.random.randint(1, 4)
        padding = 0 if kernel_dim < 3 else np.random.randint(0, kernel_dim // 2)


        # Create random input and kernel
        input_np = np.random.randint(max_val, size=(input_dim, input_dim))
        kernel_np = np.random.randint(max_val, size=(kernel_dim, kernel_dim))

        # Run your deconv function
        output_np = deconv(input_np, kernel_np, padding, stride)

        # Prepare tensors for PyTorch
        input_torch = torch.FloatTensor(input_np).unsqueeze(0).unsqueeze(0)
        kernel_torch = torch.FloatTensor(kernel_np).unsqueeze(0).unsqueeze(0)

        # Run PyTorch's ConvTranspose2D
        deconv_layer = nn.ConvTranspose2d(1, 1, kernel_dim, stride=stride, padding=padding, bias=False)
        deconv_layer.weight.data = kernel_torch
        output_torch = deconv_layer(input_torch).squeeze().detach().numpy()

        # Check if the outputs are close enough
        if not np.allclose(output_np, output_torch, atol=tol):
            print(f"Test {i + 1} failed!")
            print(f"Input: {input_np}")
            print(f"Kernel: {kernel_np}")
            print(f"Your output: {output_np}")
            print(f"PyTorch output: {output_torch}")
            return

    print(f"All {num_tests} tests passed!")


# Run the test with 100 random test cases
test_deconv(100)