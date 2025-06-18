import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import device

#scalar
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
print(scalar.item())

#vector
vector = torch.tensor([7,7])
print(vector)
print(vector.ndim)
print(vector.shape)

#matrix
matrix = torch.tensor([[7, 8], [9, 10]])
print(matrix)
print(matrix.ndim)
print(matrix[0])
print(matrix.shape)

#Tensor
TENSOR = torch.tensor([[[[1,2,3,3],
                        [3,6,9,4],
                        [2,4,5,4]],[[1,2,3,5],
                        [3,6,9,4],
                        [2,4,5,4]]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)
print(TENSOR[0])

#Random tensor
random_tensor = torch.rand(3,4)
print(random_tensor)
print(random_tensor.ndim)

#random tensor with similar shape to image tensor
random_image_size_tensor = torch.rand(size=(224,224,3)) # height width colour channel
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)

#zeros and ones
zeros = torch.zeros(size =(3,4))
print(zeros)

ones = torch.ones(size=(3,4))
print(ones)

print(ones.dtype)

# creating range of tensors and tensor like
one_to_ten = torch.arange(start= 1,end= 11, step= 1)
print(one_to_ten)

#tensors like
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)

#tensor datatype
# 3 big error 1. Tensors not right datatype 2. Tensor not right shap 3. Tensor not on the right device

float_32_tensor = torch.tensor([3.0,6.0,9.0],
                               dtype=None,  # what datatype is tensor (float32 or float16)
                               device=None, # what device is ur tensor on
                               requires_grad=False) # whether or not to track gradients with this tensor operation
print(float_32_tensor)
print(float_32_tensor.dtype)

float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)

multiTensor = float_16_tensor * float_32_tensor
print(multiTensor.dtype)
print(multiTensor)

# Getting info from tensors (tensor attributes)
# if
# 1. Tensors not right datatype use tensor.dtype
# 2. Tensor not right shap use tensor.shape
# 3. Tensor not on the right device use tenser.device

some_tensor = torch.rand(3,4)
print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device of tensor: {some_tensor.device}")

# manipulating tensor (tensor operation)
# add sub mult division
# matrix mult

# add 10
tensor = torch.tensor([1,2,3])
print(tensor +10)
print(torch.add(tensor,10))
# multi 10
print(tensor * 10)
print(torch.mul(tensor,10))
# sub 10
print(tensor - 10)
print(tensor.sub(10))

#matrix multi
print(tensor * tensor)
print(torch.matmul(tensor,tensor))

#there are two rules that performing matrix multi needs to satisfy
# 1. the inner dimensions mush match
# (3,2) matmul (3,2) won't work (ex: torch.matmul(torch.rand(3,2),torch.rand(3,2)))
# (2,3) matmul (3,2) will work
# (3,2) matmul (2,3) will work
# 2. resulting matrix has the shape of the "outer dimensions" :
# (2,3) matmul (3,2) -> (2,2)
# (3,2) matmul (2,3) -> (3,3)
print(torch.matmul(torch.rand(2,3),torch.rand(3,2)))

tensor_A = torch.tensor([[3,4],
                   [1,2],
                   [4,5]])
tensor_B = torch.tensor([[1,3],[3,4],[2,1]])

print(tensor_B.T)
print(tensor_B)
print(tensor_A.mm(tensor_B.T))

#finding the min max mean sum (tensor aggregation)
x = torch.arange(0,100,10)
print(x.dtype)
print(x.min())
print(x.max())
print(torch.mean(x.type(torch.float32)))
print(x.type(torch.float32).mean())
print(x.sum())

#find the pos min and max
#find the max val and return its index
print(x.argmax())
#find the min val and return its index
print(x.argmin())

# reshaping stacking squeezing unsqueezing tensor
# reshape - reshape an input tensor to a defined shape
# view - return a view of an input tensor of certain shape but keep the same memory as the original tensor
# stack - combine multi tensor on top of each other (vstack) or side by side (hstack)
# squeeze - removes all 1 dimensions from a tensor
# unsqeeze - add a 1 dimensions to a target tensor
# permute - return a view of the input with dimensions permuted (swapped) in a certain way
a = torch.arange(1,10)

# add extra dimension
a_reshape_19 = a.reshape(1,9)
print(a_reshape_19)
print(a_reshape_19.shape)
a_reshape_91 = a.reshape(9,1)
print(a_reshape_91)
print(a_reshape_91.shape)

# change the view
b = a.view(1,9)
print(b)
print(b.shape)

# changing b changes a (because a view of tensor shares the same memory as the original input)
b[:, 0] = 5
print(b)
print(a)

# stack tensor on top of each other
a_stack = torch.stack([a,a,a,a], dim=0)
print(a_stack)
a_stack = torch.stack([a,a,a,a], dim=1)
print(a_stack)

#torch.squeeze() - remove all single dimensions from a target tensor
a_squeeze = torch.squeeze(a)
print(a_squeeze)
print(a_squeeze.shape)

#torch.unsqeezed - adds a single dimension to a target tensor at a specific dim
a_unsqueeze = torch.unsqueeze(a_squeeze,dim=0)
print(a_unsqueeze)
print(a_unsqueeze.shape)

a_unsqueeze = torch.unsqueeze(a_squeeze,dim=1)
print(a_unsqueeze)
print(a_unsqueeze.shape)

# torch.permute - rearranges the dimensions of a target tensor in a specified order
a_original = torch.rand(size=(224,225,3)) # h,w,colour_channel
print(a_original.shape)
a_permuted = a_original.permute(2,0,1) #shift axis 0->1, 1->2, 2->0
print(a_permuted.shape) # colour_channel, h , w

# indexing (select data from tensors)
# indexing with pytorch is similar to indexing with NumPy

x = torch.arange(1,10).reshape(1,3,3)
print(x)
print(x.shape)

#index x
print(x[0])
#index x dim = 1
print(x[0][0])
# index x last dimension
print(x[0][0][0])

# use ":" to select all of a target dimension
print(x[:,0])
# get all val of 0th and 1st dimensions but only index 1 of 2nd dimension
print(x[:,:,1])

#get all val of the 0 dimensions but only the 1 index val of 1st and 2nd dimension
print(x[:,1,1])

#get index 0 of 0th and 1st dimension and all val of 2nd dim
print(x[0,0,:])

#index on x to return 9
print(x[0][2][2])

#index on x to return 3,6,9
print(x[:,:,2])

#Pytorch tensors & NumPy
#Numpy arr to tensor

arr = np.arange(1.0,8.0)
tensor = torch.from_numpy(arr) # when numpy to pytorch the type of pytorch will be float64
print(arr)
print(tensor)
print(arr.dtype)

# change val of arr
arr= arr+1
print(arr)
print(tensor)

# tensor to numpy
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(numpy_tensor)
print(numpy_tensor.dtype)
print(tensor)

#change tensor
tensor = tensor+ 1
print(numpy_tensor)
print(tensor)

#reproducbility (trying to take random out of random)

random_tensor_A=torch.rand(3,4)
random_tensor_B=torch.rand(3,4)
print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A==random_tensor_B)

# create some random but reproducible tensors
# set random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3,4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C==random_tensor_D)

# run tensor and pytorch on gpu

# check for gpu access for pytorch
print(torch.cuda.is_available())

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#count number of device
print(torch.cuda.device_count())

# put tensors and models on the gpu
tensor = torch.tensor([1,2,3])

#tensor not on gpu
print(tensor, tensor.device)

#move tensor to gpu (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)
# move tensor back to cpu
# if tensor is on gpu then it can not transform to numpy

tensor_back_on_cpu = tensor_on_gpu.cpu()
numpy_tensor = tensor_back_on_cpu.numpy()
print(numpy_tensor)

#exercies

# 1. Documentation reading
# 2. Create a random tensor with shape (7, 7).
random_tensor = torch.rand(7,7) #torch.rand(size=(7,7))
print(random_tensor)
# 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7)
# (hint: you may have to transpose the second tensor).
tensor_1 = torch.rand(1,7)
tensor_mul = random_tensor.mm(tensor_1.T)
print(tensor_mul)
# 4. Set the random seed to 0 and do 2 & 3 over again.
RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
ran_tensor = torch.rand(7,7)
tensor_2 = torch.rand(1,7)
tensor_mul = ran_tensor.mm(tensor_2.T)
print(tensor_mul)
print(tensor_mul.shape)
#5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent?
# (hint: you'll need to look into the documentation for torch.cuda for this one)
torch.cuda.manual_seed(1234)
#6. Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this).
# Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).
torch.manual_seed(1234)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
tensor_1 = torch.rand(2,3).to(device)
tensor_2 = torch.rand(2,3).to(device)
print(tensor_1)
print(tensor_2)
#7. Perform a matrix multiplication on the tensors you created in 6
# (again, you may have to adjust the shapes of one of the tensors).
tensor_mul = tensor_2.mm(tensor_1.T)
print(tensor_mul, tensor_mul.shape)
# 8. Find the maximum and minimum values of the output of 7.
print(tensor_mul.max())
print(tensor_mul.min())
# 9. Find the maximum and minimum index values of the output of 7.
print(tensor_mul.argmax())
print(tensor_mul.argmin())
# 10.Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with
# all the 1 dimensions removed to be left with a tensor of shape (10).
# Set the seed to 7 when you create it and print out the first tensor
# and it's shape as well as the second tensor and it's shape.
torch.manual_seed(7)
rand_tensor = torch.rand(1,1,1,10)
squeeze_tensor = rand_tensor.squeeze()
print(rand_tensor)
print(rand_tensor.shape)
print(squeeze_tensor)
print(squeeze_tensor.shape)