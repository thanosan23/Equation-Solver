import torch
import numpy as np

# inputs
x = torch.from_numpy(np.array([0, 1, 2], dtype=np.float32))
y = torch.from_numpy(np.array([0, 2, 4], dtype=np.float32))

x = x.reshape((1, -1))
y = y.reshape((1, -1))

# set up equation
roots = int(input("Enter the root of the equation: "))

coefficients = torch.rand(roots+1, requires_grad=True)
optimizer = torch.optim.Adam([coefficients], lr=0.001)
criterion = torch.nn.MSELoss()

# run through autograd
for i in range(5000):
    optimizer.zero_grad()
    total = torch.empty((1, x.shape[1]), requires_grad=True)
    for i, coefficient in enumerate(coefficients):
        exponent = roots-i
        val = (x**exponent)
        val = val * coefficient
        total = total + val
    loss = criterion(y, total)
    loss.backward()
    optimizer.step()

coefficients = coefficients.tolist()

# output the equation
equation = "y = "
for i, coefficient in enumerate(coefficients):
    equation += f"{coefficient:.2f}{'x^' + str(roots-i) if i != roots else ''} {'' if i == len(coefficients)-1 else '+ '}"

print(equation)
