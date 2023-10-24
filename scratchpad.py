#%%

import numpy as np
import math



# %%
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# %%


a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()


learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + x ** 3
    loss = np.square(y_pred - y).sum()

    if t % 100 == 0:
        print(t,loss)


    grad_y_pred = 2.0 * (y_pred - y )
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')



# %%
import torch
import math 


dtype = torch.float 
device = torch.device("cpu")
# device = torch.device("cuda:0")


x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype, device=device)
y = torch.sin(x)


a = torch.randn((), dtype=dtype, device=device)
b = torch.randn((), dtype=dtype, device=device)
c = torch.randn((), dtype=dtype, device=device)
d = torch.randn((), dtype=dtype, device=device)


learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + x ** 3
    loss = np.square(y_pred - y).sum()

    if t % 100 == 0:
        print(t,loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()



    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d



    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
# %%
import torch
import math 


dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)
y = torch.sin(x)


a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)



learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b * x + c * x ** 2 + x ** 3
    loss = np.square(y_pred - y).sum()

    if t % 100 == 0:
        print(t,loss)


    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

    a.grad = None
    b.grad = None
    c.grad = None
    d.grad = None

    print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

#%%
import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)


p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-6
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

for t in range(2000):
    
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)

    if t % 100 == 99:
        print(t, loss.item())

    model.zero_grad()
    loss.backward()

    optimizer.step()


linear_layer = model[0]

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

# %%



